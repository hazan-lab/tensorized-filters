import numpy as np
import torch
import torch.nn.functional as F
import tiktoken
import logging
import json
from time import time
from safetensors import safe_open
from model import FlashSTU
from config import FlashSTUConfig
from tensorized_filters.utils.filters import get_spectral_filters

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_hankel(seq_len: int, use_hankel_L: bool = False) -> np.ndarray:
    entries = np.arange(1, seq_len + 1, dtype=np.float64)
    i_plus_j = entries[:, None] + entries[None, :]

    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    elif not use_hankel_L:
        Z = 2.0 / (i_plus_j**3 - i_plus_j)
    else:
        raise ValueError("use_hankel_L must be a boolean")

    return Z

def get_spectral_filters(
    seq_len: int, 
    K: int, 
    use_hankel_L: bool = False, 
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert torch.cuda.is_available(), "CUDA is required."
    Z = get_hankel(seq_len, use_hankel_L)
    sigma, phi = np.linalg.eigh(Z)
    sigma_k, phi_k = sigma[-K:], phi[:, -K:]
    phi_k *= sigma_k ** 0.25
    filters = torch.from_numpy(phi_k)
    return filters.to(device=device, dtype=dtype)

# Load the checkpoint
print("Loading the checkpoint...")
start_time = time()
state_dict = {}
with safe_open(
    "/scratch/gpfs/mn4560/hazan-lab/hazan_lab/tensorized_filters/models/flash_stu_2/log/model_19073.safetensors",
    framework="pt",
    device="cuda",
) as f:
    for k in f.keys():
        state_dict[k] = f.get_tensor(k)

print(f"Successfully loaded the checkpoint in {time() - start_time:.2f} seconds")

# Set precision for matrix multiplication
torch.set_float32_matmul_precision("high")

# Load model configurations from JSON file
with open("config.json", "r") as file:
    config = json.load(file)

# Extract model configurations
n_embd = config["n_embd"]
n_heads = config["n_heads"]
n_layers = config["n_layers"]
seq_len = config["seq_len"]
window_size = config["window_size"]
vocab_size = config["vocab_size"]
mlp_scale = config["mlp_scale"]
bias = config["bias"]
dropout = config["dropout"]
num_eigh = config["num_eigh"]
use_hankel_L = config["use_hankel_L"]
use_flash_fft = config["use_flash_fft"]
use_approx = config["use_approx"]
use_attn = config["use_attn"]
softcap = config["softcap"]

# Model setup
config = FlashSTUConfig(
    n_embd=n_embd,
    n_heads=n_heads,
    n_layers=n_layers,
    seq_len=seq_len,
    window_size=window_size,
    vocab_size=vocab_size,
    mlp_scale=mlp_scale,
    bias=bias,
    dropout=dropout,
    num_eigh=num_eigh,
    use_hankel_L=use_hankel_L,
    use_flash_fft=use_flash_fft,
    use_approx=use_approx,
    use_attn=use_attn,
    softcap=softcap,
    torch_dtype=getattr(torch, config["torch_dtype"]),
)
phi = get_spectral_filters(seq_len, num_eigh, use_hankel_L, device, torch.float32)
model = FlashSTU(config, phi)

# Load state dictionary into the model
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Prepare tokenizer
tokenizer = tiktoken.get_encoding("o200k_base")

def generate_text(
    model, tokenizer, prompt, num_return_sequences=4, max_length=1024, device="cuda", temperature=1.0, top_k=50
):
    model.eval()
    tokens = torch.tensor([tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})], device=device)
    tokens = tokens.repeat(num_return_sequences, 1)

    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(1337)

    eos_token_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    with torch.no_grad():
        for _ in range(max_length - tokens.size(1)):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(tokens)
                logits = logits[:, -1, :]  # Get logits for the last token

                # Apply temperature scaling if temperature > 0
                if temperature > 0:
                    logits = logits / temperature

            probs = F.softmax(logits, dim=-1)  # Compute probabilities

            # Top-K sampling: set all probabilities outside the top K to 0
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(top_k_probs, 1, generator=sample_rng)
            next_token = torch.gather(top_k_indices, -1, ix)
            tokens = torch.cat((tokens, next_token), dim=1)

            # Break if EOS token is generated
            if (next_token == eos_token_id).any():
                break

    generated_sequences = []
    for i in range(num_return_sequences):
        decoded = tokenizer.decode(tokens[i].tolist())
        generated_sequences.append(decoded)

    return generated_sequences

# Example prompts for generation
prompts = [
    "The future of artificial intelligence is",
    "In the year 2050, the world will",
    "The most important scientific discovery of the 21st century is",
    "If I could change one thing about the education system, it would be",
    "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.",
]

for prompt in prompts:
    print(f"\nGenerating text for prompt: '{prompt}'\n")
    generated_texts = generate_text(model, tokenizer, prompt, num_return_sequences=2, max_length=512)
    for i, text in enumerate(generated_texts):
        print(f"Sample {i + 1}: {text}\n")
