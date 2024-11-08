import torch
import torch.nn.functional as F
import tiktoken
import logging
import json
import numpy as np
from time import time
from distributed import setup_distributed
from safetensors import safe_open
from dataloader import DistributedDataloader
from flash_stu import FlashSTU, FlashSTUConfig, get_spectral_filters
from torchmetrics.text import BLEUScore, Perplexity
from nltk.translate.bleu_score import sentence_bleu

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss as CrossEntropyLoss
except ImportError as e:
    logger.warning(f"Unable to import Triton-based cross entropy loss: {e}. Falling back to PyTorch implementation.")
    from torch.nn import CrossEntropyLoss
    
#####################################################################################################################

# Distributed setup
device, local_rank, rank, world_size, main_process = setup_distributed(seed=1337)
logger.info(f'Using device: {device}')

# Load the checkpoint
logger.info('Loading the checkpoint...')
start_time = time()
state_dict = {}
# with safe_open('/scratch/gpfs/mn4560/flash-stu/log/final_flash_stu_learnable/model_19000.safetensors', framework="pt", device='cuda') as f:
with safe_open('/scratch/gpfs/mn4560/flash-stu/log/final_flash_stu/model_19000.safetensors', framework="pt", device='cuda') as f:
    for k in f.keys():
        state_dict[k] = f.get_tensor(k)

logger.info(f'Successfully loaded the checkpoint in {time() - start_time:.2f} seconds')

# Set precision for matrix multiplication
torch.set_float32_matmul_precision("high")

#####################################################################################################################

# Load model configurations from JSON file
with open("config.json", "r") as file:
    config  = json.load(file)

# Extract model and training configurations
n_embd             = config['n_embd']
n_heads            = config['n_heads']
n_layers           = config['n_layers']
seq_len            = config['seq_len']
window_size        = config['window_size']
vocab_size         = 200064
mlp_scale          = config['mlp_scale']
bias               = config['bias']
dropout            = config['dropout']
num_eigh           = config['num_eigh']
use_hankel_L       = config['use_hankel_L']
use_flash_fft      = config['use_flash_fft']
use_approx         = config['use_approx']
use_attn           = config['use_attn']
softcap            = config['softcap']
torch_compile      = config['torch_compile']
dilation           = config['dilation']
warmup_steps       = config['warmup_steps']
eval_period        = config['eval_period']
save_period        = config['save_period']
num_epochs         = config['num_epochs']
max_lr             = config['max_lr']
min_lr             = config['min_lr']
max_norm           = config['max_norm']
global_bsz         = config['global_bsz']
bsz                = config['bsz']
fsdp               = config['fsdp']
ddp                = config['ddp']
mixed_precision    = config['mixed_precision']
torch_dtype        = getattr(torch, config['torch_dtype'])
use_cpu_offload    = config['use_cpu_offload']
sharding_strategy  = config['sharding_strategy']
auto_wrap_policy   = config['auto_wrap_policy']
backward_prefetch  = config['backward_prefetch']
forward_prefetch   = config['forward_prefetch']
sync_module_states = config['sync_module_states']
use_orig_params    = config['use_orig_params']
device_id          = config['device_id']
precision          = config['precision']
fsdp_modules       = config['fsdp_modules']
use_activation_checkpointing = config['use_activation_checkpointing']

# Validation checks
assert (
    global_bsz % (bsz * seq_len * world_size) == 0
), f"global_bsz ({global_bsz}) must be divisible by bsz * seq_len * world_size ({bsz * seq_len * world_size}), got {global_bsz % (bsz * seq_len * world_size)}"
gradient_accumulation_steps = global_bsz // (bsz * seq_len * world_size)
assert not (fsdp and ddp), "FSDP and DDP are both enabled, which is not allowed"

distributed = (fsdp or ddp) and world_size > 1
cache_enabled = not ddp

if main_process:
    logger.info(f"Training config: {config}\n")

if world_size == 1 and fsdp:
    if main_process:
        logger.info("World size is 1, disabling sharding.")
    sharding_strategy = "no_shard"

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
    torch_dtype=torch_dtype,
)
phi = get_spectral_filters(seq_len, num_eigh, use_hankel_L, device, torch_dtype)
model = FlashSTU(config, phi)

# Load state dictionary into the model
model.load_state_dict(state_dict)
model.to(device)
model.eval()

#####################################################################################################################

# Prepare tokenizer and prompt
tokenizer = tiktoken.get_encoding('o200k_base')
loss_fn = CrossEntropyLoss()
dataset = "data/fineweb-edu-10B"
val_loader = DistributedDataloader(
    bsz=bsz,
    seq_len=seq_len, 
    rank=rank, 
    world_size=world_size, 
    dataset=dataset, 
    split="val", 
    main_process=main_process,
)
val_steps = 2

# Initialize metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
perplexity = Perplexity().to(device)
bleu = BLEUScore().to(device)

def calculate_metrics(preds, targets):
    # Ensure inputs are on the correct device
    preds = preds.to(device)
    targets = targets.to(device)
    
    # Perplexity
    perp = perplexity(preds, targets)
    
    # BLEU Score
    pred_texts = [tokenizer.decode(pred) for pred in preds.argmax(dim=-1).cpu().numpy()]
    target_texts = [tokenizer.decode(target) for target in targets.cpu().numpy()]
    bleu_score = bleu(pred_texts, [[text] for text in target_texts])
    
    return perp, bleu_score

# The rest of the code remains the same
def validate(model, val_loader, tokenizer, device, loss_fn, val_steps):
    model.eval()
    val_loss = 0
    all_perplexities = []
    all_bleu_scores = []

    with torch.no_grad():
        for step, batch in zip(range(val_steps), val_loader, strict=False):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                preds = model(inputs)

            loss = loss_fn(preds.flatten(0, 1), targets.flatten(0, 1))
            loss = loss / val_steps
            val_loss += loss.detach()

            perp, bleu_score = calculate_metrics(preds, targets)
            all_perplexities.append(perp.item())
            all_bleu_scores.append(bleu_score.item())

            pred_ids = torch.argmax(preds, dim=-1).cpu().tolist()
            target_ids = targets.cpu().tolist()

            pred_texts = [tokenizer.decode(pred_ids[i]) for i in range(len(pred_ids))]
            target_texts = [tokenizer.decode(target_ids[i]) for i in range(len(target_ids))]

            for i in range(min(len(pred_texts), 2)):
                print(f"Prediction {i + 1}: {pred_texts[i][:100]}...")
                print(f"Target {i + 1}: {target_texts[i][:100]}...")
                print("-" * 30)

        print(f'Validation loss: {val_loss.item()}')
        print(f'Average Perplexity: {np.mean(all_perplexities)}')
        print(f'Average BLEU Score: {np.mean(all_bleu_scores)}')

    return val_loss.item()

def generate_text(model, tokenizer, prompt, num_return_sequences=4, max_length=32, device='cuda', temperature=1.0, top_k=50):
    model.eval()
    tokens = torch.tensor([tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})], device=device)
    tokens = tokens.repeat(num_return_sequences, 1)
    
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)
    
    eos_token_id = tokenizer.encode('<|endoftext|>', allowed_special={"<|endoftext|>"})[0]
    
    with torch.no_grad():
        for _ in range(max_length - tokens.size(1)):
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
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

# Perform validation
val_loss = validate(model, val_loader, tokenizer, device, loss_fn, val_steps)

# Generate text
prompts = [
    "The future of artificial intelligence is",
    "In the year 2050, the world will",
    "The most important scientific discovery of the 21st century is",
    "If I could change one thing about the education system, it would be",
    "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
]

for prompt in prompts:
    print(f"\nGenerating text for prompt: '{prompt}'\n")
    generated_texts = generate_text(model, tokenizer, prompt, num_return_sequences=2, max_length=512)
    for i, text in enumerate(generated_texts):
        print(f"Sample {i + 1}: {text}\n")

# Evaluate generated text
reference_texts = [
    "The future of artificial intelligence is bright and full of potential, with advancements in machine learning and neural networks paving the way for more sophisticated AI systems.",
    "In the year 2050, the world will have made significant progress in addressing climate change, with renewable energy sources becoming the primary means of power generation globally.",
    "The most important scientific discovery of the 21st century is the development of CRISPR gene-editing technology, which has revolutionized our ability to modify DNA and treat genetic diseases.",
    "If I could change one thing about the education system, it would be to focus more on practical skills and critical thinking rather than rote memorization, preparing students for the challenges of the modern world."
]

for gen_text, ref_text in zip(generated_texts, reference_texts):
    bleu_score = sentence_bleu([ref_text.split()], gen_text.split())
    
    print(f"BLEU Score: {bleu_score}")
    print("-" * 30)
