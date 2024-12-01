import math

import numpy as np
import torch

from tensorized_filters.utils.logger import logger
from tensorized_filters.utils.hankel import get_hankel


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

def compute_dimensions(n: int) -> tuple[int, int, int]:
    if n <= 2:
        raise ValueError("n must be greater than 2")

    T_prime = (math.ceil(math.sqrt(n - 2)))**2 + 2
    sqrt_T_prime = math.ceil(math.sqrt(T_prime - 2))
    k_max = sqrt_T_prime
    return T_prime, sqrt_T_prime, k_max

def get_tensorized_spectral_filters_explicit(n: int, k: int, device: torch.device) -> torch.Tensor:
    T_prime, sqrt_T_prime, k_max = compute_dimensions(n)
    k = min(k, k_max)

    Z = get_hankel(sqrt_T_prime).to(device)
    sigma, phi = torch.linalg.eigh(Z)
    sigma_k = sigma[-k:]
    phi_k = phi[:, -k:]

    result = torch.zeros(sqrt_T_prime * sqrt_T_prime, device=device)
    
    for i in range(k):
        for j in range(k):
            phi_i = phi_k[:, i] * (sigma_k[i] ** 0.25)
            phi_j = phi_k[:, j] * (sigma_k[j] ** 0.25)
            kron = torch.kron(phi_i, phi_j)
            result += kron
            
    return result


def get_tensorized_spectral_filters(
    n: int = 8192,
    k: int = 24,
    use_hankel_L: bool = False,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Compute tensorized spectral filters for given sequence length and filter count.

    Args:
        n: Sequence length
        k: Number of filters
        use_hankel_L: Hankel_main ⊗ Hankel_L? Default is Hankel_main ⊗ Hankel_main.
        device: Computation device
        dtype: Computation dtype
    """
    assert torch.cuda.is_available(), "CUDA is required."

    T_prime, sqrt_T_prime, k_max = compute_dimensions(n)
    k = min(k, k_max)

    Z = get_hankel(sqrt_T_prime)
    sigma, phi = torch.linalg.eigh(Z)
    phi_i = phi[:, -k:] * sigma[-k:] ** 0.25

    if use_hankel_L: # TODO: We may want to use Hankel_L above too if use_hankel_L is true, make another variable for this (mix != use_hankel_L)
        logger.info("Mixing Hankel_L with Hankel_main to generate tensorized filters.")
        Z_L = get_hankel(sqrt_T_prime, True)
        sigma_L, phi_L = torch.linalg.eigh(Z_L)
        phi_j = phi_L[:, -k:] * sigma_L[-k:] ** 0.25
    else:
        phi_j = phi_i

    filters = torch.kron(phi_i, phi_j)
    return filters.to(device=device, dtype=dtype)
