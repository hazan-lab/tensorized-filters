import math

import numpy as np
import torch

torch.manual_seed(1337)
device = torch.device("cuda")

@torch.jit.script
def get_hankel(n: int) -> torch.Tensor:
    """
    Generates a Hankel matrix Z, as defined in Equation (3) of the paper.

    This special matrix is used for the spectral filtering in the Spectral
    Transform Unit (STU).

    Args:
        n (int): Size of the square Hankel matrix.

    Returns:
        torch.Tensor: Hankel matrix Z of shape [n, n].
    """
    i = torch.arange(1, n + 1)
    s = i[:, None] + i[None, :]
    Z = 2.0 / (s**3 - s)
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
    sigma, phi = sigma[-K:], phi[:, -K:]
    phi *= sigma ** 0.25
    return torch.tensor(phi, device=device, dtype=dtype)

def convolve(u: torch.Tensor, v: torch.Tensor, n: int, use_approx: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len, d_in = u.shape

    sgn = torch.full((1, seq_len, 1), 1, device=u.device)
    sgn[:, 1::2] *= -1
    if use_approx:
        _, d_out = v.shape
        v = v.view(1, -1, d_out, 1).to(torch.float32)
    else:
        _, K = v.shape
        sgn = sgn.unsqueeze(-1)
        v = v.view(1, -1, K, 1, 1).to(torch.float32) # (bsz, seq_len, K, d_in, stack)
        u = u.view(bsz, -1, 1, d_in).expand(bsz, -1, K, d_in)

    v = torch.fft.rfft(v, n=n, dim=1)
    U = torch.stack([u, u * sgn], dim=-1).to(torch.float32)
    U = torch.fft.rfft(U, n=n, dim=1)
    U_conv = torch.fft.irfft(v * U, n=n, dim=1)[:, :seq_len]
    U_plus, U_minus = torch.unbind(U_conv, dim=-1)
    U_minus = U_minus * sgn

    return U_plus, U_minus

def brute_force_kron(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes the Kronecker product of two matrices A and B manually.
    
    Args:
        A (torch.Tensor): The first matrix of shape (m, n).
        B (torch.Tensor): The second matrix of shape (p, q).
        
    Returns:
        torch.Tensor: The Kronecker product of A and B of shape (m * p, n * q).
    """
    m, n = A.shape
    p, q = B.shape

    result = torch.zeros((m * p, n * q), device=A.device)
    
    # Compute the Kronecker product by iterating over each element
    for i in range(m):
        for j in range(n):
            result[i * p: (i + 1) * p, j * q: (j + 1) * q] = A[i, j] * B
    
    return result


def get_tensorized_filters_vectorized(n: int, k: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates tensorized spectral filters as defined in the paper.
    Uses tensor (Kronecker) products of eigenvectors of the Hankel matrix.
    """
    # n = 32 -> sqrt_t_prime_minus_2 = 6
    # hankel size -> (6, 6)
    # k = 24
    # (6, 24) kron (6, 24) = (36, 576)
    
    T_prime =  (math.ceil(math.sqrt(n - 2)))**2 + 2
    sqrt_T_prime = math.ceil(math.sqrt(T_prime - 2))
    Z = get_hankel(sqrt_T_prime).to(device)
    
    sigma, phi = torch.linalg.eigh(Z)

    sigma_k = sigma[-k:] # [k]
    phi_k = phi[:, -k:] # [sqrt(T'-2), k]

    # Create all pairs of eigenvalues and eigenvectors
    phi_i = phi_k # [sqrt(T'-2), k]
    phi_j = phi_k # [sqrt(T'-2), k]
    sigma_i = sigma_k
    sigma_j = sigma_k
    
    phi_i = phi_i * sigma_i ** 0.25
    phi_j = phi_j * sigma_j ** 0.25

    # want to unsqueeze out some dims to parallelize looping

    # take kronecker product
    kron_prod = torch.kron(phi_i, phi_j) # (n * n, k * k)
    
    # reshape
    kron_prod = kron_prod.reshape(sqrt_T_prime * sqrt_T_prime, k, k)

    # sum over k's
    result = kron_prod.sum(dim=(1, 2)) # (n * n)

    return result

def convolve_tensorized(
    u: torch.Tensor,    # [bsz, seq_len, d_in]
    phi: torch.Tensor,  # [sqrt(T'-2), k^2]
    n: int
) -> torch.Tensor:
    """
    Compute convolution with tensorized filters.
    """
    bsz, seq_len, d_in = u.shape

    # FFT of input and filters
    u_f = torch.fft.rfft(u.to(torch.float32), n=n, dim=1)   # [bsz, seq_len//2 + 1, d_in]
    phi_f = torch.fft.rfft(phi.to(torch.float32), n=n, dim=0)   # [seq_len//2 + 1, k^2]

    # Convolution in frequency domain
    U = torch.fft.irfft(
        u_f.unsqueeze(2) * phi_f.unsqueeze(1),
        n=n, dim=1
    )[:, :seq_len]  # [bsz, seq_len, k^2, d_in]

    return u

k = 16
n = 512

T_prime = (math.ceil(math.sqrt(n - 2)))**2 + 2
sqrt_T_prime = math.ceil(math.sqrt(T_prime))

Z = get_hankel(sqrt_T_prime - 2).to(device)
print(Z.shape)

# Get the eigenvalues and eigenvectors
sigma, phi = torch.linalg.eigh(Z)

# Select the top k eigenvalues and eigenvectors
sigma_k = sigma[-k:]
phi_k = phi[:, -k:]

# Scale eigenvectors by sigma_k to the 0.25 power
phi = phi_k * (sigma_k ** 0.25)

# Create two independent copies of phi
phi_i = phi.detach().clone()
phi_j = phi.detach().clone()


o = brute_force_kron(phi_i, phi_j)
o = get_tensorized_filters_vectorized(n, k, device)


print(o.shape)