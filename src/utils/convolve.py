import torch
import torch.nn.functional as F

from tensorized_filters.utils.numerics import nearest_power_of_two
from flashfftconv import FlashFFTConv


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

def flash_convolve(
    u: torch.Tensor, v: torch.Tensor, flash_fft: FlashFFTConv, use_approx: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flash FFT convolution.

    Args:
        u (torch.Tensor): Input tensor of shape `(B, L, d_in)`, where:
            - `B` is the batch size,
            - `L` is the sequence length,
            - `d_in` is the input dimension.
        v (torch.Tensor): Filter tensor of shape `(K, d_in)`, where:
            - `K` is the number of filters,
            - `d_in` is the input dimension.
        flash_fft (FlashFFTConv): An instance of the FlashFFTConv module, used to perform the convolution.
        use_approx (bool, optional): If `True`, performs the tensordot approximation (default is `True`).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple `(U_plus, U_minus)`:
            - `U_plus`: Convolved output tensor with positive eigenvalues.
            - Shape depends on `use_approx`:
                - If `use_approx=True`: `(B, L, d_in)`
                - If `use_approx=False`: `(B, L, K, d_in)`
            - `U_minus`: Convolved output tensor with negative eigenvalues.
            - Shape depends on `use_approx`:
                - If `use_approx=True`: `(B, L, d_in)`
                - If `use_approx=False`: `(B, L, K, d_in)`

    Raises:
        ValueError: If the input tensor shapes do not conform to the expected dimensions.

    Example:
        >>> u = torch.randn(4, 16, 32)  # (B, L, d_in)
        >>> v = torch.randn(8, 32)      # (K, d_in)
        >>> flash_fft = FlashFFTConv(n=16, dtype=torch.float32)
        >>> U_plus, U_minus = flash_convolve(u, v, flash_fft, use_approx=True)
        >>> print(U_plus.shape, U_minus.shape)
        torch.Size([4, 16, 32]) torch.Size([4, 16, 32])
        """
    bsz, seq_len, d_in = u.shape
    _, K = v.shape

    padded_len = nearest_power_of_two(seq_len, round_up=True)
    pad_len = padded_len - seq_len

    sgn = torch.full((1, 1, padded_len), 1, device=u.device)
    sgn[:, :, 1::2] = -1

    if use_approx:
        u_padded = F.pad(u.transpose(1, 2), (0, pad_len)).to(torch.bfloat16).contiguous()
        v_padded = F.pad(v.transpose(0, 1), (0, pad_len)).to(torch.float32).contiguous()
        u_conv = torch.stack([u_padded, u_padded * sgn], dim=0).reshape(2 * bsz, d_in, padded_len)
    else:
        u_k_padded = F.pad(u.transpose(1, 2), (0, pad_len)).to(torch.bfloat16).repeat_interleave(K, dim=1).contiguous()
        v_padded = F.pad(v.transpose(0, 1), (0, pad_len)).to(torch.float32).repeat(d_in, 1).contiguous()
        u_conv = torch.stack([u_k_padded, u_k_padded * sgn], dim=0).reshape(2 * bsz, K * d_in, padded_len)

    U_conv = flash_fft(u_conv, v_padded)

    # Trim the output back to the original sequence length
    U_conv = U_conv[..., :seq_len]

    u_plus, u_minus = torch.chunk(U_conv, 2, dim=0)

    if use_approx:
        u_minus = u_minus * sgn[:, :, :seq_len]
        U_plus, U_minus = u_plus.transpose(1, 2), u_minus.transpose(1, 2)
    else:
        sgn = sgn[:, :, :seq_len].unsqueeze(-1).transpose(1, 2)
        U_plus = u_plus.view(bsz, d_in, K, seq_len).permute(0, 3, 2, 1).contiguous()
        U_minus = u_minus.view(bsz, d_in, K, seq_len).permute(0, 3, 2, 1).contiguous() * sgn

    return U_plus, U_minus
