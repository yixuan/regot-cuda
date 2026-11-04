"""Matrix-free reduced-gauge implicit differentiation (last beta fixed to zero)."""

import torch

__all__ = ["rhs_Q_scale", "assemble_grad", "assemble_marginal_grad"]


def rhs_Q_scale(G, T, n, m):
    """mu = [(G . T) 1_m, ((G . T)^T 1_n)_{-m}]  (stored scale, no 1/reg)."""
    GT = G * T
    return torch.cat([GT.sum(dim=1), GT.sum(dim=0)[:m - 1]])


def assemble_grad(T, G, lam, reg, n, m):
    """dL/dM = (1/reg) * T . (lam_alpha 1^T + 1 lam_beta^T - G)."""
    # Restore the full dual vector with beta_m fixed to zero.
    lam_alpha = lam[:n]
    lam_beta = torch.zeros(m, dtype=lam.dtype, device=lam.device)
    lam_beta[:m - 1] = lam[n:]

    # Assemble the cost gradient in the original regularization scale.
    return (T * (lam_alpha[:, None] + lam_beta[None, :] - G)) / reg


def assemble_marginal_grad(lam, n, m):
    """Marginal VJPs in the reduced gauge: ga=lam_alpha, gb=[lam_beta, 0].

    rhs_Q_scale omits 1/reg, so these gradients need no reg scaling.
    They represent derivatives on sum(da)=sum(db), not independent changes
    of total source and target mass. grad_a is a view of lam.
    """
    grad_a = lam[:n]
    grad_b = lam.new_zeros(m)
    grad_b[:-1] = lam[n:]
    return grad_a, grad_b
