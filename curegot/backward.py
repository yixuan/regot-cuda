"""Implicit gradient for a converged SPLR plan.

Solve Q(T) lambda = rhs_Q_scale(G, T), then assemble dL/dM, dL/da and dL/db.
The default native diagonal uses diag(A) of the context sparse matrix,
not diag(Q(T)). SPLR reuses its factorization and optional rank-2 update;
none selects plain CG. Performance depends on workload and tolerance.
"""

from ._internal_torch import torch as _native_torch
from ._backward_core import rhs_Q_scale, assemble_grad, assemble_marginal_grad

__all__ = ["solve_backward"]

_VALID_PRECONDITIONERS = ("diag", "splr", "none")


def solve_backward(
    T,
    G,
    ctx,
    reg,
    *,
    preconditioner="diag",
    tol=1e-6,
    max_iter=2000,
):
    """Solve the backward pass and assemble the implicit gradient.

    Args:
        T:   [n, m] float64 CUDA tensor, the optimal transport plan from
             sinkhorn_splr_save_ctx (out["plan"]).
        G:   [n, m] float64 CUDA tensor, upstream gradient dL/dT.
        ctx: SPLRBackwardContext returned by sinkhorn_splr_save_ctx
             (out["context"]).
        reg: entropic regularization used in the forward pass.
        preconditioner: "diag" (default) | "splr" | "none".
        tol:     PCG relative residual tolerance (default 1e-6).
        max_iter: PCG iteration cap.

    Returns:
        dict with
            "lambda":         [n + m - 1] reduced dual solution,
            "grad":           [n, m] implicit gradient dL/dM,
            "grad_a":         [n] marginal gradient,
            "grad_b":         [m] marginal gradient, last entry zero,
            "iterations":     PCG iterations used,
            "converged":      whether PCG met tol within max_iter,
            "preconditioner": the mode actually used.

    Note: the returned "lambda" is the reduced-gauge dual (beta_m = 0);
    "grad" already accounts for the gauge internally. The marginal gradients
    represent derivatives for equal-mass perturbations, sum(da)=sum(db);
    their gauge choice does not make the last b entry independent of the loss.
    """
    # Validate the plan/upstream shapes and preconditioner mode.
    if T.dim() != 2:
        raise ValueError("T must be a 2D tensor")
    if G.shape != T.shape:
        raise ValueError("G must have the same shape as T")
    if preconditioner not in _VALID_PRECONDITIONERS:
        raise ValueError(
            f"invalid preconditioner '{preconditioner}'; "
            f"expected one of: {' | '.join(_VALID_PRECONDITIONERS)}"
        )

    # Construct the reduced backward right-hand side.
    n, m = T.shape
    mu = rhs_Q_scale(G, T, n, m)

    # Solve the true reduced system in one native C++/CUDA call.
    result = _native_torch._solve_backward_native(
        T, mu, ctx, preconditioner, tol, max_iter
    )

    # Reuse the same restored-scale adjoint for all three input gradients.
    result["grad"] = assemble_grad(T, G, result["lambda"], reg, n, m)
    result["grad_a"], result["grad_b"] = assemble_marginal_grad(result["lambda"], n, m)
    result["preconditioner"] = preconditioner
    return result
