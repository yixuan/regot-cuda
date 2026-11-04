"""Shared test-only data, independent references and autograd contracts."""

import torch

import curegot


def problem(n=16, m=19, seed=42):
    generator = torch.Generator().manual_seed(seed)
    M = torch.rand(n, m, generator=generator, dtype=torch.float64).cuda()
    a = torch.rand(n, generator=generator, dtype=torch.float64).cuda() + 0.5
    b = torch.rand(m, generator=generator, dtype=torch.float64).cuda() + 0.5
    return M, a / a.sum(), b / b.sum()


def random_gradient(plan, seed=9):
    return torch.randn(
        plan.shape, generator=torch.Generator().manual_seed(seed), dtype=plan.dtype
    ).to(plan.device)


def expect_error(error_type, message, call):
    try:
        call()
    except error_type as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected {error_type.__name__}: {message}")


def dense_reference(T, G, reg):
    """Solve explicit reduced Q once; gradients are ordered M, a, b."""
    n = T.shape[0]
    R = T[:, :-1]
    Q = torch.cat(
        [
            torch.cat([torch.diag(T.sum(1)), R], 1),
            torch.cat([R.T, torch.diag(T.sum(0)[:-1])], 1),
        ],
        0,
    )
    weighted = G * T
    rhs = torch.cat([weighted.sum(1), weighted.sum(0)[:-1]])
    adjoint = torch.linalg.solve(Q, rhs)

    # Assemble the reference without importing production gradient helpers.
    grad_a = adjoint[:n]
    grad_b = torch.cat([adjoint[n:], adjoint.new_zeros(1)])
    grad_M = T * (grad_a[:, None] + grad_b[None, :] - G) / reg
    return {
        "Q": Q,
        "rhs": rhs,
        "lambda": adjoint,
        "gradients": (grad_M, grad_a, grad_b),
    }


def check_marginals(T, a, b, atol):
    assert bool(torch.isfinite(a).all() & (a > 0).all())
    assert bool(torch.isfinite(b).all() & (b > 0).all())
    assert abs(float(a.sum() - b.sum())) <= 2e-14
    assert bool(torch.isfinite(T).all())

    error = max(
        float((T.sum(1) - a).abs().max()),
        float((T.sum(0) - b).abs().max()),
    )
    assert error <= atol, f"full marginal residual {error:.3e}"
    return error


def _input_names(M, a, b):
    return "/".join(
        name for name, value in (("M", M), ("a", a), ("b", b))
        if value.requires_grad
    )


def check_reg_gradient_rejected(M, a, b, reg):
    expect_error(
        NotImplementedError,
        "reg",
        lambda: curegot.torch.sinkhorn_autograd(
            M, a, b, M.new_tensor(reg, requires_grad=True)
        ),
    )
    print(f"PASS reg gradient rejected: inputs={_input_names(M, a, b)}", flush=True)


def check_autocast_rejected(M, a, b, reg):
    with torch.autocast("cuda"):
        expect_error(
            RuntimeError,
            "AMP/autocast",
            lambda: curegot.torch.sinkhorn_autograd(M, a, b, reg),
        )
    print(f"PASS autocast rejected: inputs={_input_names(M, a, b)}", flush=True)


def check_double_backward_rejected(
    M, a, b, reg, inputs, *, tol, backward_tol, feasibility_tol
):
    label = _input_names(M, a, b)
    print(f"CASE double backward: inputs={label}", flush=True)
    assert inputs and all(value.requires_grad for value in inputs)

    # Complete first differentiation outside the expected-error check.
    T = curegot.torch.sinkhorn_autograd(
        M, a, b, reg,
        tol=tol,
        max_iter=1000,
        backward_preconditioner="diag",
        backward_tol=backward_tol,
    )
    residual = check_marginals(T.detach(), a.detach(), b.detach(), feasibility_tol)
    G = random_gradient(M).requires_grad_(True)
    first = torch.autograd.grad(T, inputs, G, create_graph=True)
    assert len(first) == len(inputs)
    for gradient, value in zip(first, inputs):
        assert gradient.shape == value.shape
        assert gradient.dtype == value.dtype
        assert gradient.device == value.device
        assert bool(torch.isfinite(gradient).all())
        assert gradient.requires_grad and gradient.grad_fn is not None

    # A missing first-order graph must not count as second-order rejection.
    second_loss = sum(gradient.sum() for gradient in first)
    expect_error(
        RuntimeError,
        "once_differentiable",
        lambda: second_loss.backward(),
    )
    print(f"PASS double backward rejected: inputs={label}", flush=True)
    return residual
