"""Single-input marginal VJPs: dense oracle, feasible FD and autograd contracts.

Run after building/installing the candidate:
    python -B test/test_marginal_grad.py
"""

import gc
from unittest.mock import patch

import torch

import curegot

from _gradient_test_utils import (
    check_autocast_rejected,
    check_double_backward_rejected,
    check_marginals,
    check_reg_gradient_rejected,
    dense_reference,
    expect_error,
    problem,
    random_gradient,
)


MODES = ("none", "diag", "splr")
_metrics = {}


def record(name, value):
    _metrics[name] = max(_metrics.get(name, 0.0), float(value))


def print_metrics():
    for name, value in sorted(_metrics.items()):
        print(f"MARGINAL_METRIC {name}={value:.12e}", flush=True)


def small_problem():
    M = torch.tensor(
        [[0.0, 0.2, 0.4, 0.1],
         [0.3, 0.0, 0.2, 0.4],
         [0.1, 0.4, 0.0, 0.2]],
        device="cuda",
        dtype=torch.float64,
    )
    # Weak nonseparable costs reach FD accuracy with the default candidate.
    # Harder active rank-2 cases are checked separately.
    M = 0.01 * M
    a = M.new_tensor([0.25, 0.25, 0.5])
    b = M.new_tensor([0.125, 0.25, 0.125, 0.5])
    return M, a, b


def plan(M, a, b, reg, mode="diag", tol=1e-11, feasibility_tol=2e-10):
    T = curegot.torch.sinkhorn_autograd(
        M, a, b, reg,
        tol=tol,
        max_iter=1000,
        backward_preconditioner=mode,
        backward_tol=1e-12,
    )
    record(
        "full_marginal_residual",
        check_marginals(T.detach(), a.detach(), b.detach(), feasibility_tol),
    )
    return T


def close_vjps(actual, expected, scale=1.0, atol=2e-9, rtol=2e-8):
    for value, reference in zip(actual, expected):
        # Undo tiny-RHS scaling before taking norms.
        value = value / scale
        torch.testing.assert_close(value, reference, atol=atol, rtol=rtol)

        error = torch.linalg.vector_norm(value - reference)
        denominator = torch.linalg.vector_norm(reference).clamp_min(1e-30)
        record("dense_vjp_relative_error", error / denominator)


def finite_difference(loss_at, expected, name):
    # Binary steps preserve the masses of dyadic marginal directions.
    eps = 2.0 ** -12
    with torch.no_grad():
        observed = float((loss_at(eps) - loss_at(-eps)) / (2 * eps))

    expected = float(expected)
    error = abs(observed - expected)
    assert error <= 2e-7 + 2e-5 * abs(expected), (
        f"{name}: fd={observed:.12e}, vjp={expected:.12e}"
    )
    record(name + "_fd_absolute_error", error)
    record(
        name + "_fd_relative_error",
        error / max(abs(observed), abs(expected), 1e-12),
    )


def test_wiring_dense_oracle_and_one_native_solve():
    masks = (
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (False, True, True),
        (True, True, True),
    )
    for reg in (0.2, 0.5):
        for mode in MODES:
            for needs in masks:
                print(
                    f"CASE test_wiring_dense_oracle_and_one_native_solve: "
                    f"reg={reg}, mode={mode}, needs={needs}",
                    flush=True,
                )

                # Create test data for this gradient selection.
                inputs = tuple(
                    x.clone().requires_grad_(need)
                    for x, need in zip(small_problem(), needs)
                )
                M, a, b = inputs
                G = random_gradient(M)

                # Run forward and compute the independent reference.
                T = plan(M, a, b, reg, mode)
                oracle = dense_reference(T.detach(), G, reg)
                Q, rhs, reference = oracle["Q"], oracle["rhs"], oracle["gradients"]
                if all(needs):
                    result = curegot.torch.solve_backward(
                        T.detach(), G, T.grad_fn.backward_context, reg,
                        preconditioner=mode,
                        tol=1e-12,
                    )
                    assert result["converged"]

                    # Check the true residual and all manual VJPs.
                    residual = torch.linalg.vector_norm(Q @ result["lambda"] - rhs)
                    relative = residual / torch.linalg.vector_norm(rhs)
                    assert float(relative) < 2e-10
                    record("true_relative_residual", relative)
                    close_vjps(
                        (result["grad"], result["grad_a"], result["grad_b"]),
                        reference,
                    )

                # Require exactly one native solve during public backward.
                with patch.object(
                    curegot.torch,
                    "_solve_backward_native",
                    wraps=curegot.torch._solve_backward_native,
                ) as solve:
                    T.backward(G)
                assert solve.call_count == 1

                # Check gradient delivery, metadata and numerical values.
                for value, need, expected in zip(inputs, needs, reference):
                    if need:
                        assert value.grad is not None
                        assert value.grad.shape == value.shape
                        assert value.grad.dtype == value.dtype
                        assert value.grad.device == value.device
                        close_vjps((value.grad,), (expected,))
                    else:
                        assert value.grad is None
                if needs[2]:
                    assert float(b.grad[-1]) == 0.0

    # A call without differentiable inputs must still work.
    T = plan(*small_problem(), reg=0.5)
    assert not T.requires_grad
    print("PASS test_wiring_dense_oracle_and_one_native_solve", flush=True)


def test_feasible_marginal_finite_differences():
    for reg in (0.2, 0.5):
        for mode in MODES:
            print(
                f"CASE test_feasible_marginal_finite_differences: "
                f"reg={reg}, mode={mode}",
                flush=True,
            )

            # Create test data and compute marginal gradients.
            M, a, b = small_problem()
            a.requires_grad_(True)
            b.requires_grad_(True)
            G = random_gradient(M, seed=123)
            T = plan(M, a, b, reg, mode)
            (T * G).sum().backward()
            ga, gb = a.grad, b.grad
            a0, b0 = a.detach(), b.detach()

            # Span each fixed-mass simplex with e_i - e_last.
            for name, marginal, gradient in (("a", a0, ga), ("b", b0, gb)):
                for i in range(marginal.numel() - 1):
                    direction = torch.zeros_like(marginal)
                    direction[i], direction[-1] = 1.0, -1.0

                    def loss_at(step):
                        aa = a0 + step * direction if name == "a" else a0
                        bb = b0 + step * direction if name == "b" else b0
                        return (G * plan(M, aa, bb, reg, mode)).sum()

                    finite_difference(loss_at, gradient @ direction, name)

            # Change both masses together, including the gauge-fixed b[-1].
            da, db = torch.zeros_like(a0), torch.zeros_like(b0)
            da[0], db[-1] = 1.0, 1.0
            finite_difference(
                lambda h: (
                    G * plan(M, a0 + h * da, b0 + h * db, reg, mode)
                ).sum(),
                ga @ da + gb @ db,
                "joint_mass",
            )
            finite_difference(
                lambda h: (
                    G * plan(M, (1 + h) * a0, (1 + h) * b0, reg, mode)
                ).sum(),
                ga @ a0 + gb @ b0,
                "mass_scaling",
            )

    print("PASS test_feasible_marginal_finite_differences", flush=True)


def test_zero_tiny_rhs_and_repeated_backward():
    for mode in MODES:
        print(f"CASE test_zero_tiny_rhs_and_repeated_backward: mode={mode}", flush=True)

        # Create test data and the ordinary-scale reference.
        M, a, b = (x.clone().requires_grad_(True) for x in small_problem())
        G = random_gradient(M)
        T = plan(M, a, b, 0.5, mode)
        reference = dense_reference(T.detach(), G, 0.5)["gradients"]

        # Require exactly zero gradients for an exactly zero RHS.
        T.backward(torch.zeros_like(G), retain_graph=True)
        for value in (M, a, b):
            assert int(torch.count_nonzero(value.grad)) == 0

        # Reuse the graph after GC with tiny and ordinary upstream scales.
        for scale in (1e-160, 1.0):
            for value in (M, a, b):
                value.grad = None
            gc.collect()
            T.backward(G * scale, retain_graph=True)
            close_vjps((M.grad, a.grad, b.grad), reference, scale)

    print("PASS test_zero_tiny_rhs_and_repeated_backward", flush=True)


def test_nondefault_stream_and_strided_upstream():
    for mode in MODES:
        print(f"CASE test_nondefault_stream_and_strided_upstream: mode={mode}", flush=True)

        # Create data and order the new stream after its producers.
        base = small_problem()
        G = random_gradient(base[0])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            M, a, b = (x.clone().requires_grad_(True) for x in base)
            T = plan(M, a, b, 0.5, mode)
            storage = torch.zeros_like(T.T.contiguous())
            upstream = storage.T
            assert not upstream.is_contiguous()

            # Delay the strided upstream producer on this stream.
            torch.cuda._sleep(2000000)
            storage.copy_(G.T)
            T.backward(upstream)
            observed = tuple(x.grad.clone() for x in (M, a, b))

        # Compare after the stream finishes.
        stream.synchronize()
        reference = dense_reference(T.detach(), G, 0.5)["gradients"]
        close_vjps(observed, reference)

    print("PASS test_nondefault_stream_and_strided_upstream", flush=True)


def test_active_rank2_marginals():
    # Create the existing harder problem with active rank-2 state.
    M, a, b = (x.requires_grad_(True) for x in problem(100, 120, 40))
    G = random_gradient(M, seed=13)
    T = plan(M, a, b, 0.03, "splr", tol=1e-8, feasibility_tol=2e-7)
    context = T.grad_fn.backward_context
    assert context.low_rank_enabled and context.adopted_forward_state

    # Run backward and compare all VJPs with the independent reference.
    reference = dense_reference(T.detach(), G, 0.03)["gradients"]
    T.backward(G)
    close_vjps((M.grad, a.grad, b.grad), reference, atol=2e-8, rtol=1e-7)
    print("PASS test_active_rank2_marginals", flush=True)


def test_immediate_fallback_marginals():
    # Solve once to get a converged starting point for the real fallback.
    M, a, b = (x.requires_grad_(True) for x in problem())
    G = random_gradient(M)
    native_forward = curegot.torch.sinkhorn_splr_save_ctx
    controls = {"density": 1.0, "candidate_sinkhorn_iter": 0}
    with torch.no_grad():
        initial = curegot.torch.sinkhorn_splr(
            M, a, b, 0.3, 1e-8, 1000, 0, **controls
        )

    # Exercise immediate convergence without adding public forward knobs.
    def warm_forward(*args):
        return native_forward(*args, **controls, x0=initial["dual"])

    with patch.object(
        curegot.torch, "sinkhorn_splr_save_ctx", side_effect=warm_forward
    ):
        T = plan(M, a, b, 0.3, "splr", tol=1e-8, feasibility_tol=2e-7)
    assert T.grad_fn.backward_context.is_fallback
    assert not T.grad_fn.backward_context.low_rank_enabled

    # Check all gradients against the independent reference.
    reference = dense_reference(T.detach(), G, 0.3)["gradients"]
    T.backward(G)
    close_vjps((M.grad, a.grad, b.grad), reference)
    print("PASS test_immediate_fallback_marginals", flush=True)


def test_existing_contracts_with_marginal_gradients():
    # Check shared restrictions with the original marginal data and reg.
    M, a, b = small_problem()
    a.requires_grad_(True)
    b.requires_grad_(True)
    check_reg_gradient_rejected(M, a, b, 0.5)
    check_autocast_rejected(M, a, b, 0.5)

    # Exercise marginal-only paths through the same second-order check.
    for needs in ((False, True, False), (False, False, True), (False, True, True)):
        cost, source, target = (
            value.detach().clone().requires_grad_(need)
            for value, need in zip((M, a, b), needs)
        )
        inputs = tuple(value for value in (cost, source, target) if value.requires_grad)
        residual = check_double_backward_rejected(
            cost, source, target, 0.5, inputs,
            tol=1e-11,
            backward_tol=1e-12,
            feasibility_tol=2e-10,
        )
        record("full_marginal_residual", residual)

    expect_error(
        ValueError,
        "m >= 2",
        lambda: curegot.torch.sinkhorn_autograd(M[:, :1], a, b[:1], 0.5),
    )

    # Check marginal gradients for the smallest supported shape.
    M = M.new_tensor([[0.1, 0.3]])
    a = M.new_tensor([1.0], requires_grad=True)
    b = M.new_tensor([0.5, 0.5], requires_grad=True)
    G = M.new_tensor([[0.7, -0.2]])
    T = plan(M, a, b, 0.5)
    reference = dense_reference(T.detach(), G, 0.5)["gradients"]
    T.backward(G)
    close_vjps((a.grad, b.grad), reference[1:])
    print("PASS test_existing_contracts_with_marginal_gradients", flush=True)


def main():
    if not __debug__:
        raise RuntimeError("Tests require assertions; do not use -O or PYTHONOPTIMIZE.")
    assert torch.cuda.is_available(), "Requires CUDA float64"

    test_wiring_dense_oracle_and_one_native_solve()
    test_feasible_marginal_finite_differences()
    test_zero_tiny_rhs_and_repeated_backward()
    test_nondefault_stream_and_strided_upstream()
    test_active_rank2_marginals()
    test_immediate_fallback_marginals()
    test_existing_contracts_with_marginal_gradients()

    print_metrics()
    print("MARGINAL GRADIENT VALIDATION PASS", flush=True)


if __name__ == "__main__":
    main()
