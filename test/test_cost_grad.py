#!/usr/bin/env python3
"""Single-input cost-gradient validation; requires CUDA float64.

Run from an installed candidate: python -B test/test_cost_grad.py
"""

import gc
from unittest.mock import patch

import torch

import curegot

from _gradient_test_utils import (
    check_autocast_rejected,
    check_double_backward_rejected,
    check_reg_gradient_rejected,
    dense_reference,
    expect_error,
    problem,
    random_gradient,
)


def relative(actual, expected):
    denominator = torch.linalg.norm(expected)
    assert denominator > 0
    return float(torch.linalg.norm(actual - expected) / denominator)


def test_backward_against_dense_reference():
    # Create test data.
    M, a, b = problem()
    G = random_gradient(M)
    options = {"reg": 0.3, "tol": 1e-8}

    for mode in ("none", "diag", "splr"):
        # Run differentiable SPLR OT.
        cost = M.clone().requires_grad_(True)
        plan = curegot.torch.sinkhorn_autograd(
            cost,
            a,
            b,
            **options,
            backward_preconditioner=mode,
            backward_tol=1e-12,
        )

        # Construct a scalar loss and run backward.
        loss = (plan * G).sum()
        loss.backward()

        # Compute the independent dense reference.
        expected = dense_reference(plan.detach(), G, options["reg"])["gradients"][0]

        # Each preconditioner must give the correct cost gradient.
        error = relative(cost.grad, expected)
        assert error < 1e-8
        print(
            f"PASS {mode} backward against dense reference: relative error={error:.3e}"
        )
    print("PASS test_backward_against_dense_reference", flush=True)


def test_finite_difference():
    # Create test data.
    M, a, b = problem()
    G = random_gradient(M)
    options = {"reg": 0.3, "tol": 1e-8}

    # Compute the autograd gradient for each preconditioner.
    gradients = []
    for mode in ("none", "diag", "splr"):
        cost = M.clone().requires_grad_(True)
        plan = curegot.torch.sinkhorn_autograd(
            cost,
            a,
            b,
            **options,
            backward_preconditioner=mode,
            backward_tol=1e-12,
        )
        (plan * G).sum().backward()
        gradients.append((mode, float(cost.grad[2, 3])))

    # Perturb one cost entry.
    epsilon = 1e-3
    direction = torch.zeros_like(M)
    direction[2, 3] = 1
    with torch.no_grad():
        plus = curegot.torch.sinkhorn_autograd(
            M + epsilon * direction, a, b, **options
        )
        minus = curegot.torch.sinkhorn_autograd(
            M - epsilon * direction, a, b, **options
        )

    # Compute the central finite difference.
    finite_difference = float(((plus - minus) * G).sum() / (2 * epsilon))

    # Compare the numerical and autograd gradients.
    for mode, gradient in gradients:
        error = abs(finite_difference - gradient)
        assert error < 1e-5
        print(f"PASS {mode} finite difference: absolute error={error:.3e}")
    print("PASS test_finite_difference", flush=True)


def test_active_rank2():
    # Create a problem with an active rank-2 update.
    M, a, b = problem(100, 120, 40)
    G = random_gradient(M, seed=13)

    # Run differentiable SPLR OT.
    cost = M.clone().requires_grad_(True)
    plan = curegot.torch.sinkhorn_autograd(
        cost,
        a,
        b,
        reg=0.03,
        tol=1e-8,
        max_iter=200,
        backward_preconditioner="splr",
        backward_tol=1e-12,
    )

    # Require actual rank-2 activation and saved factorization reuse.
    context = plan.grad_fn.backward_context
    assert context.low_rank_enabled and context.adopted_forward_state

    # Run backward through the public API.
    plan.backward(G)

    # Compare with the independent dense reference.
    expected = dense_reference(plan.detach(), G, 0.03)["gradients"][0]
    assert relative(cost.grad, expected) < 1e-8
    print("PASS public backward with active rank-2 SPLR")
    print("PASS test_active_rank2", flush=True)


def test_forward_context():
    # Exercise public forward controls through the real native solver.
    M, a, b = problem()
    G = random_gradient(M)
    cost = M.clone().requires_grad_(True)
    with patch.object(
        curegot.torch,
        "sinkhorn_splr_save_ctx",
        wraps=curegot.torch.sinkhorn_splr_save_ctx,
    ) as forward:
        plan = curegot.torch.sinkhorn_autograd(
            cost,
            a,
            b,
            0.3,
            tol=1e-8,
            max_iter=200,
            verbose=1,
            backward_tol=1e-12,
        )
    assert forward.call_args.args[4:] == (1e-8, 200, 1)

    # Verify backward after forwarding the requested convergence controls.
    plan.backward(G)
    expected = dense_reference(plan.detach(), G, 0.3)["gradients"][0]
    assert relative(cost.grad, expected) < 1e-8

    # Create a controlled legacy forward result for context adoption/fallback.
    options = {"density": 1.0, "candidate_sinkhorn_iter": 0}
    legacy = curegot.torch.sinkhorn_splr(M, a, b, 0.3, 1e-8, 1000, 0, **options)

    for name, kwargs in (
        ("normal", {}),
        ("immediate", {"x0": legacy["dual"]}),
    ):
        # Save a usable context at the final forward point.
        saved = curegot.torch.sinkhorn_splr_save_ctx(
            M, a, b, 0.3, 1e-8, 1000, 0, **options, **kwargs
        )
        assert saved["converged"]
        assert relative(saved["plan"], legacy["plan"]) < 1e-10

        # Require adoption normally and fallback at immediate convergence.
        context = saved["context"]
        if name == "normal":
            assert context.adopted_forward_state
        else:
            assert saved["niter"] == 0 and context.is_fallback
            assert not context.low_rank_enabled

        # Use the saved context for backward.
        result = curegot.torch.solve_backward(
            saved["plan"], G, context, 0.3, preconditioner="splr", tol=1e-12
        )

        # Compare the cost gradient with the independent dense reference.
        expected = dense_reference(saved["plan"], G, 0.3)["gradients"][0]
        assert result["converged"] and relative(result["grad"], expected) < 1e-8
        print(f"PASS save_ctx {name}: usable context and dense cost gradient")
    print("PASS test_forward_context", flush=True)


def test_nondefault_stream():
    # Create test data and order the new stream after its producers.
    M, a, b = problem()
    G = random_gradient(M)
    stream = torch.cuda.Stream(device=M.device)
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream):
        # Run differentiable SPLR OT on the non-default stream.
        cost = M.clone().requires_grad_(True)
        plan = curegot.torch.sinkhorn_autograd(
            cost,
            a,
            b,
            reg=0.3,
            tol=1e-8,
            backward_preconditioner="splr",
            backward_tol=1e-12,
        )

        # Warm backward before delaying the upstream-gradient producer.
        delayed_gradient = torch.zeros_like(plan)
        torch.autograd.grad(plan, cost, torch.ones_like(plan), retain_graph=True)

        # Produce the upstream gradient on the same stream.
        torch.cuda._sleep(300000000)
        delayed_gradient.copy_(G)

        # Run backward on the same stream.
        plan.backward(delayed_gradient)

        # Consume the gradient immediately.
        observed = cost.grad.clone()

    # Compare against the reference after the stream finishes.
    stream.synchronize()
    expected = dense_reference(plan.detach(), G, 0.3)["gradients"][0]
    assert relative(observed, expected) < 1e-8
    print("PASS public backward / delayed gradient / same-stream consumer")
    print("PASS test_nondefault_stream", flush=True)


def test_lifetime_and_repeated_backward():
    # Create test data.
    M, a, b = problem()
    G = random_gradient(M)
    options = {"reg": 0.3, "tol": 1e-8}

    for mode in ("none", "diag", "splr"):
        # Build the forward graph.
        cost = M.clone().requires_grad_(True)
        plan = curegot.torch.sinkhorn_autograd(
            cost,
            a,
            b,
            **options,
            backward_preconditioner=mode,
            backward_tol=1e-12,
        )
        loss = (plan * G).sum()

        # Run the first backward and compare with the dense reference.
        loss.backward(retain_graph=True)
        expected = dense_reference(plan.detach(), G, options["reg"])["gradients"][0]
        assert relative(cost.grad, expected) < 1e-8
        first = cost.grad.clone()

        # Run repeated backward after garbage collection.
        cost.grad = None
        gc.collect()
        loss.backward()

        # Reusing the graph must produce the same gradient.
        assert relative(cost.grad, first) < 1e-8
        print(f"PASS {mode} repeated backward")

    # Build a graph whose forward-local plan can be released.
    cost = M.clone().requires_grad_(True)
    plan = curegot.torch.sinkhorn_autograd(
        cost, a, b, **options, backward_preconditioner="splr", backward_tol=1e-12
    )
    expected = dense_reference(plan.detach(), 2 * plan.detach(), options["reg"])["gradients"][0]
    loss = plan.square().sum()

    # Release forward-local references.
    del plan

    # Run backward after garbage collection.
    gc.collect()
    loss.backward()

    # Compare against the dense reference.
    assert relative(cost.grad, expected) < 1e-8
    print("PASS context lifetime after releasing forward locals")
    print("PASS test_lifetime_and_repeated_backward", flush=True)


def test_sizes_and_failures():
    # Unsupported public shapes must fail before solving.
    for n, m in ((4, 1), (0, 4)):
        M = torch.ones(n, m, device="cuda", dtype=torch.float64)
        a = torch.full((n,), 1 / max(n, 1), device=M.device, dtype=M.dtype)
        b = torch.full((m,), 1 / max(m, 1), device=M.device, dtype=M.dtype)
        expect_error(
            ValueError,
            "n >= 1",
            lambda: curegot.torch.sinkhorn_autograd(M, a, b, reg=0.3),
        )

    # Create the smallest supported problem.
    M = torch.tensor(
        [[0.1, 0.3]], device="cuda", dtype=torch.float64, requires_grad=True
    )
    a = M.new_ones(1)
    b = M.new_full((2,), 0.5)

    # Run differentiable SPLR OT.
    plan = curegot.torch.sinkhorn_autograd(M, a, b, reg=0.3, tol=1e-8)

    # The single-row plan is fixed by the column marginals.
    assert torch.allclose(plan, b[None, :], rtol=0, atol=1e-8)

    # Its cost gradient must be finite and effectively zero.
    plan.square().sum().backward()
    assert torch.isfinite(M.grad).all() and M.grad.abs().max() < 1e-8

    # Create test data.
    M, a, b = problem()

    # Reject forward iteration-limit failure.
    expect_error(
        RuntimeError,
        "forward solver did not converge",
        lambda: curegot.torch.sinkhorn_autograd(
            M, a, b, reg=0.3, tol=1e-8, max_iter=1
        ),
    )

    # Build a converged forward graph with a one-step backward limit.
    cost = M.clone().requires_grad_(True)
    plan = curegot.torch.sinkhorn_autograd(
        cost,
        a,
        b,
        reg=0.3,
        tol=1e-8,
        backward_preconditioner="none",
        backward_max_iter=1,
    )

    # Reject backward non-convergence without returning a cost gradient.
    expect_error(
        RuntimeError,
        "backward PCG did not converge",
        lambda: plan.backward(random_gradient(plan)),
    )
    assert cost.grad is None
    print("PASS smallest shape / invalid shapes / forward and backward failures")
    print("PASS test_sizes_and_failures", flush=True)


def test_tiny_gradient():
    # Create the compact normalization regression.
    M = torch.tensor(
        [[0.1, 0.4, 0.8], [0.9, 0.2, 0.5]],
        device="cuda",
        dtype=torch.float64,
        requires_grad=True,
    )
    a = M.new_tensor([0.4, 0.6])
    b = M.new_tensor([0.2, 0.3, 0.5])
    G = M.new_tensor([[1.0, -0.5, 0.3], [0.2, 0.8, -0.4]])

    # Run public backward with a tiny upstream gradient and default settings.
    plan = curegot.torch.sinkhorn_autograd(M, a, b, reg=0.3, tol=1e-8)
    scale = 1e-160
    plan.backward(G * scale)

    # Compute the ordinary-scale dense reference.
    expected = dense_reference(plan.detach(), G, 0.3)["gradients"][0]

    # Undo the scale before taking norms so the oracle cannot underflow.
    assert torch.count_nonzero(M.grad) > 0
    assert relative(M.grad / scale, expected) < 1e-8
    print("PASS public tiny-gradient normalization (1e-160)")
    print("PASS test_tiny_gradient", flush=True)


def test_shared_autograd_contracts():
    # Create representative M-only inputs.
    M, a, b = problem()
    M.requires_grad_(True)

    # Check the same public restrictions used by the marginal tests.
    check_reg_gradient_rejected(M, a, b, 0.3)
    check_autocast_rejected(M, a, b, 0.3)
    check_double_backward_rejected(
        M, a, b, 0.3, (M,),
        tol=1e-8,
        backward_tol=1e-12,
        feasibility_tol=1e-7,
    )
    print("PASS test_shared_autograd_contracts", flush=True)


def main():
    if not __debug__:
        raise RuntimeError("Tests require assertions; do not use -O or PYTHONOPTIMIZE.")
    assert torch.cuda.is_available(), "Requires CUDA float64"

    test_backward_against_dense_reference()
    test_finite_difference()
    test_active_rank2()
    test_forward_context()
    test_nondefault_stream()
    test_lifetime_and_repeated_backward()
    test_sizes_and_failures()
    test_tiny_gradient()
    test_shared_autograd_contracts()

    print("COST GRADIENT VALIDATION PASS", flush=True)


if __name__ == "__main__":
    main()
