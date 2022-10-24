#!/usr/bin/env python

"""



Tests for calls to ``ConstrainedOptimizer.`` class.



"""

import cooper_test_utils
import pytest
import torch

import cooper


def test_inequality_multiplier():
    # For inequality constraints, the multipliers should be non-negative
    init_tensor = torch.tensor([0.1, -1.0, 2.5])
    multiplier = cooper.multipliers.DenseMultiplier(init_tensor, positive=True)
    multiplier.project_()
    assert torch.allclose(multiplier(), torch.tensor([0.1, 0.0, 2.5]))


def test_dual_optimizer():
    # Test that the first step along masked directions is like instantiating optim and doing step
    # correct (there is no
    # momentum so stuff happens differently)
    # https://discuss.pytorch.org/t/some-questions-about-the-adam-optimizer/129291/2

    pass


@pytest.mark.parametrize("aim_device", ["cpu", "cuda"])
def test_dual_optimzer(aim_device):
    """
    Verify constrained executions restart dual variables correctly on a toy 2D
    problem.
    """

    primal_optim_cls = torch.optim.SGD
    primal_optim_kwargs = {"lr": 1e-2, "momentum": 0.3}

    test_problem_data = cooper_test_utils.build_test_problem(
        aim_device=aim_device,
        primal_optim_cls=primal_optim_cls,
        primal_init=[0.0, -1.0],
        dual_optim_cls=torch.optim.Adam,
        use_ineq=True,
        use_proxy_ineq=False,
        dual_restarts=True,
        alternating=False,
        primal_optim_kwargs=primal_optim_kwargs,
        dual_optim_kwargs={"lr": 1e-2},
    )

    params, cmp, coop, formulation, device, mktensor = test_problem_data.as_tuple()

    for step_id in range(10):
        coop.zero_grad()

        # When using the unconstrained formulation, lagrangian = loss
        lagrangian = formulation.composite_objective(cmp.closure, params)
        formulation.custom_backward(lagrangian)

        coop.step()

    # Setup
    formulation.ineq_multipliers.grad[1] = 5
    coop.restart_dual()
