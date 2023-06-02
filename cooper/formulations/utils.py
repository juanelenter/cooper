from typing import Optional, Tuple

import torch

from cooper.constraints.constraint_state import ConstraintState, ConstraintType


def extract_and_patch_violations(constraint_state: ConstraintState) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts the violation and strict violation from the constraint state. If
    strict violations are not provided, patches them with the violation. This function
    also unsqueeze the violation tensors to ensure thay have at least 1-dimension."""

    violation = constraint_state.violation
    if len(violation.shape) == 0:
        violation = violation.unsqueeze(0)

    strict_violation = constraint_state.strict_violation
    if strict_violation is None:
        strict_violation = constraint_state.violation
    if len(strict_violation.shape) == 0:
        strict_violation = strict_violation.unsqueeze(0)

    return violation, strict_violation


def compute_primal_weighted_violation(
    constraint_factor: torch.Tensor, constraint_state: ConstraintState
) -> Optional[torch.Tensor]:
    """Computes the sum of constraint violations weighted by the associated constraint
    factors (multipliers or penalty coefficients), while preserving the gradient for the
    primal variables.

    Args:
        constraint_factor: The value of the multiplier or penalty coefficient for the
            constraint group.
        constraint_state: The current state of the constraint.
    """

    if constraint_state.skip_primal_contribution:
        # Ignore the primal contribution if the constraint is marked as non-contributing
        # to the primal Lagrangian.
        return None
    else:
        violation, _ = extract_and_patch_violations(constraint_state)

        if constraint_factor is None:
            raise ValueError("The constraint factor tensor must be provided if the primal contribution is not skipped.")
        if violation is None:
            raise ValueError("The violation tensor must be provided if the primal contribution is not skipped.")

        # When computing the gradient of the Lagrangian with respect to the primal
        # variables, we do not need to differentiate the multiplier. So we detach the
        # multiplier to avoid computing its gradient.
        # In the case of a penalty coefficient, the detach call is a no-op.
        return torch.einsum("i...,i...->", constraint_factor.detach(), violation)


def compute_dual_weighted_violation(
    constraint_factor: torch.Tensor, constraint_state: ConstraintState
) -> Optional[torch.Tensor]:
    """Computes the sum of constraint violations weighted by the associated constraint
    factors (multipliers or penalty coefficients), while preserving the gradient for the
    dual variables.

    When computing the gradient of the Lagrangian with respect to the dual variables, we
    only need the _value_ of the constraint violation and not its gradient. So we detach
    the violation to avoid computing its gradient. Note that this enables the use of
    non-differentiable constraints for updating the multipliers.

    This insight was originally presented by Cotter et al. in the paper "Optimization
    with Non-Differentiable Constraints with Applications to Fairness, Recall, Churn,
    and Other Goals" under the name of "proxy" constraints.
    (https://jmlr.org/papers/v20/18-616.html, Sec. 4.2)

    Args:
        multiplier_value: The value of the multiplier for the constraint group.
        constraint_state: The current state of the constraint.
    """

    if constraint_state.skip_dual_contribution:
        # Ignore the primal contribution if the constraint is marked as non-contributing
        # to the dual Lagrangian.
        return None
    elif not constraint_factor.requires_grad:
        # If the constraint factor corresponds to a penalty coefficient, we can skip
        # the computation of the dual contribution since the penalty coefficient is not
        # trainable.
        return None
    else:
        multiplier_value = constraint_factor
        if multiplier_value is None:
            raise ValueError("The constraint factor tensor must be provided if the dual contribution is not skipped.")

        # Strict violation represents the "actual" violation of the constraint. When
        # provided, we use the strict violation to update the value of the multiplier.
        # Otherwise, we default to using the differentiable violation.
        _, strict_violation = extract_and_patch_violations(constraint_state)

        return torch.einsum("i...,i...->", multiplier_value, strict_violation.detach())


def compute_quadratic_penalty(
    penalty_coefficient_value: torch.Tensor, constraint_state: ConstraintState, constraint_type: ConstraintType
) -> Optional[torch.Tensor]:
    # TODO(juan43ramirez): Add documentation

    if constraint_state.skip_primal_contribution:
        return None
    else:
        violation, strict_violation = extract_and_patch_violations(constraint_state)

        if violation is None:
            raise ValueError("The violation tensor must be provided if the primal contribution is not skipped.")

        if constraint_type == ConstraintType.INEQUALITY:
            # Compute filter based on strict constraint violation
            const_filter = strict_violation >= 0

            # TODO(juan43ramirez): We used to also penalize inequality constraints
            # with a non-zero multiplier. This seems confusing to me.
            # const_filter = torch.logical_or(strict_violation >= 0, multiplier_value.detach() > 0)

            sq_violation = const_filter * (violation**2)

        elif constraint_type == ConstraintType.EQUALITY:
            # Equality constraints do not need to be filtered
            sq_violation = violation**2
        else:
            raise ValueError(f"{constraint_type} is incompatible with quadratic penalties.")

        if penalty_coefficient_value.numel() == 1:
            # One penalty coefficient shared across all constraints.
            return torch.sum(sq_violation) * penalty_coefficient_value / 2
        else:
            # One penalty coefficient per constraint.
            if violation.shape != penalty_coefficient_value.shape:
                raise ValueError("The violation tensor must have the same shape as the penalty coefficient tensor.")

            return torch.einsum("i...,i...->", penalty_coefficient_value, sq_violation) / 2