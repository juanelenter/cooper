import warnings
from collections.abc import Sequence
from typing import List, Optional, TypeVar, Union

import torch

from cooper.constraints import ConstraintGroup
from cooper.multipliers import MULTIPLIER_TYPE
from cooper.optim import CooperOptimizerState, UnconstrainedOptimizer, constrained_optimizers
from cooper.utils import OneOrSequence, ensure_sequence


def sanity_check_constraints_and_optimizer(
    constrained_optimizer: constrained_optimizers.ConstrainedOptimizer,
    constraint_groups: Optional[OneOrSequence[ConstraintGroup]] = None,
):
    """
    Execute sanity checks on the properties of the provided constraints and optimizer to
    raise appropriate exceptions or warnings when detecting invalid or untested
    configurations.
    """

    if constraint_groups is not None:

        constraint_groups = ensure_sequence(constraint_groups)

        fn_restart_on_feasible = lambda constraint: getattr(constraint.multiplier, "restart_on_feasible", False)
        any_restart_on_feasible = any(map(fn_restart_on_feasible, constraint_groups))

        fn_augmented_lagrangian = lambda constraint: constraint.formulation.formulation_type == "augmented_lagrangian"
        any_is_augmented_lagrangian = any(map(fn_augmented_lagrangian, constraint_groups))

        if constrained_optimizer.alternating and any_restart_on_feasible:
            warnings.warn("Using alternating updates with dual restarts is untested. Please use with caution.")

        if any_is_augmented_lagrangian and not constrained_optimizer.alternating:
            raise RuntimeError("Augmented Lagrangian formulation requires alternating updates.")


def create_optimizer_from_kwargs(
    primal_optimizers: OneOrSequence[torch.optim.Optimizer],
    extrapolation: bool,
    alternating: bool,
    dual_optimizers: Optional[OneOrSequence[torch.optim.Optimizer]] = None,
    constraint_groups: Optional[OneOrSequence[ConstraintGroup]] = None,
    multipliers: Optional[OneOrSequence[MULTIPLIER_TYPE]] = None,
) -> Union[UnconstrainedOptimizer, constrained_optimizers.ConstrainedOptimizer]:
    """Creates a constrained or unconstrained optimizer from a set of keyword arguments.
    This method disambiguates the appropriate optimizer class to instantiate.

    Args:
        primal_optimizers: Optimizer(s) for the primal variables.
        dual_optimizer: Optional optimizer(s) for the dual variables.
        extrapolation: Whether the optimizer uses extrapolation.
        alternating: Whether the optimizer performs alternating updates.
    """

    if dual_optimizers is None:
        return UnconstrainedOptimizer(primal_optimizers=primal_optimizers)

    optimizer_kwargs = dict(
        primal_optimizers=primal_optimizers,
        dual_optimizers=dual_optimizers,
        multipliers=multipliers,
        constraint_groups=constraint_groups,
    )

    if extrapolation:
        return constrained_optimizers.ExtrapolationConstrainedOptimizer(**optimizer_kwargs)
    elif alternating:
        return constrained_optimizers.AlternatingConstrainedOptimizer(**optimizer_kwargs)
    else:
        return constrained_optimizers.SimultaneousConstrainedOptimizer(**optimizer_kwargs)


def load_cooper_optimizer_from_state_dict(
    cooper_optimizer_state: CooperOptimizerState,
    primal_optimizers: OneOrSequence[torch.optim.Optimizer],
    dual_optimizers: Optional[OneOrSequence[torch.optim.Optimizer]] = None,
    constraint_groups: Optional[OneOrSequence[ConstraintGroup]] = None,
    multipliers: Optional[OneOrSequence[MULTIPLIER_TYPE]] = None,
):
    """Creates a Cooper optimizer and loads the state_dicts contained in a
    :py:class:`~cooper.optim.CooperOptimizerState` onto instantiated primal and dual
    optimizers and constraint groups or multipliers.
    """

    primal_optimizers = ensure_sequence(primal_optimizers)
    primal_optimizer_states = cooper_optimizer_state.primal_optimizer_states

    if len(primal_optimizer_states) != len(primal_optimizers):
        raise ValueError("The number of primal optimizers does not match the number of primal optimizer states.")

    for primal_optimizer, primal_state in zip(primal_optimizers, primal_optimizer_states):
        primal_optimizer.load_state_dict(primal_state)

    dual_optimizer_states = cooper_optimizer_state.dual_optimizer_states
    if dual_optimizers is None:
        if dual_optimizer_states is not None:
            raise ValueError("Optimizer state dict contains `dual_optimizer_states` but `dual_optimizers` is None.")
    else:
        dual_optimizers = ensure_sequence(dual_optimizers)

        if dual_optimizer_states is None:
            raise ValueError("`dual_optimizers` were provided but state dict does not contain `dual_optimizer_states`.")

        if len(dual_optimizer_states) != len(dual_optimizers):
            raise ValueError("The number of dual optimizers does not match the number of dual optimizer states.")

        for dual_optimizer, dual_state in zip(dual_optimizers, dual_optimizer_states):
            dual_optimizer.load_state_dict(dual_state)

    if multipliers is not None:
        multipliers = ensure_sequence(multipliers)
    else:
        multipliers = [constraint.multiplier for constraint in ensure_sequence(constraint_groups)]

    if len(primal_optimizer_states) != len(multipliers):
        raise ValueError("The number of multipliers does not match the number of multiplier states.")

    if len(multipliers) > 0:
        if cooper_optimizer_state.multiplier_states is None:
            raise ValueError("Unable to load multiplier states since state dict does not contain `multiplier_states`.")

        for multiplier, multiplier_state in zip(multipliers, cooper_optimizer_state.multiplier_states):
            multiplier.load_state_dict(multiplier_state)

    # Since we have extracted the multiplier information above, we discard the constraint_groups below
    return create_optimizer_from_kwargs(
        primal_optimizers=primal_optimizers,
        extrapolation=cooper_optimizer_state.extrapolation,
        alternating=cooper_optimizer_state.alternating,
        dual_optimizers=dual_optimizers,
        constraint_groups=None,
        multipliers=multipliers,
    )