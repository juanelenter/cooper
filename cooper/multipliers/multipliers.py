"""Classes for modeling dual variables (e.g. Lagrange multipliers)."""
import abc

import torch


class ConstantMultiplier:
    """
    Constant (non-trainable) multiplier class used for penalized formulations.

    Args:
        init: Value of the multiplier.
    """

    def __init__(self, init: torch.Tensor):
        if init.requires_grad:
            raise ValueError("Constant multiplier should not be trainable.")
        self.weight = init
        self.device = init.device

    def __call__(self):
        """Return the current value of the multiplier."""
        return self.weight

    def state_dict(self):
        return {"weight": self.weight}

    def load_state_dict(self, state_dict):
        self.weight = state_dict["weight"]


class ExplicitMultiplier(torch.nn.Module):
    """
    A dense multiplier. Holds a :py:class:`~torch.nn.parameter.Parameter`, which
    contains the value of the Lagrange multipliers associated with the equality or
    inequality constraints of a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`.

    Args:
        init: Initial value of the multiplier.
        positive: Whether to enforce non-negativity on the values of the multiplier.
    """

    def __init__(self, init: torch.Tensor, *, positive: bool = False):
        super().__init__()
        self.weight = torch.nn.Parameter(init)
        self.positive = positive
        self.device = self.weight.device

    def project_(self):
        """
        Ensures non-negativity for multipliers associated with inequality constraints.
        """
        if self.positive:
            self.weight.relu_()

    def restart_if_feasible_(self, feasible_indices: torch.Tensor):
        """
        In-place restart function for multipliers.

        Args:
            feasible_indices: Indices or binary masks denoting the feasible constraints.
        """

        if not self.positive:
            raise RuntimeError("Restarts are only supported for inequality multipliers")

        self.weight.data[feasible_indices, ...] = 0.0
        if self.weight.grad is not None:
            self.weight.grad[feasible_indices, ...] = 0.0

    def state_dict(self):
        _state_dict = super().state_dict()
        _state_dict["positive"] = self.positive
        return _state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.positive = state_dict["positive"]
        self.device = self.weight.device


class DenseMultiplier(ExplicitMultiplier):
    def forward(self):
        """Return the current value of the multiplier."""
        return self.weight

    def __repr__(self):
        constraint_type = "ineq" if self.positive else "eq"
        rep = f"DenseMultiplier({constraint_type}, shape={self.weight.shape})"
        return rep


class SparseMultiplier(ExplicitMultiplier):
    def forward(self, indices: torch.Tensor):
        """Return the current value of the multiplier at the provided indices."""
        return torch.nn.functional.embedding(indices, self.weight, sparse=True).squeeze()

    def __repr__(self):
        constraint_type = "ineq" if self.positive else "eq"
        rep = f"SparseMultiplier({constraint_type}, shape={self.weight.shape})"
        return rep


class ImplicitMultiplier(torch.nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self):
        pass
