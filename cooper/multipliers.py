#!/usr/bin/env python

"""Classes for modeling dual variables (e.g. Lagrange multipliers)."""

import abc

import torch


class BaseMultiplier(torch.nn.Module, metaclass=abc.ABCMeta):
    """
    Base class for Lagrange multipliers. This base class can be extended to
    different types of multipliers: Dense, Sparse or implicit multipliers.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    @abc.abstractmethod
    def parameters(self):
        """Returns the parameters associated with the multiplier."""
        pass

    @property
    @abc.abstractmethod
    def grad(self):
        """
        Returns the gradient of trainable parameters associated with the
        multipliers. In the case of implicit multipliers, this corresponds to
        the gradient with respect to the parameters of the model which predicts
        the multiplier values.
        """
        pass

    @abc.abstractmethod
    def forward(self):
        """
        Returns the *actual* value of the multipliers. When using implicit
        multipliers, the signature of this method may be change to enable
        passing the "features" of the constraint to predict the corresponding
        multiplier.
        """
        pass

    @abc.abstractmethod
    def project_(self):
        """
        In-place projection function for multipliers.
        """
        pass

    @abc.abstractmethod
    def restart_if_feasible_(self):
        """
        In-place restart function for multipliers.
        """
        pass


class DenseMultiplier(BaseMultiplier):
    """
    A dense multiplier. Holds a :py:class:`~torch.nn.parameter.Parameter`,
    which contains the value of the Lagrange multipliers associated with the
    equality or inequality constraints of a
    :py:class:`~cooper.problem.ConstrainedMinimizationProblem`.

    Args:
        init: Initial value of the multiplier.
        positive: Whether to enforce non-negativity on the values of the
            multiplier.
    """

    def __init__(self, init: torch.Tensor, *, positive: bool = False):
        super().__init__()

        # Create one Parameter for each individual dual variable. The dual
        # optimizer will consider a parameter group for each variable, which
        # allows it to keep track of separate states for different variables.
        # This is useful when employing dual restarts on some dual variables and
        # not others
        self._parameters = [torch.nn.Parameter(_) for _ in init.flatten()]

        self.shape = init.shape
        self.device = init.device

        self.positive = positive

    @property
    def parameters(self):
        """Returns the parameters associated with the multiplier."""
        return self._parameters

    @property
    def data(self):
        """Returns current values stored in the multiplier tensor."""
        values = [_.data for _ in self._parameters]
        return torch.stack(values).unflatten(0, self.shape)

    @property
    def grad(self):
        """Returns current gradient stored in the multiplier tensor."""
        grads = [_.grad for _ in self._parameters]
        return torch.stack(grads).unflatten(0, self.shape)

    @property
    def feasible_dict(self) -> torch.Tensor:
        """
        Returns a dict of multipliers associated with constraints that are
        currently feasible.
        """

        assert (
            self.positive
        ), "Feasibility can only be checked for inequality constraints"

        mask = self.grad > 0
        as_dict = {p: mask[i].item() for i, p in enumerate(self._parameters)}

        return as_dict

    def forward(self):
        """Return the current value of the multiplier."""
        return torch.stack(self._parameters).unflatten(0, self.shape)

    def project_(self):
        """
        Ensures multipliers associated with inequality constraints reamain
        non-negative.
        """
        if self.positive:
            for param in self._parameters:
                param.data = torch.relu(param.data)

    def restart_if_feasible_(self):
        """
        In-place restart function for multipliers.
        """

        assert self.positive, "Restarts is only supported for inequality multipliers"

        # Call to formulation._populate_gradients has already flipped sign
        # A currently *positive* gradient means original defect is negative, so
        # the constraint is being satisfied.

        # The code below still works in the case of proxy constraints, since the
        # multiplier updates are computed based on *non-proxy* constraints
        feasible_dict = self.feasible_dict

        for param, is_feasible in feasible_dict.items():
            if is_feasible:
                assert len(param.shape) == 0
                param.data.fill_(0.0)
                param.grad.fill_(0.0)

    def __str__(self):
        return str(self.forward().data)

    def __repr__(self):
        pos_str = "inequality" if self.positive else "equality"
        rep = "DenseMultiplier(" + pos_str + ", " + str(self.forward().data) + ")"
        return rep

    def __members(self):
        return (self.positive, self.forward())

    def __hash__(self):
        return hash(self.__members())

    def __eq__(self, other):

        if type(other) is type(self):

            self_positive, self_weight = self.__members()
            other_positive, other_weight = other.__members()

            positive_check = self_positive == other_positive
            weight_check = torch.allclose(self_weight, other_weight)

            return positive_check and weight_check

        else:
            return False
