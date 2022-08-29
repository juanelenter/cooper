#!/usr/bin/env python

"""Tests for Multiplier class."""

import torch

import cooper


def test_multipliers_init():

    init_tensor = torch.randn(100, 1)
    multiplier = cooper.multipliers.DenseMultiplier(init_tensor)
    multiplier.project_()
    assert torch.allclose(multiplier(), init_tensor)

    init_tensor = torch.tensor([0.0, -1.0, 2.5])
    multiplier = cooper.multipliers.DenseMultiplier(init_tensor)
    multiplier.project_()
    assert torch.allclose(multiplier(), init_tensor)


def test_custom_projection():
    class CustomProjectionMultiplier(cooper.multipliers.DenseMultiplier):
        def project_(self):
            # Project multipliers so that maximum non-zero entry is exactly 1
            max_entry = torch.relu(self()).max()
            if max_entry > 0:
                for param in self.parameters:
                    param.data = param.data / max_entry

    init_tensor = torch.randn(100, 1)
    multiplier = CustomProjectionMultiplier(init_tensor)

    multiplier.project_()
    assert torch.allclose(multiplier().data.max(), torch.tensor([1.0]))
