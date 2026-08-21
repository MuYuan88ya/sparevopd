# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Global mini-batch normalization for teacher-entropy token weights."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn


def _distributed_world_size(process_group=None) -> int:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(group=process_group)


def _all_reduce_(tensor: torch.Tensor, op: torch.distributed.ReduceOp, process_group=None) -> None:
    if _distributed_world_size(process_group) > 1:
        torch.distributed.all_reduce(tensor, op=op, group=process_group)


def global_valid_token_count(
    loss_mask: torch.Tensor,
    *,
    reduction_device: Optional[torch.device | str | int] = None,
    process_group=None,
) -> torch.Tensor:
    """Return the global number of valid tokens as a detached float64 scalar."""

    local_count = loss_mask.detach().to(dtype=torch.float64).sum()
    if reduction_device is not None:
        local_count = local_count.to(device=reduction_device)
    _all_reduce_(local_count, torch.distributed.ReduceOp.SUM, process_group)
    if not torch.isfinite(local_count).item() or local_count.item() <= 0:
        raise ValueError(
            "Teacher entropy reweighting requires a positive finite token count, "
            f"got {local_count.item()}"
        )
    return local_count


class TeacherEntropyWeightAccumulator:
    """Accumulate detached confidence statistics across one optimizer mini-batch."""

    def __init__(
        self,
        *,
        beta: float,
        global_token_count: torch.Tensor,
        process_group=None,
    ) -> None:
        if isinstance(beta, bool) or not isinstance(beta, (int, float)):
            raise TypeError(f"teacher_entropy_beta must be numeric, got {type(beta).__name__}")
        if not math.isfinite(float(beta)) or float(beta) < 0:
            raise ValueError(f"teacher_entropy_beta must be finite and non-negative, got {beta}")
        if global_token_count.ndim != 0:
            raise ValueError("global_token_count must be a scalar tensor")
        if not torch.isfinite(global_token_count).item() or global_token_count.item() <= 0:
            raise ValueError(
                "Teacher entropy reweighting requires a positive finite global token count, "
                f"got {global_token_count.item()}"
            )

        self.beta = float(beta)
        self.global_token_count = global_token_count.detach().to(dtype=torch.float64)
        self.process_group = process_group
        self.dp_size = _distributed_world_size(process_group)
        device = self.global_token_count.device
        self._sums = torch.zeros(5, dtype=torch.float64, device=device)
        self._entropy_min = torch.tensor(float("inf"), dtype=torch.float64, device=device)
        self._entropy_max = torch.tensor(float("-inf"), dtype=torch.float64, device=device)
        self._confidence_min = torch.tensor(float("inf"), dtype=torch.float64, device=device)
        self._confidence_max = torch.tensor(float("-inf"), dtype=torch.float64, device=device)
        self._finalized = False

    def add(self, teacher_entropy: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        """Accumulate one micro-batch and return detached unnormalized confidence weights."""

        if self._finalized:
            raise RuntimeError("TeacherEntropyWeightAccumulator cannot be updated after finalize()")
        if teacher_entropy.shape != loss_mask.shape:
            raise ValueError(
                "teacher_entropy and loss_mask must have the same shape, "
                f"got {tuple(teacher_entropy.shape)} and {tuple(loss_mask.shape)}"
            )

        entropy = teacher_entropy.detach().to(dtype=torch.float32)
        valid_mask = loss_mask.detach().to(device=entropy.device, dtype=torch.bool)
        valid_entropy = entropy[valid_mask]
        if valid_entropy.numel() == 0:
            return torch.ones_like(entropy)
        if not torch.isfinite(valid_entropy).all().item():
            raise ValueError("Teacher entropy contains non-finite values on valid SDPO tokens")
        if torch.any(valid_entropy < 0).item():
            raise ValueError("Teacher entropy must be non-negative on valid SDPO tokens")

        valid_confidence = torch.exp(-self.beta * valid_entropy)
        if not torch.isfinite(valid_confidence).all().item():
            raise ValueError("Teacher entropy confidence contains non-finite values")

        stats_device = self.global_token_count.device
        valid_entropy_stats = valid_entropy.to(device=stats_device, dtype=torch.float64)
        valid_confidence_stats = valid_confidence.to(device=stats_device, dtype=torch.float64)
        self._sums[0] += valid_entropy_stats.numel()
        self._sums[1] += valid_entropy_stats.sum()
        self._sums[2] += valid_confidence_stats.sum()
        self._sums[3] += valid_confidence_stats.square().sum()
        self._entropy_min = torch.minimum(self._entropy_min, valid_entropy_stats.min())
        self._entropy_max = torch.maximum(self._entropy_max, valid_entropy_stats.max())
        self._confidence_min = torch.minimum(self._confidence_min, valid_confidence_stats.min())
        self._confidence_max = torch.maximum(self._confidence_max, valid_confidence_stats.max())

        confidence = torch.ones_like(entropy)
        confidence[valid_mask] = valid_confidence
        return confidence.detach()

    def add_weighted_loss_sum(self, weighted_loss_sum: torch.Tensor | float) -> None:
        """Accumulate ``sum(confidence * corrected_jsd)`` for exact global metrics."""

        if self._finalized:
            raise RuntimeError("TeacherEntropyWeightAccumulator cannot be updated after finalize()")
        value = torch.as_tensor(
            weighted_loss_sum,
            dtype=torch.float64,
            device=self.global_token_count.device,
        ).detach()
        if value.ndim != 0 or not torch.isfinite(value).item():
            raise ValueError(f"weighted_loss_sum must be a finite scalar, got {value}")
        self._sums[4] += value

    def finalize(self) -> tuple[float, dict[str, float]]:
        """All-reduce statistics and return the gradient scale and global metrics."""

        if self._finalized:
            raise RuntimeError("TeacherEntropyWeightAccumulator.finalize() may only be called once")
        self._finalized = True

        _all_reduce_(self._sums, torch.distributed.ReduceOp.SUM, self.process_group)
        for tensor in (self._entropy_min, self._confidence_min):
            _all_reduce_(tensor, torch.distributed.ReduceOp.MIN, self.process_group)
        for tensor in (self._entropy_max, self._confidence_max):
            _all_reduce_(tensor, torch.distributed.ReduceOp.MAX, self.process_group)

        observed_count, entropy_sum, confidence_sum, confidence_square_sum, weighted_loss_sum = (
            value.item() for value in self._sums
        )
        expected_count = self.global_token_count.item()
        if not math.isclose(observed_count, expected_count, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "Teacher entropy token accounting does not match the optimizer mini-batch: "
                f"expected {expected_count}, observed {observed_count}"
            )
        if not math.isfinite(confidence_sum) or confidence_sum <= 0:
            raise ValueError(f"Teacher entropy confidence sum must be positive and finite, got {confidence_sum}")
        if not math.isfinite(confidence_square_sum) or confidence_square_sum <= 0:
            raise ValueError(
                "Teacher entropy squared-confidence sum must be positive and finite, "
                f"got {confidence_square_sum}"
            )

        normalization_scale = expected_count / confidence_sum
        normalized_min = self._confidence_min.item() * normalization_scale
        normalized_max = self._confidence_max.item() * normalization_scale
        normalized_second_moment = (
            normalization_scale * normalization_scale * confidence_square_sum / expected_count
        )
        normalized_std = math.sqrt(max(normalized_second_moment - 1.0, 0.0))
        effective_token_fraction = confidence_sum * confidence_sum / (
            expected_count * confidence_square_sum
        )
        metrics = {
            "self_distillation/teacher_entropy_mean": entropy_sum / expected_count,
            "self_distillation/teacher_entropy_min": self._entropy_min.item(),
            "self_distillation/teacher_entropy_max": self._entropy_max.item(),
            "self_distillation/entropy_weight_min": normalized_min,
            "self_distillation/entropy_weight_max": normalized_max,
            "self_distillation/entropy_weight_std": normalized_std,
            "self_distillation/entropy_weight_effective_token_fraction": effective_token_fraction,
            "self_distillation/entropy_weight_normalization_scale": normalization_scale,
            "self_distillation/entropy_weighted_jsd_token_mean": weighted_loss_sum / confidence_sum,
            "self_distillation/entropy_beta": self.beta,
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"Teacher entropy reweighting produced non-finite metrics: {metrics}")
        return normalization_scale, metrics


def scale_gradients_(parameters: Iterable[nn.Parameter], scale: float) -> None:
    """Scale accumulated gradients in-place before unscale/clip/optimizer step."""

    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError(f"Gradient normalization scale must be positive and finite, got {scale}")
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(float(scale))
