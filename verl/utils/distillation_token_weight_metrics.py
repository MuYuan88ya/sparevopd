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

"""Global optimizer-mini-batch metrics for distillation token weights."""

from __future__ import annotations

import math
from typing import Optional

import torch

_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
_QUANTILE_NAMES = ("p10", "p25", "p50", "p75", "p90", "p99")


def _world_size(process_group=None) -> int:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(group=process_group)


def _rank(process_group=None) -> int:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank(group=process_group)


def _all_reduce_(tensor: torch.Tensor, op: torch.distributed.ReduceOp, process_group=None) -> None:
    if _world_size(process_group) > 1:
        torch.distributed.all_reduce(tensor, op=op, group=process_group)


class DistillationTokenWeightMetricsAccumulator:
    """Collect exact moments and a bounded deterministic sample of valid token weights."""

    def __init__(
        self,
        *,
        mode: str,
        global_token_count: torch.Tensor,
        sample_size: int = 2048,
        sample_seed: int = 0,
        vocab_size: Optional[int] = None,
        process_group=None,
    ) -> None:
        if mode not in {"visual_advantage", "teacher_entropy"}:
            raise ValueError(f"Unsupported distillation token weight mode: {mode!r}")
        if global_token_count.ndim != 0 or global_token_count.item() <= 0:
            raise ValueError("global_token_count must be a positive scalar tensor")
        if sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got {sample_size}")
        if mode == "teacher_entropy" and (vocab_size is None or vocab_size <= 1):
            raise ValueError("teacher_entropy metrics require vocab_size > 1")

        self.mode = mode
        self.global_token_count = global_token_count.detach().to(dtype=torch.float64)
        self.sample_size = int(sample_size)
        self.sample_seed = int(sample_seed)
        self.vocab_size = vocab_size
        self.process_group = process_group
        self._device = self.global_token_count.device
        # count, sum(w), sum(w^2), sum(loss), sum(loss^2), sum(w*loss), zero(w),
        # sum(source), sum(source^2), positive(source), sum(pos_logp), sum(neg_logp)
        self._sums = torch.zeros(12, dtype=torch.float64, device=self._device)
        self._weight_min = torch.tensor(float("inf"), dtype=torch.float64, device=self._device)
        self._weight_max = torch.tensor(float("-inf"), dtype=torch.float64, device=self._device)
        self._source_min = torch.tensor(float("inf"), dtype=torch.float64, device=self._device)
        self._source_max = torch.tensor(float("-inf"), dtype=torch.float64, device=self._device)
        self._sample_keys = torch.empty(0, dtype=torch.int64, device=self._device)
        self._sample_values = torch.empty((0, 3), dtype=torch.float32, device=self._device)
        self._local_token_offset = 0
        self._has_teacher_log_probs = mode == "visual_advantage"
        self._finalized = False

    def add(
        self,
        *,
        token_weights: torch.Tensor,
        raw_per_token_loss: torch.Tensor,
        loss_mask: torch.Tensor,
        source_signal: torch.Tensor,
        positive_teacher_log_probs: Optional[torch.Tensor] = None,
        negative_teacher_log_probs: Optional[torch.Tensor] = None,
    ) -> None:
        if self._finalized:
            raise RuntimeError("DistillationTokenWeightMetricsAccumulator cannot be updated after finalize()")
        expected_shape = raw_per_token_loss.shape
        named_tensors = {
            "token_weights": token_weights,
            "loss_mask": loss_mask,
            "source_signal": source_signal,
        }
        for name, tensor in named_tensors.items():
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"{name} must match raw_per_token_loss shape {tuple(expected_shape)}, "
                    f"got {tuple(tensor.shape)}"
                )
        if self._has_teacher_log_probs:
            if positive_teacher_log_probs is None or negative_teacher_log_probs is None:
                raise ValueError("visual_advantage metrics require positive and negative teacher log-probs")
            if positive_teacher_log_probs.shape != expected_shape or negative_teacher_log_probs.shape != expected_shape:
                raise ValueError("Teacher log-probs must match raw_per_token_loss shape")

        valid_mask = loss_mask.detach().to(dtype=torch.bool)
        weights = token_weights.detach()[valid_mask].to(device=self._device, dtype=torch.float64)
        losses = raw_per_token_loss.detach()[valid_mask].to(device=self._device, dtype=torch.float64)
        source = source_signal.detach()[valid_mask].to(device=self._device, dtype=torch.float64)
        if weights.numel() == 0:
            return
        for name, values in (("token weights", weights), ("raw token losses", losses), ("source signal", source)):
            if not torch.isfinite(values).all().item():
                raise ValueError(f"Distillation {name} contain non-finite values")
        if torch.any(weights < 0).item():
            raise ValueError("Distillation token weights must be non-negative")

        self._sums[0] += weights.numel()
        self._sums[1] += weights.sum()
        self._sums[2] += weights.square().sum()
        self._sums[3] += losses.sum()
        self._sums[4] += losses.square().sum()
        self._sums[5] += (weights * losses).sum()
        self._sums[6] += (weights <= 0).sum()
        self._sums[7] += source.sum()
        self._sums[8] += source.square().sum()
        self._sums[9] += (source > 0).sum()
        self._weight_min = torch.minimum(self._weight_min, weights.min())
        self._weight_max = torch.maximum(self._weight_max, weights.max())
        self._source_min = torch.minimum(self._source_min, source.min())
        self._source_max = torch.maximum(self._source_max, source.max())

        if self._has_teacher_log_probs:
            positive_log_probs = positive_teacher_log_probs.detach()[valid_mask].to(
                device=self._device, dtype=torch.float64
            )
            negative_log_probs = negative_teacher_log_probs.detach()[valid_mask].to(
                device=self._device, dtype=torch.float64
            )
            if not torch.isfinite(positive_log_probs).all().item() or not torch.isfinite(
                negative_log_probs
            ).all().item():
                raise ValueError("Teacher log-probs contain non-finite values")
            self._sums[10] += positive_log_probs.sum()
            self._sums[11] += negative_log_probs.sum()

        self._update_sample(weights, losses, source)

    def _update_sample(self, weights: torch.Tensor, losses: torch.Tensor, source: torch.Tensor) -> None:
        token_indices = torch.arange(
            self._local_token_offset,
            self._local_token_offset + weights.numel(),
            dtype=torch.int64,
            device=self._device,
        )
        self._local_token_offset += weights.numel()
        rank_offset = (_rank(self.process_group) + 1) * 104729
        seed_offset = self.sample_seed % 2147483647
        priorities = (token_indices * 1103515245 + rank_offset + seed_offset) % 2147483647
        candidate_values = torch.stack((weights, losses, source), dim=-1).to(dtype=torch.float32)
        all_keys = torch.cat((self._sample_keys, priorities))
        all_values = torch.cat((self._sample_values, candidate_values), dim=0)
        keep_count = min(self.sample_size, all_keys.numel())
        selected = torch.topk(all_keys, k=keep_count, largest=True, sorted=False).indices
        self._sample_keys = all_keys[selected]
        self._sample_values = all_values[selected]

    def _gather_global_sample(self) -> torch.Tensor:
        padded_keys = torch.full((self.sample_size,), -1, dtype=torch.int64, device=self._device)
        padded_values = torch.zeros((self.sample_size, 3), dtype=torch.float32, device=self._device)
        local_size = self._sample_keys.numel()
        padded_keys[:local_size] = self._sample_keys
        padded_values[:local_size] = self._sample_values

        world_size = _world_size(self.process_group)
        if world_size == 1:
            gathered_keys = padded_keys
            gathered_values = padded_values
        else:
            key_parts = [torch.empty_like(padded_keys) for _ in range(world_size)]
            value_parts = [torch.empty_like(padded_values) for _ in range(world_size)]
            torch.distributed.all_gather(key_parts, padded_keys, group=self.process_group)
            torch.distributed.all_gather(value_parts, padded_values, group=self.process_group)
            gathered_keys = torch.cat(key_parts)
            gathered_values = torch.cat(value_parts)

        valid = gathered_keys >= 0
        gathered_keys = gathered_keys[valid]
        gathered_values = gathered_values[valid]
        keep_count = min(self.sample_size, gathered_keys.numel())
        selected = torch.topk(gathered_keys, k=keep_count, largest=True, sorted=False).indices
        return gathered_values[selected].to(dtype=torch.float64).cpu()

    def finalize(self, *, normalization_scale: float = 1.0) -> dict[str, float]:
        if self._finalized:
            raise RuntimeError("DistillationTokenWeightMetricsAccumulator.finalize() may only be called once")
        self._finalized = True
        if not math.isfinite(normalization_scale) or normalization_scale <= 0:
            raise ValueError(f"normalization_scale must be positive and finite, got {normalization_scale}")

        _all_reduce_(self._sums, torch.distributed.ReduceOp.SUM, self.process_group)
        for tensor in (self._weight_min, self._source_min):
            _all_reduce_(tensor, torch.distributed.ReduceOp.MIN, self.process_group)
        for tensor in (self._weight_max, self._source_max):
            _all_reduce_(tensor, torch.distributed.ReduceOp.MAX, self.process_group)
        sample = self._gather_global_sample()

        count, weight_sum, weight_square_sum, loss_sum, loss_square_sum, weight_loss_sum = (
            self._sums[:6].tolist()
        )
        expected_count = self.global_token_count.item()
        if not math.isclose(count, expected_count, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "Token weight metric count does not match the optimizer mini-batch: "
                f"expected {expected_count}, observed {count}"
            )

        normalized_weight_sum = normalization_scale * weight_sum
        normalized_weight_square_sum = normalization_scale**2 * weight_square_sum
        weight_mean = normalized_weight_sum / count
        weight_variance = max(normalized_weight_square_sum / count - weight_mean**2, 0.0)
        loss_mean = loss_sum / count
        loss_variance = max(loss_square_sum / count - loss_mean**2, 0.0)
        covariance = normalization_scale * weight_loss_sum / count - weight_mean * loss_mean
        correlation_denominator = math.sqrt(weight_variance * loss_variance)
        correlation = covariance / correlation_denominator if correlation_denominator > 0 else 0.0
        effective_fraction = (
            normalized_weight_sum**2 / (count * normalized_weight_square_sum)
            if normalized_weight_square_sum > 0
            else 0.0
        )
        weighted_to_raw_ratio = (
            normalization_scale * weight_loss_sum / loss_sum if abs(loss_sum) > 1e-12 else 0.0
        )

        sample_weights = sample[:, 0] * normalization_scale
        sample_losses = sample[:, 1]
        sample_source = sample[:, 2]
        quantiles = torch.quantile(sample_weights, torch.tensor(_QUANTILES, dtype=torch.float64))
        sorted_indices = torch.argsort(sample_weights, descending=True)
        top_ten_count = max(1, math.ceil(sample_weights.numel() * 0.10))
        top_one_count = max(1, math.ceil(sample_weights.numel() * 0.01))
        sampled_weight_sum = sample_weights.sum().item()
        sampled_loss_sum = sample_losses.sum().item()

        metrics = {
            "self_distillation/token_weight_mean": weight_mean,
            "self_distillation/token_weight_std": math.sqrt(weight_variance),
            "self_distillation/token_weight_min": self._weight_min.item() * normalization_scale,
            "self_distillation/token_weight_max": self._weight_max.item() * normalization_scale,
            "self_distillation/token_weight_zero_fraction": self._sums[6].item() / count,
            "self_distillation/token_weight_below_0_1_fraction": (sample_weights < 0.1).double().mean().item(),
            "self_distillation/token_weight_below_0_5_fraction": (sample_weights < 0.5).double().mean().item(),
            "self_distillation/token_weight_above_2_fraction": (sample_weights > 2.0).double().mean().item(),
            "self_distillation/token_weight_above_5_fraction": (sample_weights > 5.0).double().mean().item(),
            "self_distillation/token_weight_effective_token_fraction": effective_fraction,
            "self_distillation/token_weight_top_10pct_mass_fraction": (
                sample_weights[sorted_indices[:top_ten_count]].sum().item() / sampled_weight_sum
                if sampled_weight_sum > 0
                else 0.0
            ),
            "self_distillation/token_weight_top_1pct_mass_fraction": (
                sample_weights[sorted_indices[:top_one_count]].sum().item() / sampled_weight_sum
                if sampled_weight_sum > 0
                else 0.0
            ),
            "self_distillation/token_weight_jsd_pearson": correlation,
            "self_distillation/weighted_to_raw_jsd_ratio": weighted_to_raw_ratio,
            "self_distillation/top_weight_decile_loss_fraction": (
                sample_losses[sorted_indices[:top_ten_count]].sum().item() / sampled_loss_sum
                if abs(sampled_loss_sum) > 1e-12
                else 0.0
            ),
        }
        metrics.update(
            {
                f"self_distillation/token_weight_{name}": value.item()
                for name, value in zip(_QUANTILE_NAMES, quantiles, strict=True)
            }
        )
        metrics.update(self._mode_metrics(sample_source, count))
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"Distillation token weight metrics contain non-finite values: {metrics}")
        return metrics

    def _mode_metrics(self, sample_source: torch.Tensor, count: float) -> dict[str, float]:
        source_sum = self._sums[7].item()
        source_square_sum = self._sums[8].item()
        source_mean = source_sum / count
        source_std = math.sqrt(max(source_square_sum / count - source_mean**2, 0.0))
        source_quantiles = torch.quantile(sample_source, torch.tensor(_QUANTILES, dtype=torch.float64))
        if self.mode == "visual_advantage":
            positive_log_prob_mean = self._sums[10].item() / count
            negative_log_prob_mean = self._sums[11].item() / count
            metrics = {
                "self_distillation/visual_advantage_margin_mean": source_mean,
                "self_distillation/visual_advantage_margin_std": source_std,
                "self_distillation/visual_advantage_margin_min": self._source_min.item(),
                "self_distillation/visual_advantage_margin_max": self._source_max.item(),
                "self_distillation/positive_teacher_win_fraction": self._sums[9].item() / count,
                "self_distillation/positive_teacher_log_prob_mean": positive_log_prob_mean,
                "self_distillation/negative_teacher_log_prob_mean": negative_log_prob_mean,
                "self_distillation/positive_negative_log_prob_gap_mean": (
                    positive_log_prob_mean - negative_log_prob_mean
                ),
            }
            metrics.update(
                {
                    f"self_distillation/visual_advantage_margin_{name}": value.item()
                    for name, value in zip(_QUANTILE_NAMES, source_quantiles, strict=True)
                }
            )
            return metrics

        max_entropy = math.log(self.vocab_size)
        normalized_source = sample_source / max_entropy
        metrics = {
            "self_distillation/teacher_entropy_std": source_std,
            "self_distillation/teacher_normalized_entropy_mean": source_mean / max_entropy,
            "self_distillation/teacher_low_entropy_fraction": (normalized_source <= 0.25).double().mean().item(),
            "self_distillation/teacher_medium_entropy_fraction": (
                ((normalized_source > 0.25) & (normalized_source <= 0.60)).double().mean().item()
            ),
            "self_distillation/teacher_high_entropy_fraction": (normalized_source > 0.60).double().mean().item(),
        }
        metrics.update(
            {
                f"self_distillation/teacher_entropy_{name}": value.item()
                for name, value in zip(_QUANTILE_NAMES, source_quantiles, strict=True)
            }
        )
        return metrics
