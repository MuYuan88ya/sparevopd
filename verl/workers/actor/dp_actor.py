# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Single Process Actor
"""

import logging
import math
import os
import time
from types import SimpleNamespace
from typing import Any, Optional

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_self_distillation_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.distillation_token_weight_metrics import DistillationTokenWeightMetricsAccumulator
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.metric import AggregationType, Metric, reduce_metrics
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.teacher_entropy_reweighting import (
    TeacherEntropyWeightAccumulator,
    global_valid_token_count,
    scale_gradients_,
)
from verl.utils.teacher_visual_token_mask import (
    VisualTokenSelectionResult,
    VisualTokenSelectionSession,
    apply_image_token_keep_mask,
    create_visual_token_selection_strategy,
)
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, slice_input_tensor, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TrustRegionTeacher(nn.Module):
    def __init__(self, ref_module: nn.Module, student_module: nn.Module, mix_coef: float) -> None:
        super().__init__()
        self.ref_module = ref_module
        self.student_module = student_module
        self.mix_coef = float(mix_coef)

    def forward(self, *args, **kwargs):
        ref_out = self.ref_module(*args, **kwargs)
        student_out = self.student_module(*args, **kwargs)
        ref_logits = ref_out.logits if hasattr(ref_out, "logits") else ref_out[0]
        student_logits = student_out.logits if hasattr(student_out, "logits") else student_out[0]
        logits = torch.lerp(ref_logits, student_logits, self.mix_coef)
        return SimpleNamespace(logits=logits)


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.teacher_module: Optional[nn.Module] = None
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        self.teacher_visual_token_selection_strategy = None
        self.teacher_visual_image_token_id = None
        self.teacher_visual_spatial_merge_size = None
        self.contrastive_visual_advantage_enabled = False
        self.contrastive_negative_mode = "complement-random-equal"
        self.contrastive_negative_seed = 0
        self.teacher_entropy_reweighting_enabled = False
        self.teacher_entropy_beta = 1.0
        self.teacher_vocab_size = None
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        if self_distillation_cfg is not None:
            self.contrastive_visual_advantage_enabled = bool(
                self_distillation_cfg.get("contrastive_visual_advantage", False)
            )
            self.contrastive_negative_mode = self_distillation_cfg.get(
                "contrastive_negative_mode", "complement-random-equal"
            )
            self.contrastive_negative_seed = int(
                self_distillation_cfg.get("contrastive_negative_seed", 0)
            )
            self.teacher_entropy_reweighting_enabled = bool(
                self_distillation_cfg.get("teacher_entropy_reweighting", False)
            )
            self.teacher_entropy_beta = float(self_distillation_cfg.get("teacher_entropy_beta", 1.0))
            if not math.isfinite(self.teacher_entropy_beta) or self.teacher_entropy_beta < 0:
                raise ValueError(
                    "teacher_entropy_beta must be finite and non-negative, "
                    f"got {self.teacher_entropy_beta}"
                )
            if self.teacher_entropy_reweighting_enabled and self.contrastive_visual_advantage_enabled:
                raise ValueError(
                    "teacher_entropy_reweighting and contrastive_visual_advantage are mutually exclusive"
                )
            if self.contrastive_visual_advantage_enabled and self_distillation_cfg.get(
                "teacher_visual_token_mode", "crop"
            ) != "same_image_bbox_mask":
                raise ValueError(
                    "contrastive_visual_advantage requires "
                    "teacher_visual_token_mode='same_image_bbox_mask'"
                )
            if self.teacher_entropy_reweighting_enabled:
                if self_distillation_cfg.get("teacher_visual_token_mode", "crop") != "same_image_bbox_mask":
                    raise ValueError(
                        "teacher_entropy_reweighting requires "
                        "teacher_visual_token_mode='same_image_bbox_mask'"
                    )
                if self.config.policy_loss.get("loss_mode", "vanilla") != "vopd":
                    raise ValueError("teacher_entropy_reweighting requires policy_loss.loss_mode='vopd'")
                if self.config.loss_agg_mode != "token-mean":
                    raise ValueError("teacher_entropy_reweighting requires actor.loss_agg_mode='token-mean'")
                if self.config.entropy_coeff != 0:
                    raise ValueError("teacher_entropy_reweighting requires actor.entropy_coeff=0")
                if self.config.use_kl_loss:
                    raise ValueError("teacher_entropy_reweighting requires actor.use_kl_loss=False")
                self.teacher_vocab_size = self._resolve_actor_vocab_size()
        if (
            self_distillation_cfg is not None
            and self_distillation_cfg.get("teacher_visual_token_mode", "crop") == "same_image_bbox_mask"
        ):
            if not self.use_remove_padding:
                raise ValueError("same_image_bbox_mask requires actor.use_remove_padding=True")
            if not bool(self_distillation_cfg.get("teacher_always_on", False)):
                raise ValueError("same_image_bbox_mask requires self_distillation.teacher_always_on=True")
            if self.use_ulysses_sp:
                raise ValueError("same_image_bbox_mask does not support ulysses sequence parallelism")
            strategy_name = self_distillation_cfg.get(
                "teacher_visual_token_selection_strategy", "fixed_stratified"
            )
            strategy_kwargs = self_distillation_cfg.get("teacher_visual_token_selection_kwargs", {})
            if self.contrastive_visual_advantage_enabled:
                if strategy_name != "fixed_stratified":
                    raise ValueError(
                        "contrastive_visual_advantage requires "
                        "teacher_visual_token_selection_strategy='fixed_stratified'"
                    )
                background_keep_ratio = float(strategy_kwargs.get("background_keep_ratio", 0.1))
                if background_keep_ratio > 0.30:
                    raise ValueError(
                        "contrastive sparse teachers require background_keep_ratio <= 0.30, "
                        f"got {background_keep_ratio}"
                    )
                if self.contrastive_negative_mode != "complement-random-equal":
                    raise ValueError(
                        "contrastive_negative_mode must be 'complement-random-equal', "
                        f"got {self.contrastive_negative_mode!r}"
                    )
            self.teacher_visual_token_selection_strategy = create_visual_token_selection_strategy(
                strategy_name,
                strategy_kwargs,
            )
            (
                self.teacher_visual_image_token_id,
                self.teacher_visual_spatial_merge_size,
            ) = self._resolve_actor_visual_constants()

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

    def _update_teacher(self) -> None:
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        if not self_distillation_cfg or loss_mode != "vopd":
            return
        teacher_model_source = getattr(self_distillation_cfg, "teacher_model_source", "legacy")
        if teacher_model_source != "legacy":
            return
        teacher_regularization = getattr(self_distillation_cfg, "teacher_regularization", "ema")
        if self.teacher_module is None or self.teacher_module is self.actor_module:
            raise ValueError("Teacher updates require a separate teacher_module in the actor worker.")
        with torch.no_grad():
            if teacher_regularization == "ema":
                update_rate = getattr(self_distillation_cfg, "teacher_update_rate", 0.0)
                if update_rate == 0.0:
                    return
                for teacher_param, student_param in zip(
                    self.teacher_module.parameters(),
                    self.actor_module.parameters(),
                    strict=True,
                ):
                    student_data = student_param.data.to(device=teacher_param.device)
                    teacher_param.data.mul_(1.0 - update_rate).add_(student_data, alpha=update_rate)
                return

            if teacher_regularization == "progressive":
                teacher_update_interval = getattr(self_distillation_cfg, "teacher_update_interval", None)
                if teacher_update_interval is None:
                    raise ValueError("Progressive teacher requires self_distillation.teacher_update_interval.")
                global_steps = getattr(self, "_current_global_steps", None)
                if global_steps is None or global_steps % teacher_update_interval != 0:
                    return
                for teacher_param, student_param in zip(
                    self.teacher_module.parameters(),
                    self.actor_module.parameters(),
                    strict=True,
                ):
                    teacher_param.data.copy_(student_param.data.to(device=teacher_param.device))
                for teacher_buffer, student_buffer in zip(
                    self.teacher_module.buffers(),
                    self.actor_module.buffers(),
                    strict=True,
                ):
                    teacher_buffer.data.copy_(student_buffer.data.to(device=teacher_buffer.device))
                return

            return

    @staticmethod
    def _has_non_empty_multi_modal_inputs(multi_modal_inputs) -> bool:
        if multi_modal_inputs is None:
            return False
        for inputs in multi_modal_inputs:
            if inputs is None:
                continue
            inputs = getattr(inputs, "data", inputs)
            if isinstance(inputs, dict):
                if not inputs:
                    continue
                for value in inputs.values():
                    if value is None:
                        continue
                    if isinstance(value, torch.Tensor) and value.numel() == 0:
                        continue
                    return True
            else:
                return True
        return False

    def _resolve_actor_visual_constants(self) -> tuple[int, int]:
        candidates = [self.actor_module]
        visited = set()
        image_token_id = None
        spatial_merge_size = None
        while candidates:
            module = candidates.pop(0)
            if module is None or id(module) in visited:
                continue
            visited.add(id(module))
            config = getattr(module, "config", None)
            if image_token_id is None:
                image_token_id = getattr(config, "image_token_id", None)
            if spatial_merge_size is None:
                vision_config = getattr(config, "vision_config", None)
                spatial_merge_size = getattr(vision_config, "spatial_merge_size", None)
            if image_token_id is not None and spatial_merge_size is not None:
                return int(image_token_id), int(spatial_merge_size)
            for attr_name in ("module", "_fsdp_wrapped_module", "base_model", "model"):
                child = getattr(module, attr_name, None)
                if child is not None and child is not module:
                    candidates.append(child)
        raise ValueError(
            "Visual token selection requires model config image_token_id and "
            "vision_config.spatial_merge_size"
        )

    def _resolve_actor_vocab_size(self) -> int:
        candidates = [self.actor_module]
        visited = set()
        while candidates:
            module = candidates.pop(0)
            if module is None or id(module) in visited:
                continue
            visited.add(id(module))
            config = getattr(module, "config", None)
            text_config = getattr(config, "text_config", None)
            vocab_size = getattr(text_config, "vocab_size", None) or getattr(config, "vocab_size", None)
            if vocab_size is not None:
                return int(vocab_size)
            for attr_name in ("module", "_fsdp_wrapped_module", "base_model", "model"):
                child = getattr(module, attr_name, None)
                if child is not None and child is not module:
                    candidates.append(child)
        raise ValueError("Unable to resolve vocab_size from the actor model")

    @staticmethod
    def _as_per_sample_inputs(raw_multi_modal_inputs, batch_size: int, *, context: str) -> list[dict]:
        if raw_multi_modal_inputs is None:
            raise ValueError(f"{context} requires multi_modal_inputs")
        if isinstance(raw_multi_modal_inputs, dict):
            raw_multi_modal_inputs = [raw_multi_modal_inputs]
        else:
            raw_multi_modal_inputs = list(raw_multi_modal_inputs)
        if len(raw_multi_modal_inputs) != batch_size:
            raise ValueError(
                f"{context} multimodal inputs must contain one item per sample, "
                f"got items={len(raw_multi_modal_inputs)} and batch={batch_size}"
            )

        normalized_inputs = []
        for sample_idx, inputs in enumerate(raw_multi_modal_inputs):
            inputs = getattr(inputs, "data", inputs)
            if not isinstance(inputs, dict):
                raise TypeError(f"{context} multi_modal_inputs[{sample_idx}] must be a dict, got {type(inputs)}")
            normalized_inputs.append(dict(inputs))
        return normalized_inputs

    def _build_visual_token_selection_session(
        self,
        selection_metadata,
        attention_mask: torch.Tensor,
    ) -> VisualTokenSelectionSession:
        if self.teacher_visual_token_selection_strategy is None:
            raise ValueError("Teacher visual token selection strategy was not initialized")
        if self.teacher_visual_spatial_merge_size is None:
            raise ValueError("Teacher visual spatial merge size was not initialized")
        if selection_metadata is None:
            raise ValueError("Teacher visual token selection metadata is missing")
        if attention_mask.ndim != 2:
            raise ValueError(
                "Teacher visual token selection attention_mask must have shape [batch, sequence], "
                f"got {tuple(attention_mask.shape)}"
            )
        if isinstance(selection_metadata, dict):
            selection_metadata = [selection_metadata]
        else:
            selection_metadata = list(selection_metadata)
        batch_size = attention_mask.shape[0]
        if len(selection_metadata) != batch_size:
            raise ValueError(
                "Teacher visual token selection metadata must contain one item per sample, "
                f"got items={len(selection_metadata)} and batch={batch_size}"
            )
        normalized_metadata = []
        for sample_idx, raw_metadata in enumerate(selection_metadata):
            metadata = getattr(raw_metadata, "data", raw_metadata)
            if not isinstance(metadata, dict):
                raise TypeError(
                    f"Teacher visual token selection metadata[{sample_idx}] must be a dict, got {type(metadata)}"
                )
            normalized_metadata.append(dict(metadata))
        valid_sequence_lengths = attention_mask.sum(dim=-1).detach().cpu().tolist()
        return VisualTokenSelectionSession(
            strategy=self.teacher_visual_token_selection_strategy,
            metadata=normalized_metadata,
            valid_sequence_lengths=valid_sequence_lengths,
            spatial_merge_size=self.teacher_visual_spatial_merge_size,
            contrastive_negative_mode=(
                getattr(self, "contrastive_negative_mode", "complement-random-equal")
                if getattr(self, "contrastive_visual_advantage_enabled", False)
                else None
            ),
            contrastive_negative_seed=getattr(self, "contrastive_negative_seed", 0),
            global_step=(
                getattr(self, "_current_global_steps", None)
                if getattr(self, "contrastive_visual_advantage_enabled", False)
                else None
            ),
        )

    @torch.no_grad()
    def _finalize_teacher_visual_token_mask(
        self,
        teacher_inputs: dict,
        selection_result: Optional[VisualTokenSelectionResult],
        *,
        mask_kind: str = "positive",
    ) -> tuple[dict, dict[str, float]]:
        if selection_result is None:
            raise ValueError("Student forward did not produce a visual token selection result")
        if not isinstance(selection_result, VisualTokenSelectionResult):
            raise TypeError(f"Expected VisualTokenSelectionResult, got {type(selection_result)}")

        batch_size = teacher_inputs["input_ids"].shape[0]
        if len(selection_result) != batch_size:
            raise ValueError(
                "Visual token selection result must contain one entry per teacher sample, "
                f"got batch={batch_size} and results={len(selection_result)}"
            )
        if mask_kind == "positive":
            selected_masks = selection_result.keep_masks
        elif mask_kind == "negative":
            if selection_result.negative_keep_masks is None:
                raise ValueError("Visual token selection result does not contain negative keep masks")
            selected_masks = selection_result.negative_keep_masks
        else:
            raise ValueError(f"mask_kind must be 'positive' or 'negative', got {mask_kind!r}")

        normalized_inputs = self._as_per_sample_inputs(
            teacher_inputs.get("multi_modal_inputs"),
            batch_size,
            context="Teacher visual token selection",
        )
        if self.teacher_visual_image_token_id is None:
            raise ValueError("Teacher visual image token id was not initialized")
        original_attention_mask = teacher_inputs["attention_mask"]
        teacher_attention_mask = original_attention_mask.clone()
        response_start_idx = teacher_inputs["response_start_idx"]
        finalized_multi_modal_inputs = []
        target_token_count = 0
        background_token_count = 0
        kept_background_token_count = 0

        pair_overlap_count = 0
        for sample_idx, (inputs, keep_mask, target_keep_mask, positive_keep_mask) in enumerate(
            zip(
                normalized_inputs,
                selected_masks,
                selection_result.target_keep_masks,
                selection_result.keep_masks,
                strict=True,
            )
        ):
            teacher_attention_mask[sample_idx] = apply_image_token_keep_mask(
                input_ids=teacher_inputs["input_ids"][sample_idx],
                attention_mask=original_attention_mask[sample_idx],
                image_token_id=self.teacher_visual_image_token_id,
                image_token_keep_mask=keep_mask,
            )
            response_start = int(response_start_idx[sample_idx].item())
            if not torch.equal(
                teacher_attention_mask[sample_idx, response_start:],
                original_attention_mask[sample_idx, response_start:],
            ):
                raise RuntimeError(f"Student-embedding selector modified response tokens for sample {sample_idx}")

            # The same detached mask is consumed by both attention unpadding and
            # Qwen's visual-feature filtering.  Qwen moves it to the visual
            # embedding device at the final injection point when necessary.
            teacher_keep_mask = keep_mask
            inputs["image_token_keep_mask"] = teacher_keep_mask
            finalized_multi_modal_inputs.append(inputs)

            sample_target_count = int(target_keep_mask.sum().item())
            sample_token_count = int(keep_mask.numel())
            sample_kept_count = int(teacher_keep_mask.sum().item())
            target_token_count += sample_target_count
            background_token_count += sample_token_count - sample_target_count
            if mask_kind == "positive":
                kept_background_token_count += sample_kept_count - sample_target_count
            else:
                kept_background_token_count += sample_kept_count
                pair_overlap_count += int(
                    (teacher_keep_mask.to(device=positive_keep_mask.device) & positive_keep_mask).sum().item()
                )

        finalized_teacher_inputs = dict(teacher_inputs)
        finalized_teacher_inputs["attention_mask"] = teacher_attention_mask
        finalized_teacher_inputs["multi_modal_inputs"] = finalized_multi_modal_inputs
        total_visual_tokens = target_token_count + background_token_count
        kept_visual_tokens = target_token_count + kept_background_token_count
        if mask_kind == "positive":
            metrics = {
                "actor/teacher_target_visual_tokens": float(target_token_count),
                "actor/teacher_background_visual_tokens": float(background_token_count),
                "actor/teacher_kept_background_visual_tokens": float(kept_background_token_count),
                "actor/teacher_kept_visual_tokens": float(kept_visual_tokens),
                "actor/teacher_visual_keep_ratio": kept_visual_tokens / max(total_visual_tokens, 1),
                "actor/teacher_llm_effective_tokens": float(teacher_attention_mask.sum().item()),
            }
            metrics.update(
                {
                    f"actor/teacher_visual_strategy/{key}": float(value)
                    for key, value in selection_result.metrics.items()
                }
            )
        else:
            metrics = {
                "actor/negative_teacher_kept_visual_tokens": float(kept_background_token_count),
                "actor/negative_teacher_visual_keep_ratio": (
                    kept_background_token_count / max(total_visual_tokens, 1)
                ),
                "actor/negative_teacher_llm_effective_tokens": float(teacher_attention_mask.sum().item()),
                "actor/teacher_visual_pair_overlap": float(pair_overlap_count),
            }
        return finalized_teacher_inputs, metrics

    @staticmethod
    def _add_tail_bucket(log_probs: torch.Tensor) -> torch.Tensor:
        log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
        log_s = torch.clamp(log_s, max=-1e-7)
        tail_log = torch.log(-torch.expm1(log_s))
        return torch.cat([log_probs, tail_log], dim=-1)

    @staticmethod
    def _build_response_positions(
        response_start_idx: torch.Tensor,
        response_length: int,
        seqlen: int,
    ) -> torch.Tensor:
        if response_start_idx.dim() != 1:
            raise ValueError(f"response_start_idx must be rank-1, got shape {tuple(response_start_idx.shape)}")
        if (response_start_idx < 1).any():
            raise ValueError("response_start_idx must be >= 1 so response logits have a preceding context token.")

        offsets = torch.arange(response_length, device=response_start_idx.device, dtype=response_start_idx.dtype)
        response_positions = response_start_idx.unsqueeze(1) - 1 + offsets.unsqueeze(0)
        if response_positions.numel() > 0:
            if response_positions.min().item() < 0 or response_positions.max().item() >= seqlen:
                raise ValueError(
                    f"Response positions out of bounds for seqlen={seqlen}: "
                    f"min={response_positions.min().item()}, max={response_positions.max().item()}"
                )
        return response_positions.to(dtype=torch.long)

    @staticmethod
    def _select_response_positions(
        hidden_states: torch.Tensor,
        response_length: int,
        response_start_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if response_start_idx is None:
            return hidden_states[:, -response_length - 1 : -1, ...]

        response_positions = DataParallelPPOActor._build_response_positions(
            response_start_idx=response_start_idx,
            response_length=response_length,
            seqlen=hidden_states.size(1),
        )
        if hidden_states.dim() == 2:
            return torch.gather(hidden_states, dim=1, index=response_positions)

        gather_index = response_positions.view(
            response_positions.size(0),
            response_positions.size(1),
            *([1] * (hidden_states.dim() - 2)),
        ).expand(response_positions.size(0), response_positions.size(1), *hidden_states.shape[2:])
        return torch.gather(hidden_states, dim=1, index=gather_index)

    def _dump_self_distillation_log_probs(
        self,
        *,
        meta_info: dict,
        self_distillation_cfg,
        dump_chunks: list[dict[str, torch.Tensor]],
    ) -> None:
        dump_root = self_distillation_cfg.get("log_prob_dump_dir", None)
        if not dump_root or not dump_chunks:
            return

        global_step = meta_info.get("global_steps")
        if global_step is None:
            return

        experiment_name = os.environ.get("EXPERIMENT", "unknown_experiment")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        normalized_root = os.path.normpath(dump_root)
        if os.path.basename(normalized_root) == experiment_name:
            save_dir = normalized_root
        else:
            save_dir = os.path.join(normalized_root, experiment_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{int(global_step)}.rank{rank}.pt")

        student_log_probs = torch.cat([chunk["student_log_probs"] for chunk in dump_chunks], dim=0)
        teacher_log_probs = torch.cat([chunk["teacher_log_probs"] for chunk in dump_chunks], dim=0)

        torch.save(
            {
                "student_log_probs": student_log_probs,
                "teacher_log_probs": teacher_log_probs,
                "global_step": int(global_step),
                "rank": rank,
                "experiment_name": experiment_name,
                "distribution_size": int(student_log_probs.shape[-1]),
                "num_valid_tokens": int(student_log_probs.shape[0]),
                "distillation_topk": self_distillation_cfg.get("distillation_topk", None),
                "distillation_add_tail": bool(self_distillation_cfg.get("distillation_add_tail", False)),
            },
            save_path,
        )

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
        return_all_logps: bool = False,
        distill_topk: Optional[int] = None,
        topk_indices: Optional[torch.Tensor] = None,
        module: Optional[nn.Module] = None,
        visual_token_selection_session: Optional[VisualTokenSelectionSession] = None,
    ) -> dict[str, Any]:
        """
        Returns:
            dict[str, Any]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
                if distill_topk or topk_indices is set:
                    topk_logps: (bs, response_len, k)
                    topk_indices: (bs, response_len, k)
                Visual token selection results are retained by the optional
                ``visual_token_selection_session``, not returned by the model.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        use_topk = distill_topk is not None or topk_indices is not None
        compute_all_logps = return_all_logps and not use_topk
        return_topk_indices = use_topk and topk_indices is None
        if (return_all_logps or use_topk) and self.use_fused_kernels:
            raise ValueError("Logit distillation requires disabling fused kernels.")

        model = module or self.actor_module

        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
                and not return_all_logps
                and not use_topk
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=model,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            response_start_idx = micro_batch.get("response_start_idx")
            if response_start_idx is not None:
                response_start_idx = response_start_idx.to(device=input_ids.device, dtype=torch.long)
                if response_start_idx.shape != (batch_size,):
                    raise ValueError(
                        f"response_start_idx shape must be ({batch_size},), got {tuple(response_start_idx.shape)}"
                    )
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(model, "module", model).config,
                        "vision_config",
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                if visual_token_selection_session is not None:
                    extra_args["visual_token_selection_session"] = visual_token_selection_session

                output = model(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating
                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)
                    all_logps_rmpad = torch.log_softmax(logits_rmpad, dim=-1) if compute_all_logps else None

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits_rmpad.shape[-1])
                            topk_logits_rmpad, topk_indices_rmpad = torch.topk(logits_rmpad, topk, dim=-1)
                        else:
                            topk = topk_indices.size(-1)
                            full_topk_indices = torch.zeros(
                                batch_size,
                                seqlen,
                                topk,
                                device=topk_indices.device,
                                dtype=topk_indices.dtype,
                            )
                            if response_start_idx is None:
                                full_topk_indices[:, -response_length - 1 : -1, :] = topk_indices
                            else:
                                response_positions = self._build_response_positions(
                                    response_start_idx=response_start_idx.to(device=topk_indices.device),
                                    response_length=response_length,
                                    seqlen=seqlen,
                                )
                                batch_indices = torch.arange(batch_size, device=topk_indices.device).unsqueeze(1)
                                full_topk_indices[batch_indices, response_positions, :] = topk_indices
                            topk_indices_rmpad = index_first_axis(
                                rearrange(full_topk_indices, "b s k -> (b s) k"), indices
                            )
                            if self.use_ulysses_sp:
                                topk_indices_rmpad = slice_input_tensor(
                                    topk_indices_rmpad.unsqueeze(0), dim=1, padding=True
                                ).squeeze(0)
                            topk_logits_rmpad = torch.gather(logits_rmpad, dim=-1, index=topk_indices_rmpad)
                        logsumexp_rmpad = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True)
                        topk_logps_rmpad = topk_logits_rmpad - logsumexp_rmpad

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if use_topk:
                        topk_logps_rmpad = gather_outputs_and_unpad(
                            topk_logps_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        if return_topk_indices:
                            topk_indices_rmpad = gather_outputs_and_unpad(
                                topk_indices_rmpad,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]
                    if compute_all_logps:
                        all_logps_rmpad = all_logps_rmpad[:0]
                    if use_topk:
                        topk_logps_rmpad = topk_logps_rmpad[:0]
                        if return_topk_indices:
                            topk_indices_rmpad = topk_indices_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if compute_all_logps:
                    full_all_logps = pad_input(
                        hidden_states=all_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if use_topk:
                    full_topk_logps = pad_input(
                        hidden_states=topk_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    if return_topk_indices:
                        full_topk_indices = pad_input(
                            hidden_states=topk_indices_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = self._select_response_positions(
                        full_entropy.squeeze(-1),
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = self._select_response_positions(
                        full_sum_pi_squared.squeeze(-1),
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                log_probs = self._select_response_positions(
                    full_log_probs.squeeze(-1),
                    response_length=response_length,
                    response_start_idx=response_start_idx,
                )
                if compute_all_logps:
                    all_logps = self._select_response_positions(
                        full_all_logps,
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                if use_topk:
                    topk_logps = self._select_response_positions(
                        full_topk_logps,
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                    if return_topk_indices:
                        topk_indices = self._select_response_positions(
                            full_topk_indices,
                            response_length=response_length,
                            response_start_idx=response_start_idx,
                        )

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                if visual_token_selection_session is not None:
                    extra_args["visual_token_selection_session"] = visual_token_selection_session

                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating
                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = self._select_response_positions(
                        logits,
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if compute_all_logps:
                        all_logps = torch.log_softmax(logits, dim=-1)
                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits.size(-1))
                            topk_logits, topk_indices = torch.topk(logits, topk, dim=-1)
                        else:
                            topk_logits = torch.gather(logits, dim=-1, index=topk_indices)
                        logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
                        topk_logps = topk_logits - logsumexp
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            if compute_all_logps:
                outputs["all_logps"] = all_logps
            if use_topk:
                outputs["topk_logps"] = topk_logps
                if return_topk_indices:
                    outputs["topk_indices"] = topk_indices
            return outputs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
            return grad_norm

        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self._current_global_steps = data.meta_info.get("global_steps")
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

        self_distillation_enabled = loss_mode == "vopd"
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        contrastive_visual_advantage_enabled = False
        teacher_entropy_reweighting_enabled = False
        if self_distillation_enabled:
            if self_distillation_cfg is None:
                raise ValueError(f"loss_mode={loss_mode} requires actor.self_distillation config.")
            contrastive_visual_advantage_enabled = bool(
                self_distillation_cfg.get("contrastive_visual_advantage", False)
            )
            teacher_entropy_reweighting_enabled = bool(
                self_distillation_cfg.get("teacher_entropy_reweighting", False)
            )
            self_distillation_required_keys = {
                "teacher_input_ids",
                "teacher_attention_mask",
                "teacher_position_ids",
                "teacher_response_start_idx",
                "self_distillation_mask",
            }
            assert self_distillation_required_keys.issubset(set(data.batch.keys())), (
                f"Missing required keys: {self_distillation_required_keys - set(data.batch.keys())}"
            )

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
        ]
        if not self_distillation_enabled or "advantages" in data.batch.keys():
            select_keys.append("advantages")
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self_distillation_enabled:
            select_keys.extend(list(self_distillation_required_keys))
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )
        has_teacher_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("teacher_multi_modal_inputs")
        )
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if has_teacher_multi_modal_inputs:
            non_tensor_select_keys.append("teacher_multi_modal_inputs")
        if "teacher_visual_token_selection_metadata" in data.non_tensor_batch:
            non_tensor_select_keys.append("teacher_visual_token_selection_metadata")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        if self_distillation_enabled:
            metrics["actor/grpo_loss"] = 0.0
            metrics["actor/vopd_loss"] = 0.0
            metrics["actor/vopd_loss_weighted"] = 0.0
        distill_dump_chunks = []
        stage_wall_time_totals = None
        if self_distillation_enabled:
            stage_wall_time_totals = {
                "timing_s/update_actor/student_forward": 0.0,
                "timing_s/update_actor/teacher_forward": 0.0,
                "timing_s/update_actor/loss_compute": 0.0,
                "timing_s/update_actor/backward": 0.0,
                "timing_s/update_actor/optimizer_step": 0.0,
                "timing_s/update_actor/teacher_ema_update": 0.0,
            }
            if contrastive_visual_advantage_enabled:
                stage_wall_time_totals["timing_s/update_actor/negative_teacher_forward"] = 0.0
        did_update = False
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                entropy_weight_accumulator = None
                token_weight_metrics_accumulator = None
                entropy_loss_metric_starts = None
                if teacher_entropy_reweighting_enabled or contrastive_visual_advantage_enabled:
                    mini_distillation_mask = mini_batch.batch["self_distillation_mask"]
                    if teacher_entropy_reweighting_enabled and not torch.all(
                        mini_distillation_mask > 0.5
                    ).item():
                        raise ValueError(
                            "teacher_entropy_reweighting requires every optimizer mini-batch sample "
                            "to be routed to SDPO; GRPO fallback is not supported"
                        )
                    mini_loss_mask = mini_batch.batch["response_mask"] * mini_distillation_mask.unsqueeze(1)
                    global_token_count = global_valid_token_count(
                        mini_loss_mask,
                        reduction_device=get_device_id(),
                    )
                    token_weight_mode = (
                        "teacher_entropy" if teacher_entropy_reweighting_enabled else "visual_advantage"
                    )
                    sample_seed = int(self._current_global_steps or 0) * 100003 + batch_idx
                    token_weight_metrics_accumulator = DistillationTokenWeightMetricsAccumulator(
                        mode=token_weight_mode,
                        global_token_count=global_token_count,
                        sample_seed=sample_seed,
                        vocab_size=self.teacher_vocab_size if teacher_entropy_reweighting_enabled else None,
                    )
                if teacher_entropy_reweighting_enabled:
                    entropy_weight_accumulator = TeacherEntropyWeightAccumulator(
                        beta=self.teacher_entropy_beta,
                        global_token_count=global_token_count,
                    )
                    entropy_loss_metric_starts = {
                        key: metrics[key]
                        for key in ("actor/pg_loss", "actor/vopd_loss", "actor/vopd_loss_weighted")
                    }
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs.get("advantages")

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)
                    self_distillation_mask = (
                        model_inputs.get("self_distillation_mask") if self_distillation_enabled else None
                    )
                    policy_fallback_mask = None
                    if self_distillation_enabled and self_distillation_mask is not None:
                        policy_fallback_mask = (self_distillation_mask <= 0.5).to(response_mask.dtype)
                        micro_batch_metrics["actor/policy_fallback_fraction"] = (
                            policy_fallback_mask.float().mean().detach().item()
                        )

                    if teacher_entropy_reweighting_enabled:
                        loss_scale_factor = 1.0
                    elif self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    teacher_regularization = self_distillation_cfg.get("teacher_regularization", "ema")
                    teacher_model_source = self_distillation_cfg.get("teacher_model_source", "legacy")
                    use_trust_region_teacher = (
                        teacher_model_source == "legacy" and teacher_regularization == "trust-region"
                    )
                    if use_trust_region_teacher and self.use_fused_kernels:
                        raise ValueError("trust-region teacher requires disabling fused kernels to access logits.")
                    # all return: (bsz, response_length)
                    return_all_logps = (
                        self_distillation_cfg.full_logit_distillation
                        and not self_distillation_cfg.distillation_topk
                    )
                    distill_topk = (
                        self_distillation_cfg.distillation_topk
                        if self_distillation_cfg.full_logit_distillation
                        else None
                    )
                    use_visual_token_strategy = bool(
                        self_distillation_enabled
                        and self_distillation_cfg.get("teacher_visual_token_mode", "crop")
                        == "same_image_bbox_mask"
                    )
                    visual_token_selection_session = None
                    if use_visual_token_strategy:
                        visual_token_selection_session = self._build_visual_token_selection_session(
                            model_inputs.get("teacher_visual_token_selection_metadata"),
                            model_inputs["attention_mask"],
                        )
                    student_forward_start = time.perf_counter()
                    outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        return_all_logps=return_all_logps,
                        distill_topk=distill_topk,
                        visual_token_selection_session=visual_token_selection_session,
                    )
                    if self_distillation_enabled:
                        student_forward_time = time.perf_counter() - student_forward_start
                        stage_wall_time_totals["timing_s/update_actor/student_forward"] += student_forward_time
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None
                    student_all_logps = outputs.get("all_logps") if return_all_logps else None
                    student_topk_logps = outputs.get("topk_logps") if distill_topk else None
                    student_topk_indices = outputs.get("topk_indices") if distill_topk else None
                    visual_token_selection_result = (
                        visual_token_selection_session.result
                        if visual_token_selection_session is not None
                        else None
                    )

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    if self_distillation_enabled:
                        teacher_inputs = {
                            "responses": model_inputs["responses"],
                            "input_ids": model_inputs["teacher_input_ids"],
                            "attention_mask": model_inputs["teacher_attention_mask"],
                            "position_ids": model_inputs["teacher_position_ids"],
                            "response_start_idx": model_inputs["teacher_response_start_idx"],
                        }
                        if use_visual_token_strategy:
                            if "multi_modal_inputs" not in model_inputs:
                                raise ValueError("same_image_bbox_mask requires student multi_modal_inputs")
                            teacher_inputs["multi_modal_inputs"] = model_inputs["multi_modal_inputs"]
                        elif "teacher_multi_modal_inputs" in model_inputs:
                            teacher_inputs["multi_modal_inputs"] = model_inputs["teacher_multi_modal_inputs"]
                        teacher_model = self.teacher_module or self.actor_module
                        if use_trust_region_teacher and (
                            self.teacher_module is None or self.teacher_module is self.actor_module
                        ):
                            raise ValueError(
                                "trust-region teacher requires a separate teacher_module in the actor worker."
                            )
                        if use_visual_token_strategy:
                            teacher_inputs, selection_metrics = self._finalize_teacher_visual_token_mask(
                                teacher_inputs,
                                visual_token_selection_result,
                            )
                            micro_batch_metrics.update(selection_metrics)
                        with torch.no_grad():
                            teacher_forward_start = time.perf_counter()
                            teacher_outputs = self._forward_micro_batch(
                                teacher_inputs,
                                temperature=temperature,
                                calculate_entropy=teacher_entropy_reweighting_enabled,
                                return_all_logps=return_all_logps,
                                distill_topk=distill_topk,
                                topk_indices=student_topk_indices,
                                module=teacher_model,
                            )
                            teacher_forward_time = time.perf_counter() - teacher_forward_start
                        stage_wall_time_totals["timing_s/update_actor/teacher_forward"] += teacher_forward_time
                        teacher_log_prob = teacher_outputs["log_probs"]
                        teacher_all_logps = teacher_outputs.get("all_logps") if return_all_logps else None
                        teacher_topk_logps = teacher_outputs.get("topk_logps") if distill_topk else None
                        distillation_token_weights = None
                        distillation_token_weight_mode = None
                        distillation_token_weight_source = None
                        negative_teacher_log_prob = None
                        if contrastive_visual_advantage_enabled:
                            negative_teacher_inputs = {
                                "responses": model_inputs["responses"],
                                "input_ids": model_inputs["teacher_input_ids"],
                                "attention_mask": model_inputs["teacher_attention_mask"],
                                "position_ids": model_inputs["teacher_position_ids"],
                                "response_start_idx": model_inputs["teacher_response_start_idx"],
                                "multi_modal_inputs": model_inputs["multi_modal_inputs"],
                            }
                            negative_teacher_inputs, negative_selection_metrics = (
                                self._finalize_teacher_visual_token_mask(
                                    negative_teacher_inputs,
                                    visual_token_selection_result,
                                    mask_kind="negative",
                                )
                            )
                            micro_batch_metrics.update(negative_selection_metrics)
                            with torch.no_grad():
                                negative_teacher_forward_start = time.perf_counter()
                                negative_teacher_outputs = self._forward_micro_batch(
                                    negative_teacher_inputs,
                                    temperature=temperature,
                                    calculate_entropy=False,
                                    return_all_logps=False,
                                    distill_topk=None,
                                    module=teacher_model,
                                )
                                negative_teacher_forward_time = (
                                    time.perf_counter() - negative_teacher_forward_start
                                )
                            stage_wall_time_totals[
                                "timing_s/update_actor/negative_teacher_forward"
                            ] += negative_teacher_forward_time
                            negative_teacher_log_prob = negative_teacher_outputs["log_probs"]
                            visual_advantage_margin = (teacher_log_prob - negative_teacher_log_prob).detach()
                            distillation_token_weights = torch.relu(visual_advantage_margin)
                            distillation_token_weight_mode = "visual_advantage"
                            distillation_token_weight_source = visual_advantage_margin
                        elif teacher_entropy_reweighting_enabled:
                            teacher_entropy = teacher_outputs.get("entropys")
                            if teacher_entropy is None:
                                raise ValueError("Positive teacher forward did not return entropy")
                            entropy_loss_mask = response_mask * self_distillation_mask.unsqueeze(1)
                            distillation_token_weights = entropy_weight_accumulator.add(
                                teacher_entropy,
                                entropy_loss_mask,
                            )
                            distillation_token_weight_mode = "teacher_entropy"
                            distillation_token_weight_source = teacher_entropy.detach()
                        if self_distillation_cfg.get("log_prob_dump_dir", None):
                            if distill_topk:
                                student_distill_log_probs = student_topk_logps
                                teacher_distill_log_probs = teacher_topk_logps
                                if self_distillation_cfg.distillation_add_tail:
                                    student_distill_log_probs = self._add_tail_bucket(student_distill_log_probs)
                                    teacher_distill_log_probs = self._add_tail_bucket(teacher_distill_log_probs)
                            else:
                                student_distill_log_probs = student_all_logps
                                teacher_distill_log_probs = teacher_all_logps

                            if student_distill_log_probs is None or teacher_distill_log_probs is None:
                                raise ValueError("Missing distillation log_probs for dump.")

                            loss_mask = response_mask
                            if self_distillation_mask is not None:
                                loss_mask = loss_mask * self_distillation_mask.unsqueeze(1)
                            valid_rows = loss_mask > 0
                            if valid_rows.any():
                                distill_dump_chunks.append(
                                    {
                                        "student_log_probs": student_distill_log_probs[valid_rows]
                                        .detach()
                                        .cpu()
                                        .to(torch.float32),
                                        "teacher_log_probs": teacher_distill_log_probs[valid_rows]
                                        .detach()
                                        .cpu()
                                        .to(torch.float32),
                                    }
                                )
                        loss_compute_start = time.perf_counter()
                        vopd_loss, vopd_metrics = compute_self_distillation_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_prob,
                            response_mask=response_mask,
                            self_distillation_config=self_distillation_cfg,
                            old_log_probs=old_log_prob,
                            student_all_log_probs=student_all_logps,
                            teacher_all_log_probs=teacher_all_logps,
                            student_topk_log_probs=student_topk_logps,
                            teacher_topk_log_probs=teacher_topk_logps,
                            self_distillation_mask=self_distillation_mask,
                            distillation_token_weights=distillation_token_weights,
                            distillation_token_weight_mode=distillation_token_weight_mode,
                            distillation_token_weight_source=distillation_token_weight_source,
                            distillation_token_weight_metrics_accumulator=token_weight_metrics_accumulator,
                            positive_teacher_log_probs=(
                                teacher_log_prob if contrastive_visual_advantage_enabled else None
                            ),
                            negative_teacher_log_probs=negative_teacher_log_prob,
                            loss_agg_mode=loss_agg_mode,
                            rollout_is_weights=rollout_is_weights,
                            dp_size=(
                                entropy_weight_accumulator.dp_size
                                if teacher_entropy_reweighting_enabled
                                else 1
                            ),
                            batch_num_tokens=(
                                entropy_weight_accumulator.global_token_count
                                if teacher_entropy_reweighting_enabled
                                else self.config.global_batch_info.get("batch_num_tokens")
                            ),
                            global_batch_size=self.config.global_batch_info.get("global_batch_size"),
                            loss_scale_factor=self.config.global_batch_info.get("loss_scale_factor"),
                        )
                        if teacher_entropy_reweighting_enabled:
                            entropy_weight_accumulator.add_weighted_loss_sum(
                                vopd_metrics.pop("_self_distillation/entropy_weighted_loss_sum")
                            )
                        loss_compute_time = time.perf_counter() - loss_compute_start
                        stage_wall_time_totals["timing_s/update_actor/loss_compute"] += loss_compute_time

                        vopd_metrics["self_distillation/empty_target_batch"] = self_distillation_mask.sum().item() == 0
                        micro_batch_metrics.update(vopd_metrics)

                        if policy_fallback_mask is not None and policy_fallback_mask.any().item():
                            if advantages is None:
                                raise ValueError(
                                    "Mixed SDPO/GRPO fallback requires advantages for samples without teacher images."
                                )
                            policy_loss_fn = get_policy_loss_fn("vanilla")
                            grpo_loss, grpo_metrics = policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask * policy_fallback_mask.unsqueeze(1),
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_is_weights=rollout_is_weights,
                            )
                            pg_loss = vopd_loss + grpo_loss
                            micro_batch_metrics.update(
                                {
                                    f"actor/policy_fallback/{key.split('/', 1)[1]}": value
                                    for key, value in grpo_metrics.items()
                                }
                            )
                        else:
                            grpo_loss = None
                            pg_loss = vopd_loss
                    else:
                        # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                        # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                        policy_loss_fn = get_policy_loss_fn(loss_mode)

                        # Compute policy loss (any function is expected to return 2 values)
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                        micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    backward_start = time.perf_counter()
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    if self_distillation_enabled:
                        backward_time = time.perf_counter() - backward_start
                        stage_wall_time_totals["timing_s/update_actor/backward"] += backward_time

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    if self_distillation_enabled:
                        metrics["actor/vopd_loss"] += vopd_loss.detach().item() * loss_scale_factor
                        metrics["actor/vopd_loss_weighted"] += vopd_loss.detach().item() * loss_scale_factor
                        if grpo_loss is not None:
                            metrics["actor/grpo_loss"] += grpo_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                if teacher_entropy_reweighting_enabled:
                    normalization_scale, entropy_weight_metrics = entropy_weight_accumulator.finalize()
                    scale_gradients_(self.actor_module.parameters(), normalization_scale)
                    for key, start_value in entropy_loss_metric_starts.items():
                        metrics[key] = start_value + (metrics[key] - start_value) * normalization_scale
                    append_to_dict(metrics, entropy_weight_metrics)
                else:
                    normalization_scale = 1.0
                if token_weight_metrics_accumulator is not None:
                    token_weight_metrics = token_weight_metrics_accumulator.finalize(
                        normalization_scale=normalization_scale
                    )
                    append_to_dict(metrics, token_weight_metrics)

                optimizer_step_start = time.perf_counter()
                grad_norm = self._optimizer_step()
                if self_distillation_enabled:
                    optimizer_step_time = time.perf_counter() - optimizer_step_start
                if torch.isfinite(grad_norm).item():
                    did_update = True
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                if self_distillation_enabled:
                    stage_wall_time_totals["timing_s/update_actor/optimizer_step"] += optimizer_step_time
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        if self_distillation_enabled and distill_dump_chunks:
            self._dump_self_distillation_log_probs(
                meta_info=data.meta_info,
                self_distillation_cfg=self_distillation_cfg,
                dump_chunks=distill_dump_chunks,
            )
        if did_update:
            teacher_update_start = time.perf_counter()
            self._update_teacher()
            if self_distillation_enabled:
                stage_wall_time_totals["timing_s/update_actor/teacher_ema_update"] += (
                    time.perf_counter() - teacher_update_start
                )
        if self_distillation_enabled:
            for key, total_time in stage_wall_time_totals.items():
                metrics[key] = Metric(aggregation=AggregationType.MAX, value=total_time)
        metric_keys_to_keep_unreduced = (
            set(stage_wall_time_totals.keys()) if stage_wall_time_totals is not None else set()
        )
        local_metrics_to_reduce = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, list) or (isinstance(value, Metric) and key not in metric_keys_to_keep_unreduced)
        }
        if local_metrics_to_reduce:
            reduced_local_metrics = reduce_metrics(local_metrics_to_reduce)
            metrics.update(reduced_local_metrics)
        return metrics
