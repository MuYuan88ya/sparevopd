from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class VisualTokenSelectionBatch:
    """Dense Student context available only while a selection session runs."""

    image_embeds: torch.Tensor
    inputs_embeds: torch.Tensor
    image_mask: torch.BoolTensor
    image_grid_thw: torch.Tensor
    valid_sequence_lengths: tuple[int, ...]
    metadata: tuple[Mapping[str, Any], ...]
    spatial_merge_size: int


@dataclass(frozen=True)
class VisualTokenSelectionSample:
    """One image and its aligned multimodal Student context."""

    image_embeds: torch.Tensor
    inputs_embeds: torch.Tensor
    image_mask: torch.BoolTensor
    image_grid_thw: torch.Tensor
    target_keep_mask: torch.BoolTensor
    sample_id: Any
    grid_height: int
    grid_width: int


@dataclass(frozen=True)
class VisualTokenSelectionResult:
    """Compact selection result retained between Student and Teacher forwards."""

    keep_masks: tuple[torch.BoolTensor, ...]
    target_keep_masks: tuple[torch.BoolTensor, ...]
    sample_ids: tuple[Any, ...]
    metrics: dict[str, float]
    negative_keep_masks: tuple[torch.BoolTensor, ...] | None = None

    def __post_init__(self) -> None:
        sample_count = len(self.keep_masks)
        if len(self.target_keep_masks) != sample_count or len(self.sample_ids) != sample_count:
            raise ValueError("Visual token selection result fields must have the same sample count")
        if self.negative_keep_masks is not None and len(self.negative_keep_masks) != sample_count:
            raise ValueError("Negative visual token masks must have the same sample count as positive masks")
        for sample_idx, (keep_mask, target_keep_mask) in enumerate(
            zip(self.keep_masks, self.target_keep_masks, strict=True)
        ):
            if not isinstance(keep_mask, torch.Tensor) or not isinstance(target_keep_mask, torch.Tensor):
                raise TypeError(f"selection masks[{sample_idx}] must be torch tensors")
            if keep_mask.ndim != 1 or keep_mask.dtype != torch.bool:
                raise ValueError(f"keep_masks[{sample_idx}] must be a rank-1 bool tensor")
            if target_keep_mask.ndim != 1 or target_keep_mask.dtype != torch.bool:
                raise ValueError(f"target_keep_masks[{sample_idx}] must be a rank-1 bool tensor")
            if keep_mask.shape != target_keep_mask.shape:
                raise ValueError(f"selection masks[{sample_idx}] must have the same shape")
            if keep_mask.requires_grad or target_keep_mask.requires_grad:
                raise ValueError("visual token selection masks must not require gradients")
            if torch.any(target_keep_mask.to(device=keep_mask.device) & ~keep_mask):
                raise ValueError(f"keep_masks[{sample_idx}] dropped a target token")
            if self.negative_keep_masks is None:
                continue

            negative_keep_mask = self.negative_keep_masks[sample_idx]
            if not isinstance(negative_keep_mask, torch.Tensor):
                raise TypeError(f"negative_keep_masks[{sample_idx}] must be a torch tensor")
            if negative_keep_mask.ndim != 1 or negative_keep_mask.dtype != torch.bool:
                raise ValueError(f"negative_keep_masks[{sample_idx}] must be a rank-1 bool tensor")
            if negative_keep_mask.shape != keep_mask.shape:
                raise ValueError(f"positive and negative masks[{sample_idx}] must have the same shape")
            if negative_keep_mask.requires_grad:
                raise ValueError("negative visual token selection masks must not require gradients")
            negative_on_keep_device = negative_keep_mask.to(device=keep_mask.device)
            if torch.any(keep_mask & negative_on_keep_device):
                raise ValueError(f"positive and negative masks[{sample_idx}] must not overlap")
            if int(keep_mask.sum().item()) != int(negative_keep_mask.sum().item()):
                raise ValueError(f"positive and negative masks[{sample_idx}] must keep the same token count")
            if torch.any(target_keep_mask.to(device=negative_keep_mask.device) & negative_keep_mask):
                raise ValueError(f"negative_keep_masks[{sample_idx}] retained a target token")

    def __len__(self) -> int:
        return len(self.keep_masks)


class VisualTokenSelectionStrategy(ABC):
    """Base class that owns batching, alignment, and output validation."""

    @abstractmethod
    def select(self, sample: VisualTokenSelectionSample) -> torch.Tensor:
        """Return the visual tokens that remain visible to the Teacher."""

    def selection_metrics(
        self,
        sample: VisualTokenSelectionSample,
        keep_mask: torch.BoolTensor,
    ) -> Mapping[str, float]:
        return {}

    @torch.no_grad()
    def select_batch(self, batch: VisualTokenSelectionBatch) -> VisualTokenSelectionResult:
        samples = _split_selection_batch(batch)
        keep_masks: list[torch.BoolTensor] = []
        target_keep_masks: list[torch.BoolTensor] = []
        sample_ids: list[Any] = []
        metrics: dict[str, float] = {}

        for sample_idx, sample in enumerate(samples):
            keep_mask = self.select(sample)
            if not isinstance(keep_mask, torch.Tensor):
                raise TypeError(
                    f"Visual token selection strategy returned {type(keep_mask)}, expected torch.Tensor"
                )
            if keep_mask.ndim != 1 or keep_mask.dtype != torch.bool:
                raise ValueError("strategy keep_mask must be a rank-1 bool tensor")
            if keep_mask.shape != sample.target_keep_mask.shape:
                raise ValueError(
                    "strategy keep_mask must match target_keep_mask, "
                    f"got {tuple(keep_mask.shape)} and {tuple(sample.target_keep_mask.shape)}"
                )
            target_on_keep_device = sample.target_keep_mask.to(device=keep_mask.device)
            if torch.any(target_on_keep_device & ~keep_mask):
                raise ValueError("visual token selection strategy must retain every target token")

            compact_keep_mask = keep_mask.detach().to(dtype=torch.bool).clone()
            compact_target_mask = sample.target_keep_mask.detach().to(device="cpu", dtype=torch.bool).clone()
            keep_masks.append(compact_keep_mask)
            target_keep_masks.append(compact_target_mask)
            sample_ids.append(sample.sample_id)
            for key, value in self.selection_metrics(sample, compact_keep_mask).items():
                metrics[key] = metrics.get(key, 0.0) + float(value)

        return VisualTokenSelectionResult(
            keep_masks=tuple(keep_masks),
            target_keep_masks=tuple(target_keep_masks),
            sample_ids=tuple(sample_ids),
            metrics=metrics,
        )


class VisualTokenSelectionSession:
    """Single-use bridge between one Student forward and its Teacher forward."""

    def __init__(
        self,
        *,
        strategy: VisualTokenSelectionStrategy,
        metadata: Sequence[Mapping[str, Any]],
        valid_sequence_lengths: Sequence[int],
        spatial_merge_size: int,
        contrastive_negative_mode: str | None = None,
        contrastive_negative_seed: int = 0,
        global_step: int | None = None,
    ) -> None:
        if not isinstance(strategy, VisualTokenSelectionStrategy):
            raise TypeError(f"strategy must be a VisualTokenSelectionStrategy, got {type(strategy)}")
        self._strategy = strategy
        self._metadata = tuple(dict(item) for item in metadata)
        self._valid_sequence_lengths = tuple(int(length) for length in valid_sequence_lengths)
        self._spatial_merge_size = int(spatial_merge_size)
        self._contrastive_negative_mode = contrastive_negative_mode
        self._contrastive_negative_seed = int(contrastive_negative_seed)
        self._global_step = None if global_step is None else int(global_step)
        self._started = False
        self._result: VisualTokenSelectionResult | None = None

        if not self._metadata:
            raise ValueError("visual token selection requires at least one sample")
        if len(self._metadata) != len(self._valid_sequence_lengths):
            raise ValueError(
                "selection metadata and valid sequence lengths must have the same size, "
                f"got metadata={len(self._metadata)} and lengths={len(self._valid_sequence_lengths)}"
            )
        if any(length <= 0 for length in self._valid_sequence_lengths):
            raise ValueError(f"valid sequence lengths must be positive, got {self._valid_sequence_lengths}")
        if self._spatial_merge_size <= 0:
            raise ValueError(f"spatial_merge_size must be positive, got {self._spatial_merge_size}")
        if self._contrastive_negative_mode not in (None, "complement-random-equal"):
            raise ValueError(
                "contrastive_negative_mode must be 'complement-random-equal' when enabled, "
                f"got {self._contrastive_negative_mode!r}"
            )
        if self._contrastive_negative_mode is not None and self._global_step is None:
            raise ValueError("Contrastive visual token selection requires a global_step")

    @torch.no_grad()
    def execute(
        self,
        *,
        image_embeds: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_mask: torch.BoolTensor,
        image_grid_thw: torch.Tensor,
    ) -> None:
        if self._started:
            raise RuntimeError("Visual token selection session can only be executed once")
        self._started = True
        batch = VisualTokenSelectionBatch(
            image_embeds=image_embeds,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            image_grid_thw=image_grid_thw,
            valid_sequence_lengths=self._valid_sequence_lengths,
            metadata=self._metadata,
            spatial_merge_size=self._spatial_merge_size,
        )
        result = self._strategy.select_batch(batch)
        if self._contrastive_negative_mode is not None:
            result = _add_complement_random_equal_masks(
                result,
                global_step=self._global_step,
                seed=self._contrastive_negative_seed,
            )
        self._result = result

    @property
    def result(self) -> VisualTokenSelectionResult:
        if self._result is None:
            if self._started:
                raise RuntimeError("Visual token selection session did not complete successfully")
            raise RuntimeError("Visual token selection session has not been executed")
        return self._result


VISUAL_TOKEN_SELECTION_STRATEGY_REGISTRY: dict[str, type[VisualTokenSelectionStrategy]] = {}


def register_visual_token_selection_strategy(
    name: str,
) -> Callable[[type[VisualTokenSelectionStrategy]], type[VisualTokenSelectionStrategy]]:
    if not name:
        raise ValueError("Visual token selection strategy name must not be empty")

    def decorator(strategy_cls: type[VisualTokenSelectionStrategy]) -> type[VisualTokenSelectionStrategy]:
        if not issubclass(strategy_cls, VisualTokenSelectionStrategy):
            raise TypeError(f"Visual token selection strategy {strategy_cls} must inherit VisualTokenSelectionStrategy")
        existing = VISUAL_TOKEN_SELECTION_STRATEGY_REGISTRY.get(name)
        if existing is not None:
            raise ValueError(f"Visual token selection strategy {name!r} is already registered: {existing}")
        VISUAL_TOKEN_SELECTION_STRATEGY_REGISTRY[name] = strategy_cls
        return strategy_cls

    return decorator


def create_visual_token_selection_strategy(
    name: str,
    kwargs: Mapping[str, Any] | None = None,
) -> VisualTokenSelectionStrategy:
    strategy_cls = VISUAL_TOKEN_SELECTION_STRATEGY_REGISTRY.get(name)
    if strategy_cls is None:
        available = sorted(VISUAL_TOKEN_SELECTION_STRATEGY_REGISTRY)
        raise ValueError(f"Unknown visual token selection strategy {name!r}; available strategies: {available}")
    try:
        return strategy_cls(**dict(kwargs or {}))
    except TypeError as exc:
        raise ValueError(f"Invalid parameters for visual token selection strategy {name!r}: {exc}") from exc


def build_bbox_target_keep_mask(
    *,
    bbox: list[float] | tuple[float, float, float, float] | torch.Tensor,
    image_width: int,
    image_height: int,
    grid_height: int,
    grid_width: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Map a source-image bbox to a row-major merged visual-token grid."""
    if isinstance(bbox, torch.Tensor):
        bbox = bbox.detach().cpu().flatten().tolist()
    if len(bbox) != 4:
        raise ValueError(f"bbox must contain [x1, y1, x2, y2], got {bbox!r}")
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"image dimensions must be positive, got {(image_width, image_height)}")
    if grid_width <= 0 or grid_height <= 0:
        raise ValueError(f"visual token grid must be positive, got {(grid_width, grid_height)}")

    x1, y1, x2, y2 = (float(value) for value in bbox)
    if not (0.0 <= x1 < x2 <= image_width and 0.0 <= y1 < y2 <= image_height):
        raise ValueError(
            "bbox must be non-empty and contained in the source image, "
            f"got bbox={(x1, y1, x2, y2)} for image={(image_width, image_height)}"
        )

    col_start = min(grid_width - 1, math.floor(x1 * grid_width / image_width))
    col_end = min(grid_width, max(col_start + 1, math.ceil(x2 * grid_width / image_width)))
    row_start = min(grid_height - 1, math.floor(y1 * grid_height / image_height))
    row_end = min(grid_height, max(row_start + 1, math.ceil(y2 * grid_height / image_height)))

    target_keep_mask = torch.zeros((grid_height, grid_width), dtype=torch.bool, device=device)
    target_keep_mask[row_start:row_end, col_start:col_end] = True
    return target_keep_mask.flatten()


_INTEGER_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


def _split_selection_batch(batch: VisualTokenSelectionBatch) -> list[VisualTokenSelectionSample]:
    image_embeds = batch.image_embeds
    inputs_embeds = batch.inputs_embeds
    image_mask = batch.image_mask
    image_grid_thw = batch.image_grid_thw
    sample_count = len(batch.metadata)

    if batch.spatial_merge_size <= 0:
        raise ValueError(f"spatial_merge_size must be positive, got {batch.spatial_merge_size}")

    if image_embeds.ndim != 2 or not torch.is_floating_point(image_embeds) or image_embeds.shape[1] == 0:
        raise ValueError(
            "image_embeds must have shape [tokens, hidden] with floating-point values, "
            f"got {tuple(image_embeds.shape)} and {image_embeds.dtype}"
        )
    if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
        raise ValueError(
            "remove-padding visual selection requires inputs_embeds with shape [1, valid_tokens, hidden], "
            f"got {tuple(inputs_embeds.shape)}"
        )
    if image_mask.ndim != 2 or image_mask.shape != inputs_embeds.shape[:2] or image_mask.dtype != torch.bool:
        raise ValueError(
            "image_mask must be a rank-2 bool tensor aligned with inputs_embeds, "
            f"got {tuple(image_mask.shape)} and {image_mask.dtype}"
        )
    if inputs_embeds.shape[-1] != image_embeds.shape[-1]:
        raise ValueError(
            "image and multimodal embedding hidden dimensions must match, "
            f"got {image_embeds.shape[-1]} and {inputs_embeds.shape[-1]}"
        )
    if image_grid_thw.ndim != 2 or image_grid_thw.shape != (sample_count, 3):
        raise ValueError(
            "visual token selection requires exactly one image per sample, "
            f"got image_grid_thw={tuple(image_grid_thw.shape)} and samples={sample_count}"
        )
    if image_grid_thw.dtype not in _INTEGER_DTYPES:
        raise ValueError(f"image_grid_thw must use an integer dtype, got {image_grid_thw.dtype}")
    if len(batch.valid_sequence_lengths) != sample_count:
        raise ValueError("valid sequence lengths must contain one value per sample")
    if sum(batch.valid_sequence_lengths) != inputs_embeds.shape[1]:
        raise ValueError(
            "valid sequence lengths do not cover the remove-padding embedding, "
            f"got lengths={sum(batch.valid_sequence_lengths)} and tokens={inputs_embeds.shape[1]}"
        )

    token_counts: list[int] = []
    grid_shapes: list[tuple[int, int]] = []
    for grid in image_grid_thw.tolist():
        grid_t, patch_grid_height, patch_grid_width = (int(value) for value in grid)
        if grid_t != 1:
            raise ValueError(f"visual token selection supports still images only, got grid_t={grid_t}")
        if patch_grid_height <= 0 or patch_grid_width <= 0:
            raise ValueError(f"visual grid dimensions must be positive, got {grid}")
        if patch_grid_height % batch.spatial_merge_size or patch_grid_width % batch.spatial_merge_size:
            raise ValueError(
                f"image grid {(patch_grid_height, patch_grid_width)} is not divisible by "
                f"merge size {batch.spatial_merge_size}"
            )
        grid_height = patch_grid_height // batch.spatial_merge_size
        grid_width = patch_grid_width // batch.spatial_merge_size
        grid_shapes.append((grid_height, grid_width))
        token_counts.append(grid_height * grid_width)

    if sum(token_counts) != image_embeds.shape[0]:
        raise ValueError(
            "dense image embeddings and image_grid_thw do not match, "
            f"got embeddings={image_embeds.shape[0]} and expected tokens={sum(token_counts)}"
        )
    if int(image_mask.sum().item()) != image_embeds.shape[0]:
        raise ValueError(
            "image_mask and dense image embeddings do not match, "
            f"got placeholders={int(image_mask.sum().item())} and embeddings={image_embeds.shape[0]}"
        )

    samples: list[VisualTokenSelectionSample] = []
    sequence_offset = 0
    image_offset = 0
    for sample_idx, (metadata, sequence_length, token_count, grid_shape) in enumerate(
        zip(batch.metadata, batch.valid_sequence_lengths, token_counts, grid_shapes, strict=True)
    ):
        required_metadata = {"bbox", "image_width", "image_height", "sample_id"}
        missing_metadata = required_metadata - set(metadata)
        if missing_metadata:
            raise KeyError(f"selection metadata[{sample_idx}] is missing {sorted(missing_metadata)}")
        if metadata["sample_id"] is None:
            raise ValueError(f"selection metadata[{sample_idx}] sample_id must not be None")

        sample_inputs_embeds = inputs_embeds[0, sequence_offset : sequence_offset + sequence_length]
        sample_image_mask = image_mask[0, sequence_offset : sequence_offset + sequence_length]
        sample_image_embeds = image_embeds[image_offset : image_offset + token_count]
        if int(sample_image_mask.sum().item()) != token_count:
            raise ValueError(
                f"sample {sample_idx} image placeholders do not match image_grid_thw, "
                f"got placeholders={int(sample_image_mask.sum().item())} and expected={token_count}"
            )

        grid_height, grid_width = grid_shape
        target_keep_mask = build_bbox_target_keep_mask(
            bbox=metadata["bbox"],
            image_width=int(metadata["image_width"]),
            image_height=int(metadata["image_height"]),
            grid_height=grid_height,
            grid_width=grid_width,
            device=image_embeds.device,
        )
        samples.append(
            VisualTokenSelectionSample(
                image_embeds=sample_image_embeds,
                inputs_embeds=sample_inputs_embeds,
                image_mask=sample_image_mask,
                image_grid_thw=image_grid_thw[sample_idx],
                target_keep_mask=target_keep_mask,
                sample_id=metadata["sample_id"],
                grid_height=grid_height,
                grid_width=grid_width,
            )
        )
        sequence_offset += sequence_length
        image_offset += token_count

    return samples


_INT63_MASK = 0x7FFFFFFFFFFFFFFF


def _stable_sample_seed(sample_id: Any, seed: int) -> int:
    payload = f"{sample_id}|{seed}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little") & _INT63_MASK


def _stable_contrastive_seed(sample_id: Any, global_step: int, seed: int) -> int:
    payload = f"{sample_id}|{global_step}|{seed}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little") & _INT63_MASK


def _spread_morton_bits(values: torch.Tensor) -> torch.Tensor:
    values = values & 0xFFFF
    values = (values | (values << 8)) & 0x00FF00FF
    values = (values | (values << 4)) & 0x0F0F0F0F
    values = (values | (values << 2)) & 0x33333333
    return (values | (values << 1)) & 0x55555555


def _morton_codes(rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    return _spread_morton_bits(cols) | (_spread_morton_bits(rows) << 1)


def _stateless_stratum_hash(strata: torch.Tensor, sample_seed: int) -> torch.Tensor:
    values = (strata + sample_seed + 0x1E3779B97F4A7C15) & _INT63_MASK
    values = values ^ (values >> 30)
    values = (values * 0x3F58476D1CE4E5B9) & _INT63_MASK
    values = values ^ (values >> 27)
    values = (values * 0x14D049BB133111EB) & _INT63_MASK
    return values ^ (values >> 31)


def _add_complement_random_equal_masks(
    result: VisualTokenSelectionResult,
    *,
    global_step: int | None,
    seed: int,
) -> VisualTokenSelectionResult:
    if global_step is None:
        raise ValueError("Contrastive visual token selection requires a global_step")

    negative_keep_masks: list[torch.BoolTensor] = []
    for sample_idx, (positive_keep_mask, target_keep_mask, sample_id) in enumerate(
        zip(result.keep_masks, result.target_keep_masks, result.sample_ids, strict=True)
    ):
        positive_count = int(positive_keep_mask.sum().item())
        complement_indices = torch.nonzero(~positive_keep_mask, as_tuple=True)[0]
        complement_count = int(complement_indices.numel())
        if complement_count < positive_count:
            target_count = int(target_keep_mask.sum().item())
            raise ValueError(
                "Cannot construct an equal-size disjoint negative visual-token mask: "
                f"sample_id={sample_id!r}, total_tokens={positive_keep_mask.numel()}, "
                f"bbox_tokens={target_count}, positive_tokens={positive_count}, "
                f"complement_tokens={complement_count}. Reduce background_keep_ratio; "
                "contrastive sparse teachers require background_keep_ratio <= 0.30 and "
                "at least as many unselected tokens as selected tokens."
            )

        sample_seed = _stable_contrastive_seed(sample_id, global_step, seed)
        candidate_hashes = _stateless_stratum_hash(complement_indices, sample_seed)
        candidate_order = torch.argsort(candidate_hashes, stable=True)
        selected_indices = complement_indices[candidate_order[:positive_count]]
        negative_keep_mask = torch.zeros_like(positive_keep_mask)
        negative_keep_mask.scatter_(0, selected_indices, True)
        negative_keep_masks.append(negative_keep_mask.detach().clone())

    return VisualTokenSelectionResult(
        keep_masks=result.keep_masks,
        target_keep_masks=result.target_keep_masks,
        sample_ids=result.sample_ids,
        metrics=result.metrics,
        negative_keep_masks=tuple(negative_keep_masks),
    )


@register_visual_token_selection_strategy("fixed_stratified")
class FixedStratifiedVisualTokenSelectionStrategy(VisualTokenSelectionStrategy):
    def __init__(self, background_keep_ratio: float = 0.1, seed: int = 0) -> None:
        if not 0.0 <= background_keep_ratio <= 1.0:
            raise ValueError(f"background_keep_ratio must be in [0, 1], got {background_keep_ratio}")
        self.background_keep_ratio = float(background_keep_ratio)
        self.seed = int(seed)

    def select(self, sample: VisualTokenSelectionSample) -> torch.Tensor:
        if sample.grid_height > 0xFFFF or sample.grid_width > 0xFFFF:
            raise ValueError(
                "fixed_stratified supports merged visual grids up to 65535x65535, "
                f"got {(sample.grid_height, sample.grid_width)}"
            )
        target_keep_mask = sample.target_keep_mask
        if self.background_keep_ratio == 0.0:
            return target_keep_mask.clone()
        if self.background_keep_ratio == 1.0:
            return torch.ones_like(target_keep_mask)

        background_indices = torch.nonzero(~target_keep_mask, as_tuple=True)[0]
        background_token_count = background_indices.numel()
        kept_background_token_count = math.ceil(background_token_count * self.background_keep_ratio)
        keep_mask = target_keep_mask.clone()
        if kept_background_token_count == 0:
            return keep_mask

        rows = torch.div(background_indices, sample.grid_width, rounding_mode="floor")
        cols = torch.remainder(background_indices, sample.grid_width)
        morton_codes = _morton_codes(rows, cols)
        ordered_background = background_indices[torch.argsort(morton_codes)]
        strata = torch.arange(kept_background_token_count, device=target_keep_mask.device, dtype=torch.int64)
        starts = torch.div(strata * background_token_count, kept_background_token_count, rounding_mode="floor")
        ends = torch.div(
            (strata + 1) * background_token_count,
            kept_background_token_count,
            rounding_mode="floor",
        )
        offsets = torch.remainder(
            _stateless_stratum_hash(strata, _stable_sample_seed(sample.sample_id, self.seed)),
            ends - starts,
        )
        keep_mask.scatter_(0, ordered_background[starts + offsets], True)
        return keep_mask


def apply_image_token_keep_mask(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_token_id: int,
    image_token_keep_mask: torch.Tensor,
) -> torch.Tensor:
    """Return a cloned attention mask with discarded image placeholders disabled."""
    if input_ids.ndim != 1 or attention_mask.ndim != 1:
        raise ValueError("input_ids and attention_mask must be rank-1 tensors for one sample")
    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            f"input_ids and attention_mask must have the same shape, got {input_ids.shape} and {attention_mask.shape}"
        )

    active_image_positions = (input_ids == image_token_id) & attention_mask.bool()
    image_token_count = int(active_image_positions.sum().item())
    if image_token_keep_mask.ndim != 1 or image_token_keep_mask.numel() != image_token_count:
        raise ValueError(
            "image token keep mask must match active image placeholders, "
            f"got mask={tuple(image_token_keep_mask.shape)} and placeholders={image_token_count}"
        )

    masked_attention = attention_mask.clone()
    masked_attention[active_image_positions] = image_token_keep_mask.to(
        device=masked_attention.device,
        dtype=masked_attention.dtype,
    )
    return masked_attention
