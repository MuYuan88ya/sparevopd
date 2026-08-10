from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict

import torch


@dataclass(frozen=True)
class VisualTokenSelectionInput:
    """Inputs supplied to a visual-token selection strategy.

    The current strategy path is batch-oriented: ``image_embeds`` and
    ``target_keep_mask`` are flattened in image order, while ``image_grid_thw``
    describes each image.  ``inputs_embeds`` and ``image_mask`` are the full
    model-call tensors (after visual features have been injected for the
    student).  The optional ``sample_id`` field keeps the old one-image helper
    API source compatible for downstream strategies and tests.
    """

    image_embeds: torch.Tensor
    image_grid_thw: torch.Tensor
    target_keep_mask: torch.Tensor
    sample_id: Any = None
    inputs_embeds: Optional[torch.Tensor] = None
    image_mask: Optional[torch.Tensor] = None
    sample_ids: Optional[tuple[Any, ...]] = None
    spatial_merge_size: Optional[int] = None


@dataclass(frozen=True)
class VisualTokenSelectionOutput:
    keep_mask: torch.Tensor
    # Kept as an optional compatibility field.  New strategies should return
    # only ``keep_mask``; metrics are now computed by the actor from masks.
    metrics: dict[str, float] = field(default_factory=dict)


class VisualTokenSelectionState(TypedDict):
    """Per-micro-batch visual-token selection state shared by student and teacher.

    The state deliberately contains masks and bookkeeping only.  Dense student
    visual embeddings are consumed while the student forward is still active and
    must never be retained here.
    """

    keep_masks: list[torch.BoolTensor]
    target_keep_masks: list[torch.BoolTensor]
    token_counts: list[int]
    sample_ids: list[Any]
    metrics: dict[str, float]


class VisualTokenSelectionStrategy(ABC):
    @abstractmethod
    def select(self, selection_input: VisualTokenSelectionInput) -> VisualTokenSelectionOutput:
        """Select the merged visual tokens that remain visible to the teacher."""


@dataclass(frozen=True)
class VisualTokenSelectionRequest:
    """Per-forward request consumed by Qwen3.5 ``_get_input_embeds``.

    This object is deliberately ephemeral.  It is passed as a dedicated model
    kwarg for the student call and is never stored in a DataProto or in the
    multimodal sample dictionaries.
    """

    strategy: VisualTokenSelectionStrategy
    target_keep_mask: torch.BoolTensor
    sample_ids: tuple[Any, ...]
    spatial_merge_size: int


VISUAL_TOKEN_SELECTION_STRATEGY_REGISTRY: dict[str, type[VisualTokenSelectionStrategy]] = {}


def register_visual_token_selection_strategy(
    name: str,
) -> Callable[[type[VisualTokenSelectionStrategy]], type[VisualTokenSelectionStrategy]]:
    if not name:
        raise ValueError("Visual token selection strategy name must not be empty")

    def decorator(
        strategy_cls: type[VisualTokenSelectionStrategy],
    ) -> type[VisualTokenSelectionStrategy]:
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


_INT63_MASK = 0x7FFFFFFFFFFFFFFF


def _stable_sample_seed(sample_id: Any, seed: int) -> int:
    payload = f"{sample_id}|{seed}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little") & _INT63_MASK


def _spread_morton_bits(values: torch.Tensor) -> torch.Tensor:
    """Spread 16-bit coordinates into alternating bits using vectorized integer operations."""
    values = values & 0xFFFF
    values = (values | (values << 8)) & 0x00FF00FF
    values = (values | (values << 4)) & 0x0F0F0F0F
    values = (values | (values << 2)) & 0x33333333
    return (values | (values << 1)) & 0x55555555


def _morton_codes(rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    return _spread_morton_bits(cols) | (_spread_morton_bits(rows) << 1)


def _stateless_stratum_hash(strata: torch.Tensor, sample_seed: int) -> torch.Tensor:
    """Return deterministic non-negative int64 hashes on either CPU or GPU."""
    values = (strata + sample_seed + 0x1E3779B97F4A7C15) & _INT63_MASK
    values = values ^ (values >> 30)
    values = (values * 0x3F58476D1CE4E5B9) & _INT63_MASK
    values = values ^ (values >> 27)
    values = (values * 0x14D049BB133111EB) & _INT63_MASK
    return values ^ (values >> 31)


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


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _validate_selection_input(
    selection_input: VisualTokenSelectionInput,
) -> tuple[list[tuple[int, int]], list[int], int]:
    """Validate a one-image or whole-micro-batch selection request.

    ``target_keep_mask`` and ``image_embeds`` are always flattened in image
    order.  For the new batch interface the configured merge size determines
    each image's token count from ``image_grid_thw``; the legacy one-image
    helper infers it from the mask length.
    """

    image_embeds = selection_input.image_embeds
    image_grid_thw = selection_input.image_grid_thw
    target_keep_mask = selection_input.target_keep_mask

    if image_embeds.ndim != 2:
        raise ValueError(f"image_embeds must have shape [tokens, hidden], got {tuple(image_embeds.shape)}")
    if image_embeds.shape[1] == 0 or not torch.is_floating_point(image_embeds):
        raise ValueError("image_embeds must have a non-empty floating-point hidden dimension")
    if image_grid_thw.ndim == 1:
        if image_grid_thw.shape != (3,):
            raise ValueError(
                f"image_grid_thw must contain [t, h, w], got {tuple(image_grid_thw.shape)}"
            )
        image_grid_rows = image_grid_thw.unsqueeze(0)
    elif image_grid_thw.ndim == 2 and image_grid_thw.shape[1] == 3:
        image_grid_rows = image_grid_thw
    else:
        raise ValueError(
            "image_grid_thw must have shape [3] or [images, 3], "
            f"got {tuple(image_grid_thw.shape)}"
        )
    if image_grid_thw.dtype not in _INTEGER_DTYPES:
        raise ValueError(f"image_grid_thw must use an integer dtype, got {image_grid_thw.dtype}")
    if target_keep_mask.ndim != 1 or target_keep_mask.dtype != torch.bool:
        raise ValueError("target_keep_mask must be a rank-1 bool tensor")

    image_count = image_grid_rows.shape[0]
    if selection_input.sample_ids is not None:
        if len(selection_input.sample_ids) != image_count:
            raise ValueError(
                "sample_ids must contain one id per image, "
                f"got images={image_count} and sample_ids={len(selection_input.sample_ids)}"
            )
        if any(sample_id is None for sample_id in selection_input.sample_ids):
            raise ValueError("sample_ids must not contain None")
    elif image_count == 1:
        if selection_input.sample_id is None:
            raise ValueError("sample_id must not be None")
    else:
        raise ValueError("batch visual token selection requires sample_ids")

    configured_merge_size = selection_input.spatial_merge_size
    if configured_merge_size is not None:
        merge_size = int(configured_merge_size)
        if merge_size <= 0:
            raise ValueError(f"spatial_merge_size must be positive, got {configured_merge_size}")
    else:
        if image_count != 1:
            raise ValueError("batch visual token selection requires spatial_merge_size")
        grid_t, patch_grid_height, patch_grid_width = (int(value) for value in image_grid_rows[0].tolist())
        if grid_t != 1 or patch_grid_height <= 0 or patch_grid_width <= 0:
            raise ValueError(
                "visual token selection supports one static image, "
                f"got image_grid_thw={(grid_t, patch_grid_height, patch_grid_width)}"
            )
        merged_token_count = target_keep_mask.numel()
        if merged_token_count == 0:
            raise ValueError("target_keep_mask must not be empty")
        patch_token_count = patch_grid_height * patch_grid_width
        if patch_token_count % merged_token_count:
            raise ValueError(
                "image_grid_thw cannot be aligned with the merged-token mask, "
                f"got patch tokens={patch_token_count} and merged tokens={merged_token_count}"
            )
        merge_area = patch_token_count // merged_token_count
        merge_size = math.isqrt(merge_area)
        if merge_size**2 != merge_area:
            raise ValueError(
                "image_grid_thw and target mask imply an invalid spatial merge, "
                f"got grid={(grid_t, patch_grid_height, patch_grid_width)}, mask={merged_token_count}"
            )

    grid_shapes: list[tuple[int, int]] = []
    token_counts: list[int] = []
    for row in image_grid_rows.tolist():
        grid_t, patch_grid_height, patch_grid_width = (int(value) for value in row)
        if grid_t != 1 or patch_grid_height <= 0 or patch_grid_width <= 0:
            raise ValueError(
                "visual token selection supports one static image, "
                f"got image_grid_thw={(grid_t, patch_grid_height, patch_grid_width)}"
            )
        if patch_grid_height % merge_size or patch_grid_width % merge_size:
            raise ValueError(
                f"image grid {(patch_grid_height, patch_grid_width)} is not divisible by merge size {merge_size}"
            )
        grid_shapes.append((patch_grid_height // merge_size, patch_grid_width // merge_size))
        token_counts.append((patch_grid_height // merge_size) * (patch_grid_width // merge_size))

    expected_token_count = sum(token_counts)
    if target_keep_mask.numel() != expected_token_count:
        raise ValueError(
            "image_grid_thw and target_keep_mask do not match, "
            f"got mask={target_keep_mask.numel()} and expected tokens={expected_token_count}"
        )
    if image_embeds.shape[0] != expected_token_count:
        raise ValueError(
            "dense image embeddings and image_grid_thw do not match, "
            f"got embeddings={image_embeds.shape[0]} and expected tokens={expected_token_count}"
        )

    if selection_input.inputs_embeds is not None or selection_input.image_mask is not None:
        if selection_input.inputs_embeds is None or selection_input.image_mask is None:
            raise ValueError("inputs_embeds and image_mask must be supplied together")
        if selection_input.inputs_embeds.ndim != 3:
            raise ValueError(
                "inputs_embeds must have shape [batch, sequence, hidden], "
                f"got {tuple(selection_input.inputs_embeds.shape)}"
            )
        if selection_input.image_mask.ndim != 2 or selection_input.image_mask.dtype != torch.bool:
            raise ValueError("image_mask must be a rank-2 bool tensor")
        if selection_input.inputs_embeds.shape[:2] != selection_input.image_mask.shape:
            raise ValueError(
                "inputs_embeds and image_mask sequence dimensions do not match, "
                f"got embeds={tuple(selection_input.inputs_embeds.shape[:2])} "
                f"and mask={tuple(selection_input.image_mask.shape)}"
            )
        if int(selection_input.image_mask.sum().item()) != expected_token_count:
            raise ValueError(
                "image_mask image-token count does not match image_grid_thw, "
                f"got mask={int(selection_input.image_mask.sum().item())} and expected={expected_token_count}"
            )

    return grid_shapes, token_counts, merge_size


def select_visual_tokens(
    strategy: VisualTokenSelectionStrategy,
    selection_input: VisualTokenSelectionInput,
) -> VisualTokenSelectionOutput:
    _validate_selection_input(selection_input)
    normalized_input = VisualTokenSelectionInput(
        image_embeds=selection_input.image_embeds,
        image_grid_thw=selection_input.image_grid_thw,
        target_keep_mask=selection_input.target_keep_mask.to(device=selection_input.image_embeds.device),
        sample_id=selection_input.sample_id,
        inputs_embeds=selection_input.inputs_embeds,
        image_mask=selection_input.image_mask,
        sample_ids=selection_input.sample_ids,
        spatial_merge_size=selection_input.spatial_merge_size,
    )
    output = strategy.select(normalized_input)
    if not isinstance(output, VisualTokenSelectionOutput):
        raise TypeError(f"Visual token selection strategy returned {type(output)}, expected VisualTokenSelectionOutput")
    keep_mask = output.keep_mask
    if keep_mask.ndim != 1 or keep_mask.dtype != torch.bool:
        raise ValueError("strategy keep_mask must be a rank-1 bool tensor")
    if keep_mask.shape != normalized_input.target_keep_mask.shape:
        raise ValueError(
            "strategy keep_mask must match target_keep_mask, "
            f"got {tuple(keep_mask.shape)} and {tuple(normalized_input.target_keep_mask.shape)}"
        )
    target_keep_mask = normalized_input.target_keep_mask.to(device=keep_mask.device)
    if torch.any(target_keep_mask & ~keep_mask):
        raise ValueError("visual token selection strategy must retain every target token")
    return VisualTokenSelectionOutput(keep_mask=keep_mask.detach().clone(), metrics=dict(output.metrics))


def select_visual_tokens_from_request(
    *,
    request: VisualTokenSelectionRequest,
    image_embeds: torch.Tensor,
    inputs_embeds: torch.Tensor,
    image_mask: torch.BoolTensor,
    image_grid_thw: torch.Tensor,
) -> VisualTokenSelectionOutput:
    """Run a registered strategy on the complete model-call context."""

    selection_input = VisualTokenSelectionInput(
        image_embeds=image_embeds,
        inputs_embeds=inputs_embeds,
        image_mask=image_mask,
        image_grid_thw=image_grid_thw,
        target_keep_mask=request.target_keep_mask,
        sample_ids=tuple(request.sample_ids),
        spatial_merge_size=request.spatial_merge_size,
    )
    # The strategy is discrete by contract.  Detached views ensure it cannot
    # retain the Student's autograd graph or create a second embedding copy.
    with torch.no_grad():
        return select_visual_tokens(request.strategy, selection_input)


def select_batched_visual_tokens(
    *,
    strategy: VisualTokenSelectionStrategy,
    image_embeds: torch.Tensor,
    image_grid_thw: torch.Tensor,
    target_keep_masks: list[torch.Tensor],
    sample_ids: list[Any],
) -> list[VisualTokenSelectionOutput]:
    if image_embeds.ndim != 2:
        raise ValueError(f"image_embeds must have shape [tokens, hidden], got {tuple(image_embeds.shape)}")
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
        raise ValueError(f"image_grid_thw must have shape [images, 3], got {tuple(image_grid_thw.shape)}")
    image_count = image_grid_thw.shape[0]
    if len(target_keep_masks) != image_count or len(sample_ids) != image_count:
        raise ValueError(
            "selection metadata must contain one target mask and sample id per image, "
            f"got grids={image_count}, targets={len(target_keep_masks)}, sample_ids={len(sample_ids)}"
        )

    token_counts = [int(mask.numel()) for mask in target_keep_masks]
    if sum(token_counts) != image_embeds.shape[0]:
        raise ValueError(
            "flattened image embeddings do not match per-image target masks, "
            f"got embeddings={image_embeds.shape[0]} and masks={sum(token_counts)}"
        )

    outputs = []
    offset = 0
    for image_idx, token_count in enumerate(token_counts):
        selection_input = VisualTokenSelectionInput(
            image_embeds=image_embeds[offset : offset + token_count],
            image_grid_thw=image_grid_thw[image_idx],
            target_keep_mask=target_keep_masks[image_idx],
            sample_id=sample_ids[image_idx],
        )
        outputs.append(select_visual_tokens(strategy, selection_input))
        offset += token_count
    return outputs


def build_visual_token_selection_state(
    *,
    strategy: VisualTokenSelectionStrategy,
    image_embeds: torch.Tensor,
    image_grid_thw: torch.Tensor,
    target_keep_masks: list[torch.Tensor],
    sample_ids: list[Any],
) -> VisualTokenSelectionState:
    """Select tokens once and materialize the compact state consumed by teacher.

    ``image_embeds`` is intentionally not present in the returned object.  The
    caller can therefore delete the dense embedding immediately after this
    function returns without leaving a hidden reference in the teacher path.
    """

    selection_outputs = select_batched_visual_tokens(
        strategy=strategy,
        image_embeds=image_embeds,
        image_grid_thw=image_grid_thw,
        target_keep_masks=target_keep_masks,
        sample_ids=sample_ids,
    )
    if len(selection_outputs) != len(target_keep_masks):
        raise RuntimeError(
            "visual token selection returned an unexpected number of samples, "
            f"got outputs={len(selection_outputs)} and targets={len(target_keep_masks)}"
        )

    keep_masks: list[torch.Tensor] = []
    normalized_target_masks: list[torch.Tensor] = []
    token_counts: list[int] = []
    metrics: dict[str, float] = {}
    total_token_count = 0
    for sample_idx, (target_keep_mask, selection_output) in enumerate(
        zip(target_keep_masks, selection_outputs, strict=True)
    ):
        raw_target_keep_mask = target_keep_mask
        raw_keep_mask = selection_output.keep_mask
        if raw_target_keep_mask.dtype != torch.bool:
            raise ValueError(f"target_keep_masks[{sample_idx}] must be a bool tensor")
        if raw_keep_mask.dtype != torch.bool:
            raise ValueError(f"selection keep_masks[{sample_idx}] must be a bool tensor")
        if raw_target_keep_mask.ndim != 1 or raw_keep_mask.ndim != 1:
            raise ValueError(f"selection masks[{sample_idx}] must be rank-1 tensors")
        if raw_keep_mask.requires_grad or raw_target_keep_mask.requires_grad:
            raise ValueError("visual token selection masks must not require gradients")
        target_keep_mask = raw_target_keep_mask.detach()
        keep_mask = raw_keep_mask.detach()
        if keep_mask.numel() != target_keep_mask.numel():
            raise ValueError(
                f"selection masks[{sample_idx}] have mismatched lengths: "
                f"keep={keep_mask.numel()} and target={target_keep_mask.numel()}"
            )
        target_on_keep_device = target_keep_mask.to(device=keep_mask.device)
        if torch.any(target_on_keep_device & ~keep_mask):
            raise ValueError(f"selection keep_masks[{sample_idx}] dropped a target token")
        # Keep the masks detached and independent of strategy temporaries.  The
        # target mask is small and kept on CPU to avoid retaining GPU storage.
        keep_mask = keep_mask.to(dtype=torch.bool).clone()
        target_keep_mask = target_keep_mask.to(device="cpu", dtype=torch.bool).clone()
        keep_masks.append(keep_mask)
        normalized_target_masks.append(target_keep_mask)
        token_count = int(keep_mask.numel())
        token_counts.append(token_count)
        total_token_count += token_count
        for key, value in selection_output.metrics.items():
            metrics[key] = metrics.get(key, 0.0) + float(value)

    if total_token_count != int(image_embeds.shape[0]):
        raise RuntimeError(
            "visual token selection state does not cover the dense embedding, "
            f"got state tokens={total_token_count} and embeddings={image_embeds.shape[0]}"
        )
    if len(sample_ids) != len(keep_masks):
        raise RuntimeError(
            "visual token selection state sample ids are misaligned, "
            f"got sample_ids={len(sample_ids)} and masks={len(keep_masks)}"
        )

    return VisualTokenSelectionState(
        keep_masks=keep_masks,
        target_keep_masks=normalized_target_masks,
        token_counts=token_counts,
        sample_ids=list(sample_ids),
        metrics=metrics,
    )


def build_visual_token_selection_state_from_keep_mask(
    *,
    keep_mask: torch.Tensor,
    image_grid_thw: torch.Tensor,
    target_keep_masks: list[torch.Tensor],
    sample_ids: list[Any],
    spatial_merge_size: int,
) -> VisualTokenSelectionState:
    """Materialize actor state from the mask returned by Qwen's Student call.

    Strategy execution has already happened inside ``_get_input_embeds``.  This
    helper only validates and splits the detached flat mask, so it never needs
    the dense visual embedding.
    """

    if keep_mask.ndim != 1 or keep_mask.dtype != torch.bool:
        raise ValueError("visual token keep_mask must be a rank-1 bool tensor")
    if keep_mask.requires_grad:
        raise ValueError("visual token keep_mask must not require gradients")
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
        raise ValueError(
            "image_grid_thw must have shape [images, 3], "
            f"got {tuple(image_grid_thw.shape)}"
        )
    if len(target_keep_masks) != image_grid_thw.shape[0] or len(sample_ids) != image_grid_thw.shape[0]:
        raise ValueError(
            "visual token selection metadata must contain one entry per image, "
            f"got grids={image_grid_thw.shape[0]}, targets={len(target_keep_masks)}, sample_ids={len(sample_ids)}"
        )
    if spatial_merge_size <= 0:
        raise ValueError(f"spatial_merge_size must be positive, got {spatial_merge_size}")

    expected_counts: list[int] = []
    for grid in image_grid_thw.tolist():
        grid_t, grid_height, grid_width = (int(value) for value in grid)
        if grid_t != 1 or grid_height % spatial_merge_size or grid_width % spatial_merge_size:
            raise ValueError(
                "visual token selection supports static grids divisible by spatial_merge_size, "
                f"got grid={grid} and merge={spatial_merge_size}"
            )
        expected_counts.append((grid_height // spatial_merge_size) * (grid_width // spatial_merge_size))

    if keep_mask.numel() != sum(expected_counts):
        raise ValueError(
            "student keep_mask does not cover image_grid_thw, "
            f"got mask={keep_mask.numel()} and expected={sum(expected_counts)}"
        )

    keep_masks: list[torch.Tensor] = []
    normalized_targets: list[torch.Tensor] = []
    token_counts: list[int] = []
    offset = 0
    for sample_idx, (expected_count, target_keep_mask) in enumerate(
        zip(expected_counts, target_keep_masks, strict=True)
    ):
        if target_keep_mask.ndim != 1 or target_keep_mask.dtype != torch.bool:
            raise ValueError(f"target_keep_masks[{sample_idx}] must be a rank-1 bool tensor")
        if target_keep_mask.numel() != expected_count:
            raise ValueError(
                f"target_keep_masks[{sample_idx}] has {target_keep_mask.numel()} tokens, "
                f"expected {expected_count}"
            )
        sample_keep_mask = keep_mask[offset : offset + expected_count].detach().clone()
        target_on_keep_device = target_keep_mask.to(device=sample_keep_mask.device)
        if torch.any(target_on_keep_device & ~sample_keep_mask):
            raise ValueError(f"student keep_mask dropped a target token for sample {sample_idx}")
        keep_masks.append(sample_keep_mask)
        normalized_targets.append(target_keep_mask.detach().to(device="cpu", dtype=torch.bool).clone())
        token_counts.append(expected_count)
        offset += expected_count

    return VisualTokenSelectionState(
        keep_masks=keep_masks,
        target_keep_masks=normalized_targets,
        token_counts=token_counts,
        sample_ids=list(sample_ids),
        metrics={},
    )


@register_visual_token_selection_strategy("fixed_stratified")
class FixedStratifiedVisualTokenSelectionStrategy(VisualTokenSelectionStrategy):
    def __init__(self, background_keep_ratio: float = 0.1, seed: int = 0) -> None:
        if not 0.0 <= background_keep_ratio <= 1.0:
            raise ValueError(f"background_keep_ratio must be in [0, 1], got {background_keep_ratio}")
        self.background_keep_ratio = float(background_keep_ratio)
        self.seed = int(seed)

    def _select_single(
        self,
        *,
        target_keep_mask: torch.Tensor,
        grid_height: int,
        grid_width: int,
        sample_id: Any,
    ) -> torch.Tensor:
        if grid_height > 0xFFFF or grid_width > 0xFFFF:
            raise ValueError(
                "fixed_stratified supports merged visual grids up to 65535x65535, "
                f"got {(grid_height, grid_width)}"
            )
        if self.background_keep_ratio == 0.0:
            return target_keep_mask.clone()
        if self.background_keep_ratio == 1.0:
            return torch.ones_like(target_keep_mask)

        background_indices = torch.nonzero(~target_keep_mask, as_tuple=True)[0]
        background_token_count = background_indices.numel()
        kept_background_token_count = (
            math.ceil(background_token_count * self.background_keep_ratio)
            if self.background_keep_ratio > 0.0
            else 0
        )

        keep_mask = target_keep_mask.clone()
        if kept_background_token_count:
            rows = torch.div(background_indices, grid_width, rounding_mode="floor")
            cols = torch.remainder(background_indices, grid_width)
            morton_codes = _morton_codes(rows, cols)
            background_order = torch.argsort(morton_codes)
            ordered_background = background_indices[background_order]
            del rows, cols, morton_codes, background_order, background_indices

            strata = torch.arange(
                kept_background_token_count,
                device=target_keep_mask.device,
                dtype=torch.int64,
            )
            starts = torch.div(
                strata * background_token_count,
                kept_background_token_count,
                rounding_mode="floor",
            )
            ends = torch.div(
                (strata + 1) * background_token_count,
                kept_background_token_count,
                rounding_mode="floor",
            )
            offsets = torch.remainder(
                _stateless_stratum_hash(
                    strata,
                    _stable_sample_seed(sample_id, self.seed),
                ),
                ends - starts,
            )
            keep_mask.scatter_(0, ordered_background[starts + offsets], True)
            del ordered_background, strata, starts, ends, offsets

        return keep_mask

    def select(self, selection_input: VisualTokenSelectionInput) -> VisualTokenSelectionOutput:
        grid_shapes, token_counts, _ = _validate_selection_input(selection_input)
        target_keep_mask = selection_input.target_keep_mask
        image_count = len(token_counts)
        if image_count == 1:
            sample_id = (
                selection_input.sample_ids[0]
                if selection_input.sample_ids is not None
                else selection_input.sample_id
            )
            keep_mask = self._select_single(
                target_keep_mask=target_keep_mask,
                grid_height=grid_shapes[0][0],
                grid_width=grid_shapes[0][1],
                sample_id=sample_id,
            )
            return VisualTokenSelectionOutput(keep_mask=keep_mask)

        if selection_input.sample_ids is None:
            raise ValueError("batch fixed_stratified selection requires sample_ids")
        keep_mask = target_keep_mask.clone()
        offset = 0
        for token_count, (grid_height, grid_width), sample_id in zip(
            token_counts, grid_shapes, selection_input.sample_ids, strict=True
        ):
            selected = self._select_single(
                target_keep_mask=target_keep_mask[offset : offset + token_count],
                grid_height=grid_height,
                grid_width=grid_width,
                sample_id=sample_id,
            )
            keep_mask[offset : offset + token_count] = selected
            offset += token_count
        return VisualTokenSelectionOutput(keep_mask=keep_mask)


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
