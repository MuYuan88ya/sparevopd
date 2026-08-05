from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class VisualTokenSelectionInput:
    image_embeds: torch.Tensor
    image_grid_thw: torch.Tensor
    target_keep_mask: torch.Tensor
    sample_id: Any


@dataclass(frozen=True)
class VisualTokenSelectionOutput:
    keep_mask: torch.Tensor
    metrics: dict[str, float] = field(default_factory=dict)


class VisualTokenSelectionStrategy(ABC):
    @abstractmethod
    def select(self, selection_input: VisualTokenSelectionInput) -> VisualTokenSelectionOutput:
        """Select the merged visual tokens that remain visible to the teacher."""


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


def _morton_code(row: int, col: int) -> int:
    """Interleave row and column bits to preserve two-dimensional locality."""
    code = 0
    bit = 0
    while row or col:
        code |= (col & 1) << (2 * bit)
        code |= (row & 1) << (2 * bit + 1)
        row >>= 1
        col >>= 1
        bit += 1
    return code


def _stable_index(sample_id: Any, seed: int, stratum: int, size: int) -> int:
    payload = f"{sample_id}|{seed}|{stratum}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little") % size


def build_bbox_target_keep_mask(
    *,
    bbox: list[float] | tuple[float, float, float, float] | torch.Tensor,
    image_width: int,
    image_height: int,
    grid_height: int,
    grid_width: int,
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

    target_keep_mask = torch.zeros(grid_height * grid_width, dtype=torch.bool)
    for row in range(row_start, row_end):
        start = row * grid_width + col_start
        target_keep_mask[start : start + (col_end - col_start)] = True
    return target_keep_mask


def _validate_selection_input(selection_input: VisualTokenSelectionInput) -> tuple[int, int]:
    image_embeds = selection_input.image_embeds
    image_grid_thw = selection_input.image_grid_thw
    target_keep_mask = selection_input.target_keep_mask

    if image_embeds.ndim != 2:
        raise ValueError(f"image_embeds must have shape [tokens, hidden], got {tuple(image_embeds.shape)}")
    if image_embeds.shape[1] == 0 or not torch.is_floating_point(image_embeds):
        raise ValueError("image_embeds must have a non-empty floating-point hidden dimension")
    if image_grid_thw.ndim == 2 and image_grid_thw.shape[0] == 1:
        image_grid_thw = image_grid_thw.squeeze(0)
    if image_grid_thw.shape != (3,):
        raise ValueError(f"image_grid_thw must contain one [t, h, w] row, got {tuple(image_grid_thw.shape)}")
    if image_grid_thw.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError(f"image_grid_thw must use an integer dtype, got {image_grid_thw.dtype}")
    if target_keep_mask.ndim != 1 or target_keep_mask.dtype != torch.bool:
        raise ValueError("target_keep_mask must be a rank-1 bool tensor")
    if selection_input.sample_id is None:
        raise ValueError("sample_id must not be None")
    if image_embeds.shape[0] != target_keep_mask.numel():
        raise ValueError(
            "dense image embeddings and target keep mask do not match, "
            f"got embeddings={image_embeds.shape[0]} and mask={target_keep_mask.numel()}"
        )

    grid_t, patch_grid_height, patch_grid_width = (int(value) for value in image_grid_thw.tolist())
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
    if merge_size**2 != merge_area or patch_grid_height % merge_size or patch_grid_width % merge_size:
        raise ValueError(
            "image_grid_thw and target mask imply an invalid spatial merge, "
            f"got grid={(grid_t, patch_grid_height, patch_grid_width)}, mask={merged_token_count}"
        )
    return patch_grid_height // merge_size, patch_grid_width // merge_size


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
    return VisualTokenSelectionOutput(keep_mask=keep_mask.clone(), metrics=dict(output.metrics))


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


@register_visual_token_selection_strategy("fixed_stratified")
class FixedStratifiedVisualTokenSelectionStrategy(VisualTokenSelectionStrategy):
    def __init__(self, background_keep_ratio: float = 0.1, seed: int = 0) -> None:
        if not 0.0 <= background_keep_ratio <= 1.0:
            raise ValueError(f"background_keep_ratio must be in [0, 1], got {background_keep_ratio}")
        self.background_keep_ratio = float(background_keep_ratio)
        self.seed = int(seed)

    def select(self, selection_input: VisualTokenSelectionInput) -> VisualTokenSelectionOutput:
        grid_height, grid_width = _validate_selection_input(selection_input)
        target_keep_mask = selection_input.target_keep_mask
        background_coords = [
            (row, col)
            for row in range(grid_height)
            for col in range(grid_width)
            if not target_keep_mask[row * grid_width + col]
        ]
        background_token_count = len(background_coords)
        kept_background_token_count = (
            math.ceil(background_token_count * self.background_keep_ratio)
            if self.background_keep_ratio > 0.0
            else 0
        )

        keep_mask = target_keep_mask.clone()
        if kept_background_token_count:
            ordered_background = sorted(background_coords, key=lambda coord: (_morton_code(*coord), *coord))
            for stratum in range(kept_background_token_count):
                start = stratum * background_token_count // kept_background_token_count
                end = (stratum + 1) * background_token_count // kept_background_token_count
                choices = ordered_background[start:end]
                choice = choices[
                    _stable_index(selection_input.sample_id, self.seed, stratum, len(choices))
                ]
                keep_mask[choice[0] * grid_width + choice[1]] = True

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
