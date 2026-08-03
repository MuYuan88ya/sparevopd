from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SparseImageTokenSelection:
    keep_mask: torch.Tensor
    target_token_count: int
    background_token_count: int
    kept_background_token_count: int

    @property
    def kept_token_count(self) -> int:
        return self.target_token_count + self.kept_background_token_count


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


def build_bbox_sparse_keep_mask(
    *,
    bbox: list[float] | tuple[float, float, float, float] | torch.Tensor,
    image_width: int,
    image_height: int,
    grid_height: int,
    grid_width: int,
    background_keep_ratio: float,
    sample_id: Any,
    seed: int = 0,
) -> SparseImageTokenSelection:
    """Build a row-major keep mask over merged LLM image tokens."""
    if isinstance(bbox, torch.Tensor):
        bbox = bbox.detach().cpu().flatten().tolist()
    if len(bbox) != 4:
        raise ValueError(f"bbox must contain [x1, y1, x2, y2], got {bbox!r}")
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"image dimensions must be positive, got {(image_width, image_height)}")
    if grid_width <= 0 or grid_height <= 0:
        raise ValueError(f"visual token grid must be positive, got {(grid_width, grid_height)}")
    if not 0.0 <= background_keep_ratio <= 1.0:
        raise ValueError(f"background_keep_ratio must be in [0, 1], got {background_keep_ratio}")

    x1, y1, x2, y2 = (float(value) for value in bbox)
    if not (0.0 <= x1 < x2 <= image_width and 0.0 <= y1 < y2 <= image_height):
        raise ValueError(
            "bbox must be non-empty and contained in the source image, "
            f"got bbox={(x1, y1, x2, y2)} for image={(image_width, image_height)}"
        )

    col_start = min(grid_width - 1, math.floor(x1 * grid_width / image_width))
    col_end = max(col_start + 1, math.ceil(x2 * grid_width / image_width))
    row_start = min(grid_height - 1, math.floor(y1 * grid_height / image_height))
    row_end = max(row_start + 1, math.ceil(y2 * grid_height / image_height))
    col_end = min(grid_width, col_end)
    row_end = min(grid_height, row_end)

    keep_mask = torch.zeros(grid_height * grid_width, dtype=torch.bool)
    for row in range(row_start, row_end):
        start = row * grid_width + col_start
        keep_mask[start : start + (col_end - col_start)] = True

    target_token_count = int(keep_mask.sum().item())
    background_coords = [
        (row, col)
        for row in range(grid_height)
        for col in range(grid_width)
        if not keep_mask[row * grid_width + col]
    ]
    background_token_count = len(background_coords)
    kept_background_token_count = (
        math.ceil(background_token_count * background_keep_ratio) if background_keep_ratio > 0.0 else 0
    )

    if kept_background_token_count:
        ordered_background = sorted(background_coords, key=lambda coord: (_morton_code(*coord), *coord))
        for stratum in range(kept_background_token_count):
            start = stratum * background_token_count // kept_background_token_count
            end = (stratum + 1) * background_token_count // kept_background_token_count
            choices = ordered_background[start:end]
            choice = choices[_stable_index(sample_id, seed, stratum, len(choices))]
            keep_mask[choice[0] * grid_width + choice[1]] = True

    return SparseImageTokenSelection(
        keep_mask=keep_mask,
        target_token_count=target_token_count,
        background_token_count=background_token_count,
        kept_background_token_count=kept_background_token_count,
    )


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
