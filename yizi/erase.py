from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from yizi.cv import estimate_paper_bgr
from yizi.models import TextRegion


def _decode_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("读不了这一页")
    return bgr


def _region_mask(
    shape: tuple[int, int],
    region: TextRegion,
    gray: np.ndarray | None = None,
    paper: float | None = None,
) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    pad = 4
    x0 = max(0, int(round(region.x)) - pad)
    y0 = max(0, int(round(region.y)) - pad)
    x1 = min(w, int(round(region.x + region.w)) + pad)
    y1 = min(h, int(round(region.y + region.h)) + pad)
    if x1 <= x0 or y1 <= y0:
        return mask
    inset_x = max(1, int((x1 - x0) * 0.03))
    inset_y = max(1, int((y1 - y0) * 0.03))
    cv2.ellipse(
        mask,
        ((x0 + x1) // 2, (y0 + y1) // 2),
        (max(4, (x1 - x0) // 2 - inset_x), max(4, (y1 - y0) // 2 - inset_y)),
        0,
        0,
        360,
        255,
        -1,
    )
    if region.kind == "narration" or region.direction == "vertical":
        mask[y0 + inset_y : y1 - inset_y, x0 + inset_x : x1 - inset_x] = 255
    if gray is not None and paper is not None:
        crop = gray[y0:y1, x0:x1]
        if crop.size:
            ink = (crop < (paper - 24)).astype(np.uint8) * 255
            ink = cv2.dilate(ink, np.ones((3, 3), np.uint8), iterations=1)
            near = cv2.dilate(mask[y0:y1, x0:x1], np.ones((7, 7), np.uint8), iterations=1)
            extra = cv2.bitwise_and(ink, near)
            mask[y0:y1, x0:x1] = cv2.bitwise_or(mask[y0:y1, x0:x1], extra)
    return mask


def _is_light_fill(bgr: np.ndarray, mask: np.ndarray) -> bool:
    pix = bgr[mask > 0]
    if pix.size == 0:
        return True
    return float(pix.mean()) > 168


def _local_fill(bgr: np.ndarray, mask: np.ndarray, paper: np.ndarray) -> np.ndarray:
    pix = bgr[mask > 0]
    if pix.size == 0:
        return paper
    bright = pix[pix.mean(axis=1) > 200]
    if len(bright) > 20:
        return np.median(bright, axis=0).astype(np.uint8)
    return paper


def erase_regions(image_bytes: bytes, regions: list[TextRegion]) -> bytes:
    bgr = _decode_bgr(image_bytes)
    h, w = bgr.shape[:2]
    paper = estimate_paper_bgr(bgr).astype(np.uint8)
    work = bgr.copy()
    changed = np.zeros((h, w), np.uint8)
    targets = [r for r in regions if not r.skip]
    if not targets:
        return _rgba_png(np.zeros((h, w, 4), np.uint8))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    paper_v = float(np.mean(paper))
    light_mask = np.zeros((h, w), np.uint8)
    heavy_mask = np.zeros((h, w), np.uint8)
    for region in targets:
        mask = _region_mask((h, w), region, gray, paper_v)
        if not mask.any():
            continue
        if _is_light_fill(bgr, mask):
            light_mask = cv2.bitwise_or(light_mask, mask)
        else:
            heavy_mask = cv2.bitwise_or(heavy_mask, mask)

    if light_mask.any():
        light_mask = cv2.dilate(light_mask, np.ones((3, 3), np.uint8), iterations=1)
        fill = _local_fill(bgr, light_mask, paper)
        work[light_mask > 0] = fill
        changed = cv2.bitwise_or(changed, light_mask)
    if heavy_mask.any():
        dilated = cv2.dilate(heavy_mask, np.ones((5, 5), np.uint8), iterations=1)
        inpainted = cv2.inpaint(work, dilated, 4, cv2.INPAINT_TELEA)
        work = np.where(dilated[..., None] > 0, inpainted, work)
        changed = cv2.bitwise_or(changed, dilated)

    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[:, :, :3] = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
    rgba[:, :, 3] = changed
    return _rgba_png(rgba)


def _rgba_png(rgba: np.ndarray) -> bytes:
    im = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()
