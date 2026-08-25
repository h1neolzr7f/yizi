from __future__ import annotations

import cv2
import numpy as np


def estimate_paper_bgr(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    band = max(4, min(h, w) // 40)
    border = np.concatenate(
        [
            bgr[:band, :].reshape(-1, 3),
            bgr[-band:, :].reshape(-1, 3),
            bgr[:, :band].reshape(-1, 3),
            bgr[:, -band:].reshape(-1, 3),
        ]
    )
    return np.median(border, axis=0)


def _overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    return inter / float(max(1, min(aw * ah, bw * bh)))


def _box_of(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def _dedupe_masks(masks: list[np.ndarray], overlap: float = 0.55) -> list[np.ndarray]:
    scored: list[tuple[int, np.ndarray, tuple[int, int, int, int]]] = []
    for mask in masks:
        box = _box_of(mask)
        if not box:
            continue
        scored.append((int(mask.sum()), mask, box))
    scored.sort(key=lambda item: item[0], reverse=True)
    kept: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    for _area, mask, box in scored:
        if any(_overlap_ratio(box, other[1]) > overlap for other in kept):
            continue
        kept.append((mask, box))
    return [item[0] for item in kept]


def _fill_masks(bgr: np.ndarray) -> list[np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    paper = estimate_paper_bgr(bgr).astype(np.uint8)
    swatch = np.zeros((1, 1, 3), np.uint8)
    swatch[0, 0] = paper
    paper_v = float(max(cv2.cvtColor(swatch, cv2.COLOR_BGR2HSV)[0, 0, 2], 1))
    paper_g = float(np.median(paper))
    v_lo = max(168, min(220, int(paper_v - 18)))
    white = cv2.inRange(hsv, (0, 0, v_lo), (180, 86, 255))
    local = (gray >= min(250, paper_g + 10)).astype(np.uint8) * 255
    paper_mask = cv2.bitwise_or(white, local)
    paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = bgr.shape[:2]
    area_img = h * w
    masks: list[np.ndarray] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_img * 0.0022 or area > area_img * 0.46:
            continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull) or 1
        solidity = area / hull_area
        peri = cv2.arcLength(cnt, True) or 1
        circ = 4 * np.pi * area / (peri * peri)
        if solidity < 0.52 and circ < 0.16:
            continue
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        if not _looks_like_balloon(bgr, gray, mask):
            continue
        masks.append(mask)
    return masks


def _looks_like_balloon(bgr: np.ndarray, gray: np.ndarray, mask: np.ndarray) -> bool:
    inner = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
    sample = inner if int(inner.sum()) > 80 else mask
    pix = bgr[sample > 0]
    if pix.size == 0:
        return False
    hsv = cv2.cvtColor(pix.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    v_mean = float(hsv[:, 2].mean())
    s_mean = float(hsv[:, 1].mean())
    g_std = float(gray[sample > 0].std())
    if v_mean < 162 or s_mean > 98:
        return False
    if g_std > 58:
        return False
    ring = cv2.subtract(cv2.dilate(mask, np.ones((5, 5), np.uint8)), cv2.erode(mask, np.ones((5, 5), np.uint8)))
    if int((ring > 0).sum()) > 20:
        ring_g = gray[ring > 0]
        if float(ring_g.mean()) > v_mean - 8 and float((ring_g < 90).mean()) < 0.04:
            if v_mean < 200:
                return False
    return True


def _stroke_masks(bgr: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 36, 110)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    h, w = bgr.shape[:2]
    area_img = h * w
    masks: list[np.ndarray] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_img * 0.002 or area > area_img * 0.38:
            continue
        peri = cv2.arcLength(cnt, True) or 1
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull) or 1
        solidity = area / hull_area
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 22 or bh < 22:
            continue
        if max(bw, bh) / max(1, min(bw, bh)) > 7.5:
            continue
        if solidity < 0.48 and len(approx) > 10:
            continue
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        fill = float((mask > 0).mean())
        if fill > 0.42:
            continue
        if not _looks_like_balloon(bgr, gray, mask):
            continue
        masks.append(mask)
    return masks


def _ink_block_masks(bgr: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    paper_g = float(np.median(estimate_paper_bgr(bgr)))
    ink = (gray < paper_g - 32).astype(np.uint8) * 255
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    horiz = cv2.dilate(ink, np.ones((3, 13), np.uint8))
    vert = cv2.dilate(ink, np.ones((13, 3), np.uint8))
    merged = cv2.bitwise_or(horiz, vert)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    h, w = gray.shape
    area_img = h * w
    masks: list[np.ndarray] = []
    for i in range(1, num):
        x, y, bw, bh, area = (int(stats[i, j]) for j in range(5))
        if area < area_img * 0.0016 or area > area_img * 0.12:
            continue
        if bw < 20 or bh < 18:
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 8:
            continue
        crop = gray[y : y + bh, x : x + bw]
        if crop.size == 0:
            continue
        density = float((crop < paper_g - 32).mean())
        if density < 0.035 or density > 0.62:
            continue
        if float(crop.mean()) < 70:
            continue
        pad = 6
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        mask = np.zeros((h, w), np.uint8)
        mask[y0:y1, x0:x1] = 255
        masks.append(mask)
    return masks[:16]


def detect_bubbles(bgr: np.ndarray) -> list[np.ndarray]:
    masks = _fill_masks(bgr) + _stroke_masks(bgr) + _ink_block_masks(bgr)
    return _dedupe_masks(masks)[:28]
