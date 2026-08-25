from __future__ import annotations

import uuid

import cv2
import numpy as np

from yizi.cv import detect_bubbles, estimate_paper_bgr
from yizi.models import TextRegion


def new_region_id() -> str:
    return f"r{uuid.uuid4().hex[:8]}"


def estimate_max_chars(w: float, h: float, direction: str) -> int:
    cell = 26.0
    cols = max(1, int(round(w / cell)))
    rows = max(1, int(round(h / cell)))
    return max(4, min(80, cols * rows))


def pick_direction(w: float, h: float, kind: str) -> str:
    if kind == "narration" and h > w * 1.25:
        return "vertical"
    if h > w * 1.15:
        return "vertical"
    return "horizontal"


def _box_from_mask(mask: np.ndarray, pad: int = 4) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    h, w = mask.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    if x1 - x0 < 18 or y1 - y0 < 18:
        return None
    return x0, y0, x1 - x0, y1 - y0


def _overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    return inter / float(max(1, min(aw * ah, bw * bh)))


def _ink_density(gray: np.ndarray, box: tuple[int, int, int, int], paper: float) -> float:
    x, y, w, h = box
    crop = gray[y : y + h, x : x + w]
    if crop.size == 0:
        return 0.0
    ink = crop < (paper - 28)
    return float(ink.mean())


def _is_banner(x: float, y: float, w: float, h: float, page_w: int, page_h: int) -> bool:
    if page_w <= 0 or page_h <= 0:
        return False
    return (
        y < page_h * 0.16
        and w > page_w * 0.42
        and h < page_h * 0.18
        and w > h * 1.7
    )


def _classify_kind(box: tuple[int, int, int, int], page_w: int, page_h: int, circ: float) -> str:
    x, y, w, h = box
    cx = (x + w / 2) / max(page_w, 1)
    tall = h > w * 1.6
    side = cx < 0.22 or cx > 0.78
    if tall and (side or w < page_w * 0.22):
        return "narration"
    if circ >= 0.45 and cx < 0.42 and (w * h) < page_w * page_h * 0.07 and not tall:
        return "thought"
    if circ < 0.28 and (w * h) < page_w * page_h * 0.04:
        return "sfx"
    return "dialogue"


def detect_vertical_strips(bgr: np.ndarray, occupied: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = bgr.shape[:2]
    white = cv2.inRange(hsv, (0, 0, 188), (180, 82, 255))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((11, 3), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[tuple[int, int, int, int]] = []
    area_img = h * w
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < area_img * 0.006 or area > area_img * 0.3:
            continue
        if bh < bw * 1.75:
            continue
        if bw < 16 or bw > w * 0.32:
            continue
        box = (x, y, bw, bh)
        if any(_overlap_ratio(box, other) > 0.45 for other in occupied):
            continue
        out.append(box)
    return out


def detect_sfx_blobs(bgr: np.ndarray, occupied: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    paper = float(np.mean(estimate_paper_bgr(bgr)))
    ink = (gray < paper - 40).astype(np.uint8) * 255
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    num, _labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    h, w = gray.shape
    area_img = h * w
    out: list[tuple[int, int, int, int]] = []
    for i in range(1, num):
        x, y, bw, bh, area = (int(stats[i, j]) for j in range(5))
        if area < area_img * 0.004 or area > area_img * 0.035:
            continue
        if bw < 20 or bh < 20:
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 4.5:
            continue
        box = (x, y, bw, bh)
        if any(_overlap_ratio(box, other) > 0.25 for other in occupied):
            continue
        hull = cv2.convexHull(np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]))
        solidity = area / max(1.0, float(cv2.contourArea(hull) or (bw * bh)))
        if solidity > 0.88 and area > area_img * 0.015:
            continue
        out.append(box)
    return out[:6]


def _gap_fill_boxes(
    bgr: np.ndarray,
    gray: np.ndarray,
    paper: float,
    boxes: list[tuple[str, tuple[int, int, int, int], float]],
    occupied: list[tuple[int, int, int, int]],
    *,
    conservative: bool,
) -> None:
    h, w = gray.shape[:2]
    min_ink = 0.02 if conservative else 0.004
    overlap_cut = 0.38 if conservative else 0.5
    for mask in detect_bubbles(bgr):
        box = _box_from_mask(mask, pad=2)
        if not box:
            continue
        area = box[2] * box[3]
        if conservative and (area < w * h * 0.0035 or area > w * h * 0.22):
            continue
        if _ink_density(gray, box, paper) < min_ink and area < w * h * 0.008:
            continue
        if any(_overlap_ratio(box, other) > overlap_cut for other in occupied):
            continue
        ys, xs = np.where(mask > 0)
        peri = cv2.arcLength(np.array(list(zip(xs[::8], ys[::8]))).reshape(-1, 1, 2), True) if len(xs) > 16 else 1
        mask_area = float((mask > 0).sum())
        circ = 4 * np.pi * mask_area / max(1.0, peri * peri)
        tall = box[3] > box[2] * 1.7
        if conservative and circ < 0.20 and not tall:
            continue
        kind = _classify_kind(box, w, h, circ)
        if conservative and kind == "sfx":
            continue
        boxes.append((kind, box, circ))
        occupied.append(box)

    for box in detect_vertical_strips(bgr, occupied):
        boxes.append(("narration", box, 0.15))
        occupied.append(box)


def _contain_frac(inner: TextRegion, outer: TextRegion) -> float:
    ix0, iy0, ix1, iy1 = inner.x, inner.y, inner.x + inner.w, inner.y + inner.h
    ox0, oy0, ox1, oy1 = outer.x, outer.y, outer.x + outer.w, outer.y + outer.h
    cx0, cy0 = max(ix0, ox0), max(iy0, oy0)
    cx1, cy1 = min(ix1, ox1), min(iy1, oy1)
    if cx1 <= cx0 or cy1 <= cy0:
        return 0.0
    return ((cx1 - cx0) * (cy1 - cy0)) / max(1.0, inner.w * inner.h)


def _union_box(a: TextRegion, b: TextRegion) -> tuple[float, float, float, float]:
    x0 = min(a.x, b.x)
    y0 = min(a.y, b.y)
    x1 = max(a.x + a.w, b.x + b.w)
    y1 = max(a.y + a.h, b.y + b.h)
    return x0, y0, max(24.0, x1 - x0), max(24.0, y1 - y0)


def _merge_pair(a: TextRegion, b: TextRegion) -> TextRegion:
    winner = a
    if b.locked and not a.locked:
        winner = b
    elif (b.text or "").strip() and not (a.text or "").strip() and not a.locked:
        winner = b
    elif (not a.locked) and (not b.locked) and (b.w * b.h) > (a.w * a.h * 1.15):
        winner = b
    merged = winner.model_copy(deep=True)
    merged.x, merged.y, merged.w, merged.h = _union_box(a, b)
    if not (merged.text or "").strip():
        merged.text = (a.text or b.text or "").strip()
    if not (merged.text_short or "").strip():
        merged.text_short = a.text_short or b.text_short
    if not merged.speaker:
        merged.speaker = a.speaker or b.speaker
    merged.skip = bool(a.skip and b.skip)
    if a.kind == "dialogue" or b.kind == "dialogue":
        merged.kind = "dialogue"
    elif a.kind == "thought" or b.kind == "thought":
        merged.kind = "thought"
    merged.locked = bool(a.locked or b.locked)
    merged.direction = pick_direction(merged.w, merged.h, merged.kind)  # type: ignore[arg-type]
    merged.max_chars = estimate_max_chars(merged.w, merged.h, merged.direction)
    return merged


def consolidate_regions(regions: list[TextRegion], page_w: int = 0, page_h: int = 0) -> list[TextRegion]:
    page_area = max(1, int(page_w) * int(page_h)) if page_w and page_h else 0
    kept: list[TextRegion] = []
    for item in regions:
        region = item.model_copy(deep=True)
        manual = bool((region.extra or {}).get("manual"))
        if region.locked or manual:
            kept.append(region)
            continue
        thin_tall = region.h >= 50 and region.h > region.w * 2.0 and region.w >= 16
        if (min(region.w, region.h) < 28 or region.w * region.h < 1100) and not thin_tall:
            continue
        if page_area and region.skip and region.w * region.h < page_area * 0.008:
            continue
        if page_w and page_h and _is_banner(region.x, region.y, region.w, region.h, page_w, page_h):
            region.skip = True
            extra = dict(region.extra or {})
            extra["banner"] = True
            region.extra = extra
            if not region.locked:
                region.text = ""
                region.text_short = ""
                region.overflow = False
        kept.append(region)

    changed = True
    while changed:
        changed = False
        kept.sort(key=lambda r: r.w * r.h, reverse=True)
        nxt: list[TextRegion] = []
        used: set[str] = set()
        for i, a in enumerate(kept):
            if a.id in used:
                continue
            acc = a
            for b in kept[i + 1 :]:
                if b.id in used:
                    continue
                if acc.locked and b.locked:
                    continue
                overlap = _overlap_ratio(
                    (int(acc.x), int(acc.y), int(acc.w), int(acc.h)),
                    (int(b.x), int(b.y), int(b.w), int(b.h)),
                )
                if overlap > 0.42 or _contain_frac(b, acc) > 0.62 or _contain_frac(acc, b) > 0.62:
                    acc = _merge_pair(acc, b)
                    used.add(b.id)
                    changed = True
            used.add(a.id)
            nxt.append(acc)
        kept = nxt

    kept.sort(key=lambda r: (-(r.x + r.w / 2), r.y))
    return kept


def merge_selected_regions(regions: list[TextRegion], region_ids: list[str]) -> list[TextRegion]:
    wanted = {item for item in region_ids if item}
    picked = [item for item in regions if item.id in wanted]
    if len(picked) < 2:
        raise ValueError("先多选至少两框再合并")
    acc = picked[0]
    for item in picked[1:]:
        acc = _merge_pair(acc, item)
    return [acc if item.id == picked[0].id else item for item in regions if item.id == picked[0].id or item.id not in wanted]


def detect_regions(image_bytes: bytes, *, prefer_yolo: bool = True) -> list[TextRegion]:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return []
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    paper = float(np.mean(estimate_paper_bgr(bgr)))
    boxes: list[tuple[str, tuple[int, int, int, int], float]] = []
    occupied: list[tuple[int, int, int, int]] = []
    engine = "cv"

    if prefer_yolo:
        try:
            from yizi.bubble_yolo import detect_boxes as yolo_boxes

            for x, y, bw, bh, score in yolo_boxes(bgr):
                box = (x, y, bw, bh)
                if any(_overlap_ratio(box, other) > 0.45 for other in occupied):
                    continue
                kind = _classify_kind(box, w, h, 0.4)
                boxes.append((kind, box, 0.4))
                occupied.append(box)
                engine = "yolo"
        except Exception:
            pass

    yolo_ready = engine == "yolo" and len(boxes) >= 2
    if yolo_ready:
        _gap_fill_boxes(bgr, gray, paper, boxes, occupied, conservative=True)
    else:
        _gap_fill_boxes(bgr, gray, paper, boxes, occupied, conservative=False)
        for box in detect_sfx_blobs(bgr, occupied):
            boxes.append(("sfx", box, 0.12))
            occupied.append(box)

    regions: list[TextRegion] = []
    for kind, box, _circ in boxes:
        x, y, bw, bh = box
        if bw < 28 and bh > bw * 2.2:
            grow = min(12, 32 - bw)
            x = max(0, x - grow // 2)
            bw = min(w - x, bw + grow)
        direction = pick_direction(bw, bh, kind)
        banner = _is_banner(x, y, bw, bh, w, h)
        regions.append(
            TextRegion(
                id=new_region_id(),
                kind=kind,  # type: ignore[arg-type]
                direction=direction,  # type: ignore[arg-type]
                x=float(x),
                y=float(y),
                w=float(bw),
                h=float(bh),
                skip=kind == "sfx" or banner,
                max_chars=estimate_max_chars(bw, bh, direction),
                extra={"detected": True, "engine": engine, "banner": banner},
            )
        )
    return consolidate_regions(regions, w, h)


def merge_regions(
    existing: list[TextRegion],
    detected: list[TextRegion],
    *,
    replace: bool,
) -> list[TextRegion]:
    if not existing:
        return detected
    locked = [r.model_copy(deep=True) for r in existing if r.locked]
    if not replace:
        return [r.model_copy(deep=True) for r in existing]
    kept_boxes = [(int(r.x), int(r.y), int(r.w), int(r.h)) for r in locked]
    merged = list(locked)
    for item in detected:
        box = (int(item.x), int(item.y), int(item.w), int(item.h))
        if any(_overlap_ratio(box, other) > 0.55 for other in kept_boxes):
            continue
        merged.append(item)
    return consolidate_regions(merged)
