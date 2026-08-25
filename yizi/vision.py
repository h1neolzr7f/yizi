from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import cv2
import numpy as np

from yizi.cv import estimate_paper_bgr
from yizi.detect import _is_banner, _overlap_ratio, estimate_max_chars, new_region_id, pick_direction
from yizi.models import Page, SceneCard, Settings, TextRegion


def image_hash(image_bytes: bytes) -> str:
    small = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if small is None:
        return hashlib.sha256(image_bytes).hexdigest()[:16]
    tiny = cv2.resize(small, (16, 16), interpolation=cv2.INTER_AREA)
    return hashlib.sha256(tiny.tobytes()).hexdigest()[:16]


def _decode(image_bytes: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)


def _figure_count(bgr: np.ndarray) -> int:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = bgr.shape[:2]
    paper = estimate_paper_bgr(bgr).astype(np.float32)
    diff = np.linalg.norm(bgr.astype(np.float32) - paper, axis=2)
    sat = hsv[:, :, 1]
    body = ((diff > 36) & (sat > 28)).astype(np.uint8) * 255
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    body = cv2.morphologyEx(body, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    num, _labels, stats, _ = cv2.connectedComponentsWithStats(body, connectivity=8)
    count = 0
    area_img = h * w
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < area_img * 0.025 or area > area_img * 0.55:
            continue
        if max(bw, bh) / max(1, min(bw, bh)) > 5:
            continue
        count += 1
    return min(6, count)


def _skin_ratio(bgr: np.ndarray) -> float:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    return float((skin > 0).mean())


def _bubble_side_label(regions: list[TextRegion], page_w: int) -> str:
    if not regions:
        return "没有对白框"
    rights = sum(1 for r in regions if (r.x + r.w / 2) > page_w * 0.55)
    lefts = len(regions) - rights
    if rights > lefts:
        return "对白框多在右侧"
    if lefts > rights:
        return "对白框多在左侧"
    return "对白框左右都有"


def _color_mood(bgr: np.ndarray) -> tuple[str, str]:
    mean = bgr.reshape(-1, 3).mean(axis=0)
    b, g, r = (float(x) for x in mean)
    bright = (r + g + b) / 3.0
    h = bgr.shape[0]
    top = float(bgr[: max(1, h // 2)].mean())
    bot = float(bgr[h // 2 :].mean())
    if bright < 70:
        where, mood = "偏暗室内或夜景", "偏沉"
    elif bright > 170:
        where, mood = "偏亮纸面或白天", "偏明"
    elif bot + 24 < top:
        where, mood = "上亮下暗的夜景或室内", "偏沉"
    else:
        where, mood = "中等亮度的室内外", "平"
    if r > b + 18:
        where = "暖色" + where
        mood = "偏热"
    elif b > r + 18:
        where = "冷色" + where
        mood = "偏冷"
    return where, mood


def _histogram(bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def infer_scene_card(
    image_bytes: bytes,
    *,
    regions: list[TextRegion] | None = None,
    prev_bytes: bytes | None = None,
    notes: str = "",
    cached: SceneCard | None = None,
    force: bool = False,
    settings: Settings | None = None,
    story_note: str = "",
) -> SceneCard:
    digest = image_hash(image_bytes)
    if cached and cached.source == "human" and not force:
        from yizi.story import STALE_PHRASES

        doing = (cached.doing or "").strip()
        if not any(marker in doing for marker in STALE_PHRASES):
            card = cached.model_copy(deep=True)
            card.image_hash = digest
            return card
    if cached and cached.image_hash == digest and cached.source == "vlm" and not force:
        card = cached.model_copy(deep=True)
        card.image_hash = digest
        return card
    bgr = _decode(image_bytes)
    if bgr is None:
        return SceneCard(where="看不清这一页", who="人数不明", doing="画面读失败", image_hash=digest, source="cv")
    h, w = bgr.shape[:2]
    regions = regions or []
    figures = _figure_count(bgr)
    where, mood = _color_mood(bgr)
    side = _bubble_side_label(regions, w)
    who = "画面里几乎看不见人" if figures == 0 else f"大约 {figures} 个人影"
    doing = side
    if any(r.kind == "narration" for r in regions):
        doing += "，有竖旁白条"
    if any(r.kind == "sfx" for r in regions):
        doing += "，有拟声位置"
    jump = ""
    if prev_bytes:
        prev = _decode(prev_bytes)
        if prev is not None:
            dist = float(
                cv2.compareHist(
                    _histogram(bgr).astype(np.float32),
                    _histogram(prev).astype(np.float32),
                    cv2.HISTCMP_BHATTACHARYYA,
                )
            )
            if dist < 0.18:
                jump = "跟上页几乎同一场"
            elif dist < 0.42:
                jump = "跟上页有承接，但机位变了"
            else:
                jump = "跟上页跳场"
    intimacy: str = "none"
    if _skin_ratio(bgr) > 0.18:
        intimacy = "suggestive"
    card = SceneCard(
        where=where,
        who=who,
        doing=doing,
        mood=mood,
        jump_from_prev=jump,
        intimacy=intimacy,  # type: ignore[arg-type]
        image_hash=digest,
        source="cv",
        notes=notes or "",
    )
    vlm = try_vlm_scene(image_bytes, settings, story_note=story_note)
    if vlm:
        vlm.image_hash = digest
        vlm.source = "vlm"
        if not vlm.jump_from_prev:
            vlm.jump_from_prev = jump
        return vlm
    return card


def reading_order(regions: list[TextRegion]) -> list[TextRegion]:
    return sorted(regions, key=lambda r: (-(r.x + r.w / 2), r.y))


def annotate_regions(image_bytes: bytes, regions: list[TextRegion]) -> bytes:
    bgr = _decode(image_bytes)
    if bgr is None:
        return image_bytes
    h, w = bgr.shape[:2]
    overlay = bgr.copy()
    for index, region in enumerate(reading_order(regions), 1):
        x0 = max(0, int(region.x))
        y0 = max(0, int(region.y))
        x1 = min(w - 1, int(region.x + region.w))
        y1 = min(h - 1, int(region.y + region.h))
        if x1 <= x0 or y1 <= y0:
            continue
        color = (0, 0, 220) if region.skip else (36, 36, 255)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
        label = str(index)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        tx, ty = x0, max(th + 4, y0)
        cv2.rectangle(overlay, (tx, ty - th - 6), (tx + tw + 8, ty + 4), (0, 0, 180), -1)
        cv2.putText(overlay, label, (tx + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return buf.tobytes() if ok else image_bytes


def _boost_ink(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    light = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(light)
    return cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR)


def build_crop_sheet(image_bytes: bytes, regions: list[TextRegion], *, cell: int | None = None) -> tuple[bytes | None, list[int]]:
    bgr = _decode(image_bytes)
    if bgr is None:
        return None, []
    h, w = bgr.shape[:2]
    pairs = [
        (index, region)
        for index, region in enumerate(reading_order(regions), 1)
        if not region.skip and not region.locked
    ]
    if not pairs:
        return None, []
    if cell is None:
        cell = 400 if len(pairs) <= 3 else 320 if len(pairs) <= 6 else 268
    cols = 1 if len(pairs) <= 3 else 2
    rows = (len(pairs) + cols - 1) // cols
    gap = 8
    sheet = np.full((rows * cell + gap, cols * cell + gap, 3), 248, np.uint8)
    kept: list[int] = []
    for n, (index, region) in enumerate(pairs):
        rr, cc = divmod(n, cols)
        x0 = max(0, int(region.x - max(6.0, region.w * 0.14)))
        y0 = max(0, int(region.y - max(6.0, region.h * 0.14)))
        x1 = min(w, int(region.x + region.w + max(6.0, region.w * 0.14)))
        y1 = min(h, int(region.y + region.h + max(6.0, region.h * 0.14)))
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crop = _boost_ink(crop)
        inner = cell - 40
        ch, cw = crop.shape[:2]
        scale = min(inner / max(cw, 1), inner / max(ch, 1), 4.0)
        nw, nh = max(12, int(cw * scale)), max(12, int(ch * scale))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC)
        ox = gap + cc * cell + (cell - nw) // 2
        oy = gap + rr * cell + 30
        sheet[oy : oy + nh, ox : ox + nw] = resized
        cv2.rectangle(sheet, (gap + cc * cell + 4, gap + rr * cell + 4), (gap + cc * cell + 40, gap + rr * cell + 26), (40, 40, 200), -1)
        cv2.putText(
            sheet,
            str(index),
            (gap + cc * cell + 8, gap + rr * cell + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        kept.append(index)
    ok, buf = cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return (buf.tobytes() if ok else None), kept


def merge_lettering_payloads(page_payload: dict[str, Any] | None, crop_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not page_payload and not crop_payload:
        return None
    out = {
        "card": None,
        "items": [],
        "extra": [],
        "summary": "",
        "crop": False,
    }
    if page_payload:
        out["card"] = page_payload.get("card")
        out["extra"] = list(page_payload.get("extra") or [])
        out["summary"] = page_payload.get("summary") or ""
        out["items"] = list(page_payload.get("items") or [])
    by_i: dict[int, dict[str, Any]] = {}
    for row in out["items"]:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("i") or row.get("index") or 0)
        except (TypeError, ValueError):
            continue
        if idx:
            by_i[idx] = dict(row)
            by_i[idx]["i"] = idx
    crop_items = [row for row in (crop_payload or {}).get("items") or [] if isinstance(row, dict)]
    if crop_items:
        from collections import Counter

        scored: list[tuple[int, str, dict[str, Any]]] = []
        for row in crop_items:
            try:
                idx = int(row.get("i") or row.get("index") or 0)
            except (TypeError, ValueError):
                continue
            if not idx:
                continue
            scored.append((idx, sanitize_vlm_text(str(row.get("text") or "")), row))
        crop_texts = [text for _idx, text, _row in scored if text]
        out["crop"] = True
        if not out["card"] and crop_payload and crop_payload.get("card"):
            out["card"] = crop_payload.get("card")
        all_same = len(crop_texts) >= 3 and len(set(crop_texts)) == 1
        counts = Counter(crop_texts)
        seen: set[str] = set()
        for idx, text, row in scored:
            cur = by_i.get(idx, {"i": idx})
            if row.get("speaker"):
                cur["speaker"] = str(row["speaker"]).strip()
            action = _clean_vision_action(str(row.get("doing") or ""))
            if action:
                cur["doing"] = action
            if row.get("kind") and not cur.get("kind"):
                cur["kind"] = row["kind"]
            if not all_same:
                page_text = str(cur.get("text") or "")
                if text and counts[text] > 1 and text in seen:
                    by_i[idx] = cur
                    continue
                if text and draft_needs_crop(page_text):
                    cur["text"] = row.get("text") or text
                    seen.add(text)
            by_i[idx] = cur
    out["items"] = list(by_i.values())
    _enrich_card_from_items(out.get("card"), out["items"])
    return out


_EXAMPLE_LEAKS = ("伸指点旁边的桩", "点旁边的桩", "系缆的粗桩", "一只手抓着书包带")


def _clean_vision_action(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    if any(leak in value for leak in _EXAMPLE_LEAKS):
        return ""
    return value


def _enrich_card_from_items(card: SceneCard | None, items: list[Any]) -> None:
    if card is None:
        return
    bits: list[str] = []
    for row in items:
        if not isinstance(row, dict):
            continue
        doing = _clean_vision_action(str(row.get("doing") or ""))
        speaker = str(row.get("speaker") or "").strip()
        if not doing:
            continue
        bits.append(f"{speaker}：{doing}" if speaker else doing)
    if not bits:
        return
    extra = "；".join(bits)
    current = (card.doing or "").strip()
    if len(current) < 24 or current in {"认系船柱", "走路", "对视", "动作不明"}:
        if len(extra) > len(current):
            card.doing = extra
        return
    if extra and extra not in current:
        card.doing = f"{current}。特写：{extra}"


def _norm_kind(raw: str) -> str:
    kind = str(raw or "dialogue").strip().lower()
    if kind in {"title", "banner", "logo", "credit"}:
        return "title"
    if kind in {"dialogue", "narration", "thought", "sfx"}:
        return kind
    return "dialogue"


def _norm_dir(raw: str, w: float, h: float, kind: str) -> str:
    token = str(raw or "").strip().lower()
    if token in {"v", "vertical", "vert"}:
        return "vertical"
    if token in {"h", "horizontal", "horiz"}:
        return "horizontal"
    return pick_direction(w, h, kind if kind != "title" else "dialogue")


def _pct_box(row: dict[str, Any], page_w: int, page_h: int) -> tuple[float, float, float, float] | None:
    try:
        x, y, w, h = (float(row.get(k) or 0) for k in ("x", "y", "w", "h"))
    except (TypeError, ValueError):
        return None
    if max(x, y, w, h) <= 1.5:
        x, y, w, h = x * 100, y * 100, w * 100, h * 100
    if w < 2 or h < 2:
        return None
    return (
        max(0.0, min(page_w - 8.0, x / 100.0 * page_w)),
        max(0.0, min(page_h - 8.0, y / 100.0 * page_h)),
        max(16.0, min(page_w * 0.95, w / 100.0 * page_w)),
        max(16.0, min(page_h * 0.95, h / 100.0 * page_h)),
    )


_JP_KANA = re.compile(r"[ぁ-んァ-ン]")
_CREDIT_MARK = ("描いた", "カスガ", "スガ", "ライセンス", "BY-SA", "creativecommons")
_WEAK_DRAFT = ("黑发女生", "我听别人", "画面里的人", "女生A", "场面尴尬", "讨论和表情", "行走和交谈")
_OPEN_TAIL = ("的", "说", "我", "不", "，", "、", "‘", "“", "这", "个", "也", "是")


def sanitize_vlm_text(text: str) -> str:
    text = (text or "").strip()
    if not text or _JP_KANA.search(text):
        return ""
    return text


def draft_needs_crop(text: str) -> bool:
    cleaned = sanitize_vlm_text(text)
    if not cleaned:
        return True
    if any(marker in cleaned for marker in _WEAK_DRAFT):
        return True
    if cleaned.endswith(_OPEN_TAIL):
        return True
    return len(cleaned) <= 5 and not cleaned.endswith(("。", "！", "？", "…", "啊", "呀"))


def _is_credit_row(row: dict[str, Any], x: float, y: float, page_w: int, page_h: int) -> bool:
    text = str(row.get("text") or "")
    if _JP_KANA.search(text) or any(token in text for token in _CREDIT_MARK):
        return True
    if y < page_h * 0.14 and x < page_w * 0.28:
        return True
    if x <= 8 and y <= 8:
        return True
    return False


def apply_vlm_lettering(page: Page, payload: dict[str, Any], *, force: bool = False, write_text: bool = False) -> dict[str, int]:
    items = payload.get("items") or []
    extra = payload.get("extra") or []
    ordered = reading_order(page.regions)
    by_index = {}
    for row in items:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("i") or row.get("index") or 0)
        except (TypeError, ValueError):
            continue
        if idx:
            by_index[idx] = row
    written = 0
    skipped = 0
    for index, region in enumerate(ordered, 1):
        row = by_index.get(index)
        if not row:
            continue
        if region.locked and not force:
            continue
        kind = _norm_kind(str(row.get("kind") or region.kind))
        skip = bool(row.get("skip")) or kind == "title" or kind == "sfx"
        if kind == "title":
            kind = "dialogue"
        raw_text = str(row.get("text") or "").strip()
        text = sanitize_vlm_text(raw_text)
        if skip:
            region.skip = True
            if write_text and not region.locked:
                region.text = ""
                region.text_short = ""
                region.overflow = False
            extra_meta = dict(region.extra or {})
            extra_meta["vlm"] = True
            if _is_banner(region.x, region.y, region.w, region.h, page.width, page.height):
                extra_meta["banner"] = True
            action = _clean_vision_action(str(row.get("doing") or ""))
            if action:
                extra_meta["action"] = action
            region.extra = extra_meta
            skipped += 1
            continue
        region.skip = False
        region.kind = kind  # type: ignore[assignment]
        if row.get("dir") or row.get("direction"):
            region.direction = _norm_dir(str(row.get("dir") or row.get("direction") or ""), region.w, region.h, kind)  # type: ignore[assignment]
        if row.get("speaker"):
            region.speaker = str(row["speaker"]).strip()
        if write_text and text:
            from yizi.story import fit_text

            fitted, short, overflow = fit_text(text, region.max_chars)
            region.text = fitted
            region.text_short = short or fitted
            region.overflow = overflow or len(raw_text) > region.max_chars
            if fitted:
                written += 1
        elif write_text and raw_text and not region.locked:
            region.text = ""
            region.text_short = ""
            region.overflow = False
        extra_meta = dict(region.extra or {})
        extra_meta["vlm"] = True
        action = _clean_vision_action(str(row.get("doing") or ""))
        if action:
            extra_meta["action"] = action
        region.extra = extra_meta
    added = 0
    existing = list(page.regions)
    for row in extra:
        if not isinstance(row, dict):
            continue
        box = _pct_box(row, page.width, page.height)
        if not box:
            continue
        x, y, w, h = box
        if _is_credit_row(row, x, y, page.width, page.height):
            continue
        kind = _norm_kind(str(row.get("kind") or "dialogue"))
        skip = bool(row.get("skip")) or kind == "title" or kind == "sfx"
        if kind == "title":
            kind = "dialogue"
            skip = True
        if _is_banner(x, y, w, h, page.width, page.height):
            skip = True
        if not write_text:
            continue
        if any(
            _overlap_ratio((int(x), int(y), int(w), int(h)), (int(r.x), int(r.y), int(r.w), int(r.h))) > 0.42
            for r in existing
        ):
            continue
        direction = _norm_dir(str(row.get("dir") or row.get("direction") or ""), w, h, kind)
        from yizi.story import fit_text

        raw_extra = "" if skip else sanitize_vlm_text(str(row.get("text") or ""))
        max_chars = estimate_max_chars(w, h, direction)
        text, short, overflow = fit_text(raw_extra, max_chars) if raw_extra else ("", "", False)
        region = TextRegion(
            id=new_region_id(),
            kind=kind if kind != "title" else "dialogue",  # type: ignore[arg-type]
            direction=direction,  # type: ignore[arg-type]
            x=x,
            y=y,
            w=w,
            h=h,
            skip=skip,
            speaker=str(row.get("speaker") or "").strip(),
            max_chars=max_chars,
            text=text,
            text_short=short or text,
            overflow=overflow,
            extra={"detected": True, "engine": "vlm", "vlm": True, "banner": skip and _is_banner(x, y, w, h, page.width, page.height)},
        )
        existing.append(region)
        added += 1
        if text:
            written += 1
    from yizi.detect import consolidate_regions

    page.regions = consolidate_regions(existing, page.width, page.height)
    summary = str(payload.get("summary") or "").strip()
    if summary and (force or not (page.page_summary or "").strip()):
        page.page_summary = summary
    return {"written": written, "skipped": skipped, "added": added}


def try_vlm_lettering(
    image_bytes: bytes,
    page: Page,
    settings: Settings | None = None,
    *,
    story_note: str = "",
) -> dict[str, Any] | None:
    try:
        from yizi.vision_local import try_crop_lettering, try_page_lettering

        backend = settings.lettering_vision if settings else "qwen3vl_4b"
        ordered = reading_order(page.regions)
        note = json_note(ordered)
        annotated = annotate_regions(image_bytes, ordered) if ordered else None
        page_payload = try_page_lettering(
            image_bytes,
            backend,
            annotated=annotated,
            box_note=note,
            story_note=story_note,
        )
        sheet, ids = build_crop_sheet(image_bytes, page.regions)
        crop_payload = try_crop_lettering(sheet, backend, ids=ids, story_note=story_note) if sheet else None
        return merge_lettering_payloads(page_payload, crop_payload)
    except Exception:
        import logging

        logging.getLogger("yizi.vision").exception("本机视觉写词失败")
        return None


def json_note(regions: list[TextRegion]) -> str:
    rows = []
    for index, region in enumerate(regions, 1):
        rows.append(
            {
                "i": index,
                "kind": region.kind,
                "skip": region.skip,
                "locked": region.locked,
                "max": region.max_chars,
                "dir": "v" if region.direction == "vertical" else "h",
            }
        )
    return json.dumps(rows, ensure_ascii=False)


def try_vlm_scene(image_bytes: bytes, settings: Settings | None = None, *, story_note: str = "") -> SceneCard | None:
    """本机 VLM 只出场景卡。默认 Qwen3-VL-4B；模型未就绪或 YIZI_VLM=0 时退回几何。图不上网。"""
    try:
        from yizi.vision_local import try_scene

        backend = settings.lettering_vision if settings else "qwen3vl_4b"
        return try_scene(image_bytes, backend, story_note=story_note)
    except Exception:
        import logging

        logging.getLogger("yizi.vision").exception("本机视觉看图失败")
        return None


def enrich_region_tone(regions: list[TextRegion], card: SceneCard) -> list[TextRegion]:
    out: list[TextRegion] = []
    for region in regions:
        item = region.model_copy(deep=True)
        if not item.tone:
            if item.kind == "narration":
                item.tone = "旁白，冷静"
            elif item.kind == "thought":
                item.tone = "内心"
            elif item.kind == "sfx":
                item.tone = "拟声，可跳过"
            else:
                item.tone = card.mood or "对白"
        if not item.speaker:
            if item.kind == "narration":
                item.speaker = "旁白"
            elif item.kind == "sfx":
                item.speaker = "拟声"
            elif "左" in card.who:
                item.speaker = "左边的人"
            else:
                item.speaker = "画面里的人"
        out.append(item)
    return out


def scene_as_prompt(card: SceneCard | None, page_name: str, notes: str = "") -> dict[str, Any]:
    card = card or SceneCard()
    return {
        "page": page_name,
        "where": card.where,
        "who": card.who,
        "doing": card.doing,
        "mood": card.mood,
        "jump_from_prev": card.jump_from_prev,
        "intimacy": card.intimacy,
        "notes": notes or card.notes,
        "source": card.source,
    }
