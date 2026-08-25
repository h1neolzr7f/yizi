from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont

from yizi.fonts import load_font
from yizi.models import TextRegion

_NO_LINE_START = set("、。，．！？；：…」』）】》")


def _measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def wrap_horizontal(text: str, cols: int) -> list[str]:
    text = (text or "").replace("\r", "").strip()
    if not text:
        return []
    cols = max(1, int(cols))
    lines: list[str] = []
    for raw in text.split("\n"):
        buf = ""
        for ch in raw:
            nxt = buf + ch
            if len(nxt) <= cols:
                buf = nxt
                continue
            if buf and ch in _NO_LINE_START:
                buf += ch
                continue
            if buf:
                lines.append(buf)
            buf = ch
        if buf:
            lines.append(buf)
    return lines or [text]


def wrap_vertical_columns(text: str, rows: int) -> list[str]:
    text = (text or "").replace("\n", "").replace("\r", "").strip()
    if not text:
        return []
    rows = max(1, int(rows))
    cols: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if len(buf) >= rows:
            cols.append(buf)
            buf = ""
    if buf:
        cols.append(buf)
    return cols


def _draw_glyph(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    ch: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    stroke: tuple[int, int, int, int] | None,
    stroke_w: int,
) -> None:
    x, y = xy
    if stroke and stroke_w:
        for dx, dy in (
            (-stroke_w, 0),
            (stroke_w, 0),
            (0, -stroke_w),
            (0, stroke_w),
            (-stroke_w, -stroke_w),
            (stroke_w, -stroke_w),
            (-stroke_w, stroke_w),
            (stroke_w, stroke_w),
        ):
            draw.text((x + dx, y + dy), ch, font=font, fill=stroke)
    draw.text((x, y), ch, font=font, fill=fill)


def _fits_horizontal(text: str, box_w: int, box_h: int, size: int) -> bool:
    font = load_font(size)
    probe = Image.new("L", (8, 8))
    draw = ImageDraw.Draw(probe)
    cw, ch = _measure(draw, "字", font)
    cw = max(cw, size)
    ch = max(ch, int(size * 1.15))
    cols = max(1, int(box_w // max(1, cw)))
    lines = wrap_horizontal(text, cols)
    return len(lines) * ch <= box_h and all(len(line) <= cols for line in lines)


def _fits_vertical(text: str, box_w: int, box_h: int, size: int) -> bool:
    font = load_font(size)
    probe = Image.new("L", (8, 8))
    draw = ImageDraw.Draw(probe)
    cw, ch = _measure(draw, "字", font)
    cw = max(cw, size)
    ch = max(ch, int(size * 1.05))
    rows = max(1, int(box_h // max(1, ch)))
    cols = wrap_vertical_columns(text, rows)
    return len(cols) * int(cw * 1.15) <= box_w


def fit_font_size(text: str, w: int, h: int, direction: str, min_size: int = 11, max_size: int = 42) -> int:
    lo, hi, best = min_size, max_size, min_size
    check = _fits_vertical if direction == "vertical" else _fits_horizontal
    while lo <= hi:
        mid = (lo + hi) // 2
        if check(text, w, h, mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _paint_horizontal(im: Image.Image, region: TextRegion, text: str, size: int) -> bool:
    draw = ImageDraw.Draw(im)
    font = load_font(size)
    cw, ch = _measure(draw, "字", font)
    cw = max(cw, size)
    ch = max(ch, int(size * 1.15))
    box_w, box_h = int(region.w), int(region.h)
    cols = max(1, int(box_w // max(1, cw)))
    lines = wrap_horizontal(text, cols)
    overflow = len(lines) * ch > box_h
    if overflow:
        max_lines = max(1, box_h // max(1, ch))
        lines = lines[:max_lines]
        if lines:
            lines[-1] = lines[-1][:-1] + "…" if len(lines[-1]) > 1 else "…"
    pad_x = region.x + max(2, (box_w - cols * cw) / 2)
    pad_y = region.y + max(2, (box_h - len(lines) * ch) / 2)
    fill = (22, 22, 24, 255)
    stroke = (255, 255, 255, 230)
    for i, line in enumerate(lines):
        for j, ch_s in enumerate(line):
            _draw_glyph(draw, (pad_x + j * cw, pad_y + i * ch), ch_s, font, fill, stroke, 1)
    return overflow


def _paint_vertical(im: Image.Image, region: TextRegion, text: str, size: int) -> bool:
    draw = ImageDraw.Draw(im)
    font = load_font(size)
    cw, ch = _measure(draw, "字", font)
    cw = max(cw, size)
    ch = max(ch, int(size * 1.05))
    col_w = int(cw * 1.18)
    box_w, box_h = int(region.w), int(region.h)
    rows = max(1, int(box_h // max(1, ch)))
    cols = wrap_vertical_columns(text, rows)
    overflow = len(cols) * col_w > box_w
    if overflow:
        max_cols = max(1, box_w // max(1, col_w))
        cols = cols[:max_cols]
        if cols:
            last = cols[-1]
            cols[-1] = last[:-1] + "…" if len(last) > 1 else "…"
    right = region.x + box_w - max(2, (box_w - len(cols) * col_w) / 2) - col_w
    top = region.y + max(2, (box_h - rows * ch) / 2)
    fill = (22, 22, 24, 255)
    stroke = (255, 255, 255, 230)
    for ci, col in enumerate(cols):
        cx = right - ci * col_w
        for ri, ch_s in enumerate(col):
            _draw_glyph(draw, (cx, top + ri * ch), ch_s, font, fill, stroke, 1)
    return overflow


def typeset_page(width: int, height: int, regions: list[TextRegion]) -> tuple[bytes, list[TextRegion]]:
    im = Image.new("RGBA", (max(1, width), max(1, height)), (0, 0, 0, 0))
    updated: list[TextRegion] = []
    for region in regions:
        item = region.model_copy(deep=True)
        if item.skip:
            item.overflow = False
            updated.append(item)
            continue
        text = (item.text or item.text_short or "").strip()
        if not text:
            item.overflow = False
            updated.append(item)
            continue
        box_w = max(12, int(item.w) - 8)
        box_h = max(12, int(item.h) - 8)
        size = fit_font_size(text, box_w, box_h, item.direction)
        paint_region = item.model_copy(update={"x": item.x + 4, "y": item.y + 4, "w": box_w, "h": box_h})
        if item.direction == "vertical":
            overflow = _paint_vertical(im, paint_region, text, size)
        else:
            overflow = _paint_horizontal(im, paint_region, text, size)
        if len(text) > item.max_chars:
            overflow = True
        item.overflow = overflow
        updated.append(item)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue(), updated
