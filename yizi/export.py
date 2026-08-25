from __future__ import annotations

import io
import json
import zipfile
from typing import Any

from PIL import Image

from yizi.models import Layer, Page, Project
from yizi.store import page_dir, page_file, safe_display_name, safe_filename

_COMPOSE_CACHE: dict[tuple, bytes] = {}


def _role(layer: Layer) -> str:
    return str((layer.extra or {}).get("lettering") or "")


def _compose_key(project: Project, page: Page, include_lettering: bool) -> tuple:
    root = page_dir(project.id, page.id)
    names = [safe_filename(page.source_file or "source.png")]
    if include_lettering:
        names.extend(["erase.png", "text.png"])
    stamps = []
    for name in names:
        path = root / name
        stamps.append(path.stat().st_mtime_ns if path.exists() else 0)
    vis = tuple(
        (role, next((item.visible for item in page.layers if _role(item) == role), False))
        for role in ("erase", "text")
    )
    return (project.id, page.id, include_lettering, project.updated_at, tuple(stamps), vis)


def compose_page_png(project: Project, page_id: str, include_lettering: bool = True) -> bytes:
    page = next(p for p in project.pages if p.id == page_id)
    key = _compose_key(project, page, include_lettering)
    hit = _COMPOSE_CACHE.get(key)
    if hit is not None:
        return hit
    data = _compose_page_png(project, page, include_lettering)
    if len(_COMPOSE_CACHE) > 24:
        _COMPOSE_CACHE.clear()
    _COMPOSE_CACHE[key] = data
    return data


def _compose_page_png(project: Project, page: Page, include_lettering: bool) -> bytes:
    canvas = Image.new("RGBA", (max(1, page.width), max(1, page.height)), (255, 255, 255, 255))
    try:
        source = page_file(project.id, page.id, page.source_file or "source.png")
    except ValueError:
        source = None
    if source is not None and source.is_file():
        canvas = Image.open(source).convert("RGBA")
        if canvas.size != (page.width, page.height):
            canvas = canvas.resize((page.width, page.height), Image.Resampling.LANCZOS)
    if include_lettering:
        for role in ("erase", "text"):
            layer = next((item for item in page.layers if _role(item) == role and item.visible), None)
            if not layer:
                continue
            try:
                path = page_file(project.id, page.id, layer.file, f"{role}.png")
            except ValueError:
                continue
            if not path.is_file():
                continue
            overlay = Image.open(path).convert("RGBA")
            if overlay.size != canvas.size:
                overlay = overlay.resize(canvas.size, Image.Resampling.LANCZOS)
            canvas.alpha_composite(overlay)
    buf = io.BytesIO()
    canvas.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def export_page_layers_zip(project: Project, page: Page) -> bytes:
    root = page_dir(project.id, page.id)
    payload: dict[str, Any] = {
        "page": page.model_dump(),
        "regions": [item.model_dump() for item in page.regions],
        "scene": page.scene.model_dump() if page.scene else None,
        "outline": project.outline,
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("layers.json", json.dumps(payload, ensure_ascii=False, indent=2))
        source_name = safe_filename(page.source_file or "source.png")
        source = root / source_name
        if source.is_file():
            zf.write(source, source_name)
        for layer in page.layers:
            name = safe_filename(layer.file, "layer.png")
            path = root / name
            if path.is_file():
                zf.write(path, name)
        zf.writestr("composed.png", compose_page_png(project, page.id, include_lettering=True))
    return buf.getvalue()


def export_project_zip(project: Project) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.json", json.dumps(project.model_dump(), ensure_ascii=False, indent=2))
        for index, page in enumerate(project.pages, start=1):
            root = page_dir(project.id, page.id)
            prefix = f"pages/{index:03d}_{page.id}/"
            source_name = safe_filename(page.source_file or "source.png")
            source = root / source_name
            if source.is_file():
                zf.write(source, prefix + source_name)
            for layer in page.layers:
                name = safe_filename(layer.file, "layer.png")
                path = root / name
                if path.is_file():
                    zf.write(path, prefix + name)
            try:
                zf.writestr(
                    f"composed/{index:03d}_{safe_display_name(page.name, page.id)}.png",
                    compose_page_png(project, page.id),
                )
            except Exception:
                continue
    return buf.getvalue()
