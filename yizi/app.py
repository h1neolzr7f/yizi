from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from yizi import config
from yizi.config import ROOT, ensure_data_dirs
from yizi.export import compose_page_png, export_page_layers_zip, export_project_zip
from yizi.models import (
    LayerPatch,
    LetteringBackfillRequest,
    LetteringDetectRequest,
    LetteringRunRequest,
    LetteringSeeRequest,
    LetteringWriteRequest,
    PageOrder,
    PagePatch,
    ProjectCreate,
    ProjectPatch,
    RegionCreate,
    RegionMergeRequest,
    RegionPatch,
    SceneCard,
    SceneCardPatch,
    SettingsUpdate,
    StoryAssembleRequest,
    StoryDesignRequest,
    TextRegion,
)
from yizi.store import (
    add_empty_page,
    add_page_from_image,
    create_project,
    delete_project,
    get_page,
    list_projects,
    load_project,
    load_settings,
    page_file,
    page_source_path,
    remove_page,
    public_settings,
    safe_display_name,
    save_project,
    save_settings,
)

FRONTEND = ROOT / "frontend"


def import_sort_key(name: str) -> tuple:
    text = (name or "").replace("\\", "/").lower()
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text))


def _download_headers(filename: str) -> dict[str, str]:
    fallback = "".join(ch if ch.isascii() and ch not in '\\/"' else "_" for ch in filename).strip("._") or "download"
    return {"Content-Disposition": f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename)}"}


def _project(project_id: str):
    try:
        return load_project(project_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, "工程不存在") from exc


def _page(project, page_id: str):
    try:
        return get_page(project, page_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, "页面不存在") from exc


def _clamp_box(page, x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    width = max(1, page.width)
    height = max(1, page.height)
    box_w = min(max(24.0, float(w)), float(width))
    box_h = min(max(24.0, float(h)), float(height))
    box_x = min(max(0.0, float(x)), max(0.0, width - box_w))
    box_y = min(max(0.0, float(y)), max(0.0, height - box_h))
    return box_x, box_y, box_w, box_h


def _region(page, region_id: str) -> TextRegion:
    for region in page.regions:
        if region.id == region_id:
            return region
    raise HTTPException(404, "文字框不存在")


def _settings_out():
    settings = load_settings()
    out = public_settings(settings)
    from yizi.vision_local import status as vision_status

    st = vision_status(settings.lettering_vision)
    out["lettering_vision_ready"] = st["ready"]
    out["lettering_vision_note"] = st["note"]
    from yizi.bubble_yolo import status as detect_status

    det = detect_status()
    out["lettering_detect_ready"] = det["ready"]
    out["lettering_detect_note"] = det["note"]
    return out


def _is_masked_key(value: str) -> bool:
    compact = value.strip()
    if not compact:
        return False
    return compact.startswith("•") or set(compact) <= {"•", "·", "*", " "}


def create_app() -> FastAPI:
    ensure_data_dirs()
    app = FastAPI(title="AI漫画编剧大师", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return (FRONTEND / "index.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "name": "yizi"}

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        return _settings_out()

    @app.put("/api/settings")
    def put_settings(body: SettingsUpdate) -> dict[str, Any]:
        settings = load_settings()
        data = body.model_dump(exclude_none=True)
        raw_key = data.pop("api_key", None)
        if raw_key is None:
            raw_key = data.pop("deepseek_api_key", None)
        if raw_key is not None and not _is_masked_key(str(raw_key)):
            settings.api_key = str(raw_key).strip()
        if "deepseek_api_base" in data:
            settings.api_base = data.pop("deepseek_api_base")
        if "deepseek_model" in data:
            settings.writer_model = data.pop("deepseek_model")
        if "deepseek_outline_model" in data:
            settings.outline_model = data.pop("deepseek_outline_model")
        for key, value in data.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        save_settings(settings)
        return _settings_out()

    @app.post("/api/settings/vision/prepare")
    def prepare_vision() -> dict[str, Any]:
        settings = load_settings()
        from yizi.vision_local import ensure_backend

        try:
            info = ensure_backend(settings.lettering_vision)
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
        out = _settings_out()
        out["prepared"] = info
        return out

    @app.post("/api/settings/detect/prepare")
    def prepare_detect() -> dict[str, Any]:
        from yizi.bubble_yolo import ensure_model, status as detect_status

        try:
            ensure_model()
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
        out = _settings_out()
        out["prepared"] = detect_status()
        return out

    @app.post("/api/settings/writer/test")
    def test_writer() -> dict[str, Any]:
        from yizi.writer import ping_writer

        return ping_writer(load_settings())

    @app.get("/api/projects")
    def projects() -> list[dict[str, Any]]:
        return list_projects()

    @app.post("/api/projects")
    def new_project(body: ProjectCreate | None = None) -> dict[str, Any]:
        req = body or ProjectCreate()
        return create_project(req.name).model_dump()

    @app.post("/api/demo")
    def demo() -> dict[str, Any]:
        from yizi.fixtures import lettering_fixture_png

        project = create_project("编剧体验")
        add_page_from_image(project, lettering_fixture_png(), "示例页 1")
        add_page_from_image(project, lettering_fixture_png(), "示例页 2")
        return save_project(project).model_dump()

    @app.get("/api/projects/{project_id}")
    def read_project(project_id: str) -> dict[str, Any]:
        return _project(project_id).model_dump()

    @app.patch("/api/projects/{project_id}")
    def patch_project(project_id: str, body: ProjectPatch) -> dict[str, Any]:
        project = _project(project_id)
        if body.name is not None:
            project.name = safe_display_name(body.name, project.name or "未命名剧本")
        if body.premise is not None:
            project.premise = body.premise
        if body.cast is not None:
            project.cast = body.cast
        if body.tone is not None:
            project.tone = body.tone
        if body.style is not None:
            project.style = body.style
        if body.expanded is not None:
            project.expanded = body.expanded
        if body.outline is not None:
            project.outline = body.outline
        return save_project(project).model_dump()

    @app.delete("/api/projects/{project_id}")
    def remove_project(project_id: str) -> dict[str, Any]:
        try:
            delete_project(project_id)
        except ValueError as exc:
            raise HTTPException(404, "工程不存在") from exc
        return {"ok": True}

    @app.post("/api/projects/{project_id}/import")
    async def import_pages(project_id: str, files: list[UploadFile] = File(...)) -> dict[str, Any]:
        project = _project(project_id)
        incoming = sorted(files, key=lambda item: import_sort_key(item.filename or ""))
        added = []
        for upload in incoming:
            raw = await upload.read()
            if not raw:
                continue
            name = safe_display_name(Path(upload.filename or "").stem, f"第{len(project.pages) + 1}页")
            try:
                page = add_page_from_image(project, raw, name)
            except Exception as exc:
                raise HTTPException(400, f"读不了 {upload.filename}: {exc}") from exc
            added.append(page.id)
        save_project(project)
        return {"project": project.model_dump(), "added": added}

    @app.post("/api/projects/{project_id}/pages")
    def add_page(project_id: str, body: PagePatch | None = None) -> dict[str, Any]:
        project = _project(project_id)
        page = add_empty_page(project, body.name if body else None)
        save_project(project)
        return page.model_dump()

    @app.patch("/api/projects/{project_id}/pages/{page_id}")
    def patch_page(project_id: str, page_id: str, body: PagePatch) -> dict[str, Any]:
        project = _project(project_id)
        page = _page(project, page_id)
        if body.name is not None:
            page.name = safe_display_name(body.name, page.name or "未命名页")
        if body.notes is not None:
            page.notes = body.notes
        if body.page_summary is not None:
            page.page_summary = body.page_summary
        save_project(project)
        return page.model_dump()

    @app.delete("/api/projects/{project_id}/pages/{page_id}")
    def delete_page(project_id: str, page_id: str) -> dict[str, Any]:
        project = _project(project_id)
        _page(project, page_id)
        remove_page(project, page_id)
        return save_project(project).model_dump()

    @app.post("/api/projects/{project_id}/pages/order")
    def reorder_pages(project_id: str, body: PageOrder) -> dict[str, Any]:
        project = _project(project_id)
        ids = [page.id for page in project.pages]
        incoming = [item for item in body.page_ids if item in ids]
        if set(incoming) != set(ids) or len(incoming) != len(ids):
            raise HTTPException(400, "页序不完整")
        lookup = {page.id: page for page in project.pages}
        project.pages = [lookup[item] for item in incoming]
        return save_project(project).model_dump()

    @app.post("/api/projects/{project_id}/pages/{page_id}/lettering/detect")
    def detect(project_id: str, page_id: str, body: LetteringDetectRequest | None = None) -> dict[str, Any]:
        project = _project(project_id)
        page = _page(project, page_id)
        req = body or LetteringDetectRequest()
        if page_source_path(project.id, page) is None:
            raise HTTPException(400, "这一页还没有成页图")
        try:
            from yizi.pipeline import detect_page

            detect_page(project, page, replace=req.replace, settings=load_settings())
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        save_project(project)
        return {"page": page.model_dump(), "count": len(page.regions)}

    @app.post("/api/projects/{project_id}/pages/{page_id}/lettering/see")
    def see(project_id: str, page_id: str, body: LetteringSeeRequest | None = None) -> dict[str, Any]:
        project = _project(project_id)
        page = _page(project, page_id)
        if page_source_path(project.id, page) is None:
            raise HTTPException(400, "这一页还没有成页图")
        req = body or LetteringSeeRequest()
        from yizi.pipeline import see_page

        see_page(project, page, load_settings(), force=req.force)
        save_project(project)
        return {"page": page.model_dump(), "scene": page.scene.model_dump() if page.scene else None}

    @app.get("/api/projects/{project_id}/story/preview")
    def preview_story_endpoint(project_id: str) -> dict[str, Any]:
        project = _project(project_id)
        from yizi.store import has_assemble_undo
        from yizi.story import preview_pages

        settings = load_settings()
        out = preview_pages(project)
        out["has_undo"] = has_assemble_undo(project.id)
        out["writer_ready"] = bool((settings.api_key or "").strip())
        out["hint"] = "组装只往已导入成页嵌字，不生成新页。套话场景卡会重看。"
        return out

    @app.post("/api/projects/{project_id}/story/snapshot")
    def snapshot_story_endpoint(project_id: str) -> dict[str, Any]:
        project = _project(project_id)
        from yizi.pipeline import snapshot_assemble

        snapshot_assemble(project)
        return {"ok": True, "has_undo": True}

    @app.post("/api/projects/{project_id}/story/undo")
    def undo_story_endpoint(project_id: str) -> dict[str, Any]:
        project = _project(project_id)
        from yizi.pipeline import undo_assemble

        try:
            restored, meta = undo_assemble(project)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"project": restored.model_dump(), "meta": meta}

    @app.post("/api/projects/{project_id}/story/design")
    def design_story_endpoint(project_id: str, body: StoryDesignRequest | None = None) -> dict[str, Any]:
        project = _project(project_id)
        from yizi.pipeline import prepare_pages
        from yizi.writer import design_story

        req = body or StoryDesignRequest()
        settings = load_settings()
        prep = prepare_pages(project, settings, detect_missing=req.detect_missing, see_missing=req.see_missing)
        meta = design_story(project, settings)
        meta.update(prep)
        save_project(project)
        return {"project": project.model_dump(), "meta": meta}

    @app.post("/api/projects/{project_id}/story/assemble")
    def assemble_story_endpoint(project_id: str, body: StoryAssembleRequest | None = None) -> dict[str, Any]:
        project = _project(project_id)
        req = body or StoryAssembleRequest()
        from yizi.pipeline import assemble_manga

        meta = assemble_manga(
            project,
            load_settings(),
            compose=req.compose,
            detect_missing=req.detect_missing,
            see_missing=req.see_missing,
            snapshot=req.snapshot,
        )
        return {"project": project.model_dump(), "meta": meta}

    @app.post("/api/projects/{project_id}/lettering/write")
    def write(project_id: str, body: LetteringWriteRequest | None = None) -> dict[str, Any]:
        project = _project(project_id)
        req = body or LetteringWriteRequest()
        if req.scope in {"page", "region"} and not req.page_id:
            raise HTTPException(400, "先指定一页")
        if req.scope == "region" and not req.region_id:
            raise HTTPException(400, "先指定一框")
        from yizi.writer import write_story

        meta = write_story(
            project,
            load_settings(),
            scope=req.scope,
            page_id=req.page_id,
            region_id=req.region_id,
            force=req.force,
        )
        save_project(project)
        return {"project": project.model_dump(), "meta": meta}

    @app.post("/api/projects/{project_id}/pages/{page_id}/lettering/erase")
    def erase(project_id: str, page_id: str) -> dict[str, Any]:
        project = _project(project_id)
        page = _page(project, page_id)
        from yizi.pipeline import erase_page

        layer = erase_page(project, page)
        save_project(project)
        return {"page": page.model_dump(), "layer": layer.model_dump() if layer else None}

    @app.post("/api/projects/{project_id}/pages/{page_id}/lettering/typeset")
    def typeset(project_id: str, page_id: str) -> dict[str, Any]:
        project = _project(project_id)
        page = _page(project, page_id)
        from yizi.pipeline import typeset_current_page

        layer = typeset_current_page(project, page)
        save_project(project)
        return {"page": page.model_dump(), "layer": layer.model_dump() if layer else None}

    @app.post("/api/projects/{project_id}/lettering/backfill")
    def backfill_page_endpoint(project_id: str, body: LetteringBackfillRequest) -> dict[str, Any]:
        project = _project(project_id)
        from yizi.pipeline import backfill_page

        try:
            meta = backfill_page(
                project,
                load_settings(),
                body.page_id,
                detect=body.detect,
                see=body.see,
                write=body.write,
                compose=body.compose,
                force=body.force,
            )
        except FileNotFoundError as exc:
            raise HTTPException(404, "页面不存在") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"project": project.model_dump(), "meta": meta}

    @app.post("/api/projects/{project_id}/lettering/run")
    def run_lettering(project_id: str, body: LetteringRunRequest | None = None) -> dict[str, Any]:
        project = _project(project_id)
        req = body or LetteringRunRequest()
        page_id = req.page_id if req.scope == "page" else None
        if req.scope == "page" and not page_id:
            raise HTTPException(400, "先指定一页")
        from yizi.pipeline import run_project

        meta = run_project(
            project,
            load_settings(),
            page_id=page_id,
            replace=req.replace,
            force=req.force,
            compose=req.compose,
        )
        project = load_project(project_id)
        return {"project": project.model_dump(), "meta": meta}

    @app.post("/api/projects/{project_id}/pages/{page_id}/lettering/prune")
    def prune(project_id: str, page_id: str) -> dict[str, Any]:
        project = _project(project_id)
        page = _page(project, page_id)
        from yizi.pipeline import prune_page

        meta = prune_page(page)
        save_project(project)
        return {"page": page.model_dump(), "meta": meta}

    @app.post("/api/projects/{project_id}/pages/{page_id}/regions/merge")
    def merge_regions_api(project_id: str, page_id: str, body: RegionMergeRequest) -> dict[str, Any]:
        project = _project(project_id)
        page = _page(project, page_id)
        from yizi.pipeline import merge_page_regions

        try:
            merge_page_regions(page, body.region_ids)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        save_project(project)
        return {"page": page.model_dump()}

    @app.post("/api/projects/{project_id}/pages/{page_id}/regions")
    def add_region(project_id: str, page_id: str, body: RegionCreate) -> dict[str, Any]:
        from yizi.detect import estimate_max_chars, new_region_id, pick_direction

        project = _project(project_id)
        page = _page(project, page_id)
        x, y, w, h = _clamp_box(page, body.x, body.y, body.w, body.h)
        direction = body.direction or pick_direction(w, h, body.kind)
        region = TextRegion(
            id=new_region_id(),
            kind=body.kind,
            direction=direction,  # type: ignore[arg-type]
            x=x,
            y=y,
            w=w,
            h=h,
            skip=body.skip or body.kind == "sfx",
            speaker=body.speaker,
            max_chars=estimate_max_chars(w, h, direction),
            extra={"manual": True},
        )
        page.regions.append(region)
        save_project(project)
        return region.model_dump()

    @app.patch("/api/projects/{project_id}/pages/{page_id}/regions/{region_id}")
    def patch_region(project_id: str, page_id: str, region_id: str, body: RegionPatch) -> dict[str, Any]:
        project = _project(project_id)
        page = _page(project, page_id)
        region = _region(page, region_id)
        data = body.model_dump(exclude_none=True)
        if "text" in data and data["text"] != region.text:
            data.setdefault("locked", True)
        for key, value in data.items():
            setattr(region, key, value)
        if any(key in data for key in ("x", "y", "w", "h")):
            region.x, region.y, region.w, region.h = _clamp_box(page, region.x, region.y, region.w, region.h)
        if any(key in data for key in ("w", "h", "direction")):
            from yizi.detect import estimate_max_chars

            region.max_chars = estimate_max_chars(region.w, region.h, region.direction)
        save_project(project)
        return region.model_dump()

    @app.delete("/api/projects/{project_id}/pages/{page_id}/regions/{region_id}")
    def delete_region(project_id: str, page_id: str, region_id: str) -> dict[str, Any]:
        project = _project(project_id)
        page = _page(project, page_id)
        _region(page, region_id)
        page.regions = [item for item in page.regions if item.id != region_id]
        save_project(project)
        return page.model_dump()

    @app.put("/api/projects/{project_id}/pages/{page_id}/scene")
    def put_scene(project_id: str, page_id: str, body: SceneCardPatch) -> dict[str, Any]:
        project = _project(project_id)
        page = _page(project, page_id)
        current = page.scene or SceneCard()
        data = current.model_dump()
        data.update(body.model_dump(exclude_none=True))
        data["source"] = body.source or "human"
        page.scene = SceneCard.model_validate(data)
        save_project(project)
        return page.scene.model_dump()

    @app.get("/api/projects/{project_id}/pages/{page_id}/source.png")
    def page_source(project_id: str, page_id: str) -> Response:
        project = _project(project_id)
        page = _page(project, page_id)
        path = page_source_path(project.id, page)
        if path is None:
            raise HTTPException(404, "这一页还没有成页图")
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "public, max-age=120"})

    def _lettering_layer(page, role: str):
        return next((item for item in page.layers if (item.extra or {}).get("lettering") == role), None)

    @app.get("/api/projects/{project_id}/pages/{page_id}/layers/{role}.png")
    def page_layer_png(project_id: str, page_id: str, role: str) -> Response:
        if role not in {"erase", "text"}:
            raise HTTPException(404, "没有这一层")
        project = _project(project_id)
        page = _page(project, page_id)
        layer = _lettering_layer(page, role)
        path = page_file(project.id, page.id, (layer.file if layer else "") or f"{role}.png", f"{role}.png")
        if not path.is_file():
            raise HTTPException(404, "这一层还没有生成")
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "public, max-age=5"})

    @app.patch("/api/projects/{project_id}/pages/{page_id}/layers/{role}")
    def patch_layer(project_id: str, page_id: str, role: str, body: LayerPatch) -> dict[str, Any]:
        if role not in {"erase", "text"}:
            raise HTTPException(404, "没有这一层")
        project = _project(project_id)
        page = _page(project, page_id)
        layer = _lettering_layer(page, role)
        if not layer:
            raise HTTPException(404, "这一层还没有生成")
        if body.visible is not None:
            layer.visible = body.visible
        save_project(project)
        return layer.model_dump()

    @app.get("/api/projects/{project_id}/pages/{page_id}/preview.png")
    def page_preview(project_id: str, page_id: str) -> Response:
        project = _project(project_id)
        _page(project, page_id)
        return Response(
            compose_page_png(project, page_id, include_lettering=True),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=5"},
        )

    @app.get("/api/projects/{project_id}/pages/{page_id}/export.png")
    def export_png(project_id: str, page_id: str) -> Response:
        project = _project(project_id)
        _page(project, page_id)
        data = compose_page_png(project, page_id, include_lettering=True)
        return Response(data, media_type="image/png", headers=_download_headers(f"{page_id}.png"))

    @app.get("/api/projects/{project_id}/pages/{page_id}/layers.zip")
    def export_layers(project_id: str, page_id: str) -> Response:
        project = _project(project_id)
        page = _page(project, page_id)
        data = export_page_layers_zip(project, page)
        return Response(data, media_type="application/zip", headers=_download_headers(f"{page_id}-layers.zip"))

    @app.get("/api/projects/{project_id}/export.zip")
    def export_zip(project_id: str) -> Response:
        project = _project(project_id)
        data = export_project_zip(project)
        return Response(data, media_type="application/zip", headers=_download_headers(f"{project.name}.zip"))

    @app.middleware("http")
    async def _cache_static(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers.setdefault("Cache-Control", "public, max-age=86400")
        return response

    if FRONTEND.exists():
        app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")
    return app


app = create_app()


def run() -> None:
    import uvicorn

    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    try:
        import cv2

        cv2.setNumThreads(max(1, min(4, os.cpu_count() or 2)))
    except Exception:
        pass
    host = os.environ.get("YIZI_HOST", config.HOST)
    port = int(os.environ.get("YIZI_PORT", str(config.PORT)))
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level=os.environ.get("YIZI_LOG", "warning"),
        access_log=False,
    )
