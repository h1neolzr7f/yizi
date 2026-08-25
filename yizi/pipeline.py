from __future__ import annotations

from typing import Any

from yizi.detect import detect_regions, merge_regions
from yizi.erase import erase_regions
from yizi.export import compose_page_png
from yizi.models import Layer, Page, Project, Settings
from yizi.store import get_page, page_source_path, save_project, upsert_layer
from yizi.typeset import typeset_page
from yizi.vision import enrich_region_tone, infer_scene_card
from yizi.writer import write_story


def flatten_page_bytes(project: Project, page: Page) -> bytes:
    src = page_source_path(project.id, page)
    if src is not None:
        return src.read_bytes()
    return compose_page_png(project, page.id, include_lettering=False)


def _prev_page_bytes(project: Project, page: Page) -> bytes | None:
    ids = [item.id for item in project.pages]
    try:
        idx = ids.index(page.id)
    except ValueError:
        return None
    if idx <= 0:
        return None
    try:
        return flatten_page_bytes(project, project.pages[idx - 1])
    except Exception:
        return None


def detect_page(project: Project, page: Page, *, replace: bool = True, settings: Settings | None = None) -> list:
    if page_source_path(project.id, page) is None:
        raise ValueError("这一页还没有成页图")
    raw = flatten_page_bytes(project, page)
    prefer_yolo = True
    if settings is not None and settings.lettering_detect == "cv":
        prefer_yolo = False
    found = detect_regions(raw, prefer_yolo=prefer_yolo)
    page.regions = merge_regions(page.regions, found, replace=replace)
    return page.regions


def see_page(project: Project, page: Page, settings: Settings | None = None, *, force: bool = False) -> Page:
    raw = flatten_page_bytes(project, page)
    if not page.regions:
        detect_page(project, page, replace=True, settings=settings)
    from yizi.story import scene_needs_see
    from yizi.writer import story_context_note

    needs = force or scene_needs_see(page, project)
    if page.scene and page.scene.source == "human" and not needs:
        page.regions = enrich_region_tone(page.regions, page.scene)
        page.see_hint = "人定场景卡未改。图没有上传。"
        return page
    prev = page.scene
    page.scene = infer_scene_card(
        raw,
        regions=page.regions,
        prev_bytes=_prev_page_bytes(project, page),
        notes=page.notes,
        cached=page.scene,
        force=needs,
        settings=settings,
        story_note=story_context_note(project, page),
    )
    if prev and prev.jump_from_prev and page.scene and not page.scene.jump_from_prev:
        page.scene.jump_from_prev = prev.jump_from_prev
    page.regions = enrich_region_tone(page.regions, page.scene)
    if page.scene and page.scene.source == "vlm":
        if prev and prev.source == "vlm" and not needs:
            page.see_hint = "这一页已经看过，沿用场景卡。图没有上传。"
        else:
            page.see_hint = "本机已看过这一页，场景卡已更新。图没有上传。"
    elif page.scene and page.scene.source == "cv":
        from yizi.vision_local import status as vision_status

        st = vision_status((settings.lettering_vision if settings else "qwen3vl_4b") or "qwen3vl_4b")
        page.see_hint = st.get("note") or "本机视觉没看成，先用了几何反推。"
    else:
        page.see_hint = ""
    return page


def erase_page(project: Project, page: Page) -> Layer | None:
    targets = [item for item in page.regions if not item.skip and (item.text or item.text_short or "").strip()]
    png = erase_regions(flatten_page_bytes(project, page), targets)
    return upsert_layer(project, page, role="erase", name="擦除层", png=png)


def prune_page(page: Page) -> dict[str, int]:
    from yizi.detect import consolidate_regions

    before = len(page.regions)
    page.regions = consolidate_regions(page.regions, page.width, page.height)
    cleared = 0
    banners = 0
    scene = page.scene
    doing = ((scene.doing if scene else "") or page.page_summary or "").strip()
    where = ((scene.where if scene else "") or "").strip()
    if "，" in where:
        where = where.split("，", 1)[0].strip()
    for region in page.regions:
        extra = region.extra or {}
        if extra.get("banner"):
            banners += 1
        if region.locked:
            continue
        text = (region.text or "").strip()
        tiny = region.max_chars < 5
        from yizi.story import fit_text, is_chopped_draft, is_dump_line

        scene_dump = bool(
            region.kind == "narration"
            and text
            and (
                is_dump_line(text)
                or (doing and (text == doing or doing.startswith(text) or text.startswith(doing[:4])))
                or (where and text == where)
            )
        )
        if text and (is_dump_line(text) or is_chopped_draft(text)):
            fitted, short, overflow = fit_text(text, region.max_chars)
            if fitted != text:
                if text:
                    cleared += 1
                region.text = fitted
                region.text_short = short
                region.overflow = overflow
                text = fitted
        if region.skip or tiny:
            if text or region.overflow:
                cleared += 1
            region.text = ""
            region.text_short = ""
            region.overflow = False
            continue
        if scene_dump:
            if text:
                cleared += 1
            region.text = ""
            region.text_short = ""
            region.overflow = False
    return {
        "before": before,
        "after": len(page.regions),
        "removed": before - len(page.regions),
        "cleared": cleared,
        "banners": banners,
    }


def merge_page_regions(page: Page, region_ids: list[str]) -> list:
    from yizi.detect import merge_selected_regions

    page.regions = merge_selected_regions(page.regions, region_ids)
    return page.regions


def typeset_current_page(project: Project, page: Page) -> Layer | None:
    png, updated = typeset_page(page.width, page.height, page.regions)
    page.regions = updated
    return upsert_layer(project, page, role="text", name="文字层", png=png)


def run_project(
    project: Project,
    settings: Settings,
    *,
    page_id: str | None = None,
    replace: bool = True,
    force: bool = False,
    compose: bool = False,
) -> dict[str, Any]:
    pages = [get_page(project, page_id)] if page_id else list(project.pages)
    for page in pages:
        detect_page(project, page, replace=replace, settings=settings)
        see_page(project, page, settings)
        save_project(project)
    write_meta = write_story(
        project,
        settings,
        scope="page" if page_id else "all",
        page_id=page_id,
        force=force,
    )
    if compose:
        for page in pages:
            erase_page(project, page)
            typeset_current_page(project, page)
    save_project(project)
    return {
        "pages": [page.id for page in pages],
        "write": write_meta,
        "outline": project.outline,
        "composed": compose,
    }


def _needs_see(page: Page, project: Project | None = None) -> bool:
    from yizi.story import scene_needs_see

    return scene_needs_see(page, project)


def prepare_pages(
    project: Project,
    settings: Settings,
    *,
    detect_missing: bool = True,
    see_missing: bool = True,
) -> dict[str, int]:
    detected = 0
    seen = 0
    for page in project.pages:
        if detect_missing and not page.regions:
            detect_page(project, page, replace=False, settings=settings)
            detected += 1
        if see_missing and _needs_see(page, project):
            see_page(project, page, settings)
            seen += 1
            save_project(project)
    return {"detected": detected, "seen": seen}


def snapshot_assemble(project: Project) -> None:
    from yizi.store import snapshot_project

    snapshot_project(project)


def undo_assemble(project: Project) -> tuple[Project, dict[str, Any]]:
    from yizi.store import restore_project_snapshot

    restored = restore_project_snapshot(project.id)
    if restored is None:
        raise ValueError("没有可撤销的组装")
    for page in restored.pages:
        if page.regions:
            erase_page(restored, page)
            typeset_current_page(restored, page)
    save_project(restored)
    return restored, {"hint": "已撤回到上次组装前。图层按当时的词重铺了。", "undone": True}


def backfill_page(
    project: Project,
    settings: Settings,
    page_id: str,
    *,
    detect: bool = True,
    see: bool = True,
    write: bool = True,
    compose: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    page = get_page(project, page_id)
    detected = 0
    seen = 0
    if detect and not page.regions:
        detect_page(project, page, replace=False, settings=settings)
        detected = 1
    if see and _needs_see(page, project):
        see_page(project, page, settings)
        seen = 1
    write_meta: dict[str, Any] = {}
    if write:
        write_meta = write_story(project, settings, scope="page", page_id=page.id, force=force)
        prune_page(page)
    if compose:
        erase_page(project, page)
        typeset_current_page(project, page)
    save_project(project)
    hint = write_meta.get("hint") or "本页已按已有框回填。"
    return {
        "page_id": page.id,
        "detected": detected,
        "seen": seen,
        "write": write_meta,
        "composed": compose,
        "hint": hint,
        "engine": write_meta.get("engine") or "local",
        "fallback": write_meta.get("fallback") or "",
    }


def assemble_manga(
    project: Project,
    settings: Settings,
    *,
    compose: bool = True,
    detect_missing: bool = True,
    see_missing: bool = True,
    snapshot: bool = True,
) -> dict[str, Any]:
    from yizi.store import has_assemble_undo
    from yizi.story import assign_speakers
    from yizi.writer import design_story, write_story

    if snapshot and project.pages:
        snapshot_assemble(project)
    if not project.pages:
        design_meta = design_story(project, settings)
        save_project(project)
        return {
            "design": design_meta,
            "write": {},
            "pages": [],
            "composed": False,
            "hint": "故事草稿有了。先导入成页并检出框，再组装。按已有框填，不生成新页新框。",
        }
    prep = prepare_pages(project, settings, detect_missing=detect_missing, see_missing=see_missing)
    assign_speakers(project)
    design_meta = design_story(project, settings)
    write_meta = write_story(project, settings, scope="all")
    from yizi.story import unique_page_texts

    for page in project.pages:
        prune_page(page)
        unique_page_texts(page)
    if compose:
        for page in project.pages:
            erase_page(project, page)
            typeset_current_page(project, page)
    save_project(project)
    hint = "已按看图结果和已有框嵌进成页。不生成新页新框。锁框没动。"
    api_failed = bool(design_meta.get("fallback") or write_meta.get("fallback"))
    used_local = design_meta.get("engine") == "local" or write_meta.get("engine") == "local"
    if api_failed:
        hint += " " + (write_meta.get("hint") or design_meta.get("hint") or "编剧 API 没走成，用的是本机草稿。")
    elif used_local and not (settings.api_key or "").strip():
        hint += " 没填编剧 Key，用的是本机草稿。"
    if prep["seen"]:
        hint += f" 先看了 {prep['seen']} 页。"
    if prep["detected"]:
        hint += f" 空页补检了 {prep['detected']} 页。"
    return {
        "design": design_meta,
        "write": write_meta,
        "pages": [page.id for page in project.pages],
        "composed": compose,
        "detected": prep["detected"],
        "seen": prep["seen"],
        "hint": hint,
        "outline": project.outline,
        "tone": project.tone,
        "expanded": project.expanded,
        "has_undo": has_assemble_undo(project.id),
        "writer_engine": write_meta.get("engine") or design_meta.get("engine") or "local",
        "writer_fallback": write_meta.get("fallback") or design_meta.get("fallback") or "",
    }
