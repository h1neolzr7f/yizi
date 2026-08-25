from __future__ import annotations

import re
from typing import Any

from yizi.models import Page, Project, SceneCard, TextRegion

STALE_PHRASES = (
    "行走和交谈",
    "讨论和表情反应",
    "对话和表情反应",
    "对话和表情",
    "场面尴尬",
    "行走交谈",
)

GENERIC_LABELS = (
    "黑发女生",
    "画面里的人",
    "女生A",
    "女生B",
    "女生1",
    "女生2",
    "女生3",
    "女孩1",
    "女孩2",
    "女孩3",
    "左边女孩",
    "右边女孩",
    "左边的人",
    "右边的人",
)

_OPEN_TAIL = ("的", "说", "我", "不", "可", "了", "着", "过", "，", "、", "‘", "“", "'", '"', "这", "个", "也", "是", "在", "把", "被", "和")
_FINISHED = ("。", "！", "？", "…", "啊", "呀", "呢", "吧", "嘛", "哦", "嗯")
_OBJECT_ME = ("管我", "看我", "理我", "叫我", "找我", "救我")


def _hanging_open_tail(text: str) -> bool:
    raw = (text or "").strip()
    if not raw or raw.endswith(_FINISHED):
        return False
    if raw.endswith("我") and any(raw.endswith(token) for token in _OBJECT_ME):
        return False
    return raw.endswith(_OPEN_TAIL)

_PLACE_FAMILIES = (
    ("harbor", ("港口", "码头", "系船", "海边", "岸边", "消波")),
    ("school", ("学校", "教室", "走廊")),
)


def is_dump_line(text: str) -> bool:
    cleaned = (text or "").strip()
    return bool(cleaned) and (_has_stale_phrase(cleaned) or is_generic_label(cleaned))


def is_chopped_draft(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if any(marker in cleaned for marker in ("我听别人",)):
        return True
    if cleaned.endswith(("，", "、", "‘", "“", "'", '"')):
        return True
    if "，" in cleaned:
        last = cleaned.rsplit("，", 1)[-1].strip()
        if last and len(last) <= 2 and not last.endswith(_FINISHED):
            return True
    if cleaned.endswith("可") and not cleaned.endswith(("可以",)):
        return True
    return False


def is_generic_label(text: str) -> bool:
    token = (text or "").strip()
    if not token:
        return False
    return token in GENERIC_LABELS or any(token == item or token.startswith(item) for item in GENERIC_LABELS)


def _has_stale_phrase(text: str) -> bool:
    blob = (text or "").strip()
    return bool(blob) and any(marker in blob for marker in STALE_PHRASES)


def has_foreign_place(text: str, premise: str) -> bool:
    story_place = _place_family(premise or "")
    blob = text or ""
    if not story_place or not blob:
        return False
    if story_place == "harbor" and any(key in blob for key in ("学校", "教室", "走廊")):
        return True
    if story_place == "school" and any(key in blob for key in ("港口", "码头", "系船", "海边")):
        return True
    other = _place_family(blob)
    return bool(other and other != story_place)


def _place_family(text: str) -> str:
    blob = text or ""
    for name, keys in _PLACE_FAMILIES:
        if any(key in blob for key in keys):
            return name
    return ""


def scene_conflicts_story(page: Page, project: Project | None = None) -> bool:
    if not page.scene or not project:
        return False
    story = f"{project.premise or ''} {project.expanded or ''}"
    scene_txt = f"{page.scene.where or ''} {page.scene.doing or ''}"
    story_place = _place_family(story)
    scene_place = _place_family(scene_txt)
    return bool(story_place and scene_place and story_place != scene_place)


def scene_is_usable(page: Page, project: Project | None = None) -> bool:
    scene = page.scene
    if not scene or not (scene.where or scene.who or scene.doing):
        return False
    if _has_stale_phrase(scene.doing or ""):
        return False
    if _has_stale_phrase(page.page_summary or "") and scene.source != "vlm":
        return False
    if scene_conflicts_story(page, project):
        return False
    return True


def scene_needs_see(page: Page, project: Project | None = None) -> bool:
    scene = page.scene
    if not scene:
        return True
    if scene.source == "cv":
        return True
    return not scene_is_usable(page, project)


def parse_cast_names(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    parts = re.split(r"[、，,/]|和", raw)
    names: list[str] = []
    for part in parts:
        name = part.strip()
        if not name or is_generic_label(name) or len(name) > 8:
            continue
        if name not in names:
            names.append(name)
    return names


def infer_cast_names(project: Project) -> list[str]:
    names = parse_cast_names(project.cast)
    if names:
        return names
    blob = f"{project.name or ''} {project.premise or ''} {project.expanded or ''}"
    if "百科" in blob:
        return ["百科娘", "同伴", "路人"]
    collected: list[str] = []
    for page in project.pages:
        who = (page.scene.who if page.scene else "") or ""
        for name in parse_cast_names(who):
            if name not in collected and name not in {"三人", "两人", "四个", "两位", "三个", "两个"}:
                collected.append(name)
        for region in page.regions:
            speaker = (region.speaker or "").strip()
            if speaker and speaker != "旁白" and not is_generic_label(speaker) and speaker not in collected:
                collected.append(speaker)
    if collected:
        return collected[:6]
    return ["甲", "乙", "丙"]


def sync_cast(project: Project) -> list[str]:
    names = infer_cast_names(project)
    if names and not (project.cast or "").strip():
        project.cast = "、".join(names)
    return parse_cast_names(project.cast) or names


def _pick_name(names: list[str], region: TextRegion, used: dict[str, str]) -> str:
    if not names:
        return "有人"
    key = f"{int(region.x // 40)}:{int(region.y // 40)}"
    if key in used:
        return used[key]
    taken = set(used.values())
    for name in names:
        if name not in taken:
            used[key] = name
            return name
    name = names[len(used) % len(names)]
    used[key] = name
    return name


def assign_speakers(project: Project) -> list[str]:
    names = sync_cast(project)
    for page in project.pages:
        used: dict[str, str] = {}
        for region in page.regions:
            speaker = (region.speaker or "").strip()
            if region.skip:
                if is_generic_label(speaker):
                    region.speaker = ""
                continue
            if region.kind == "narration":
                if not speaker or is_generic_label(speaker):
                    region.speaker = "旁白"
                continue
            if not speaker or is_generic_label(speaker):
                region.speaker = _pick_name(names, region, used)
            elif speaker not in names and speaker != "旁白":
                mapped = _pick_name(names, region, used)
                region.speaker = mapped
    return names


def _usable_clause(text: str, max_chars: int) -> str:
    raw = (text or "").strip()
    if not raw or is_dump_line(raw) or is_generic_label(raw) or _hanging_open_tail(raw):
        return ""
    if len(raw) <= max_chars:
        return raw
    return ""


def fit_text(text: str, max_chars: int) -> tuple[str, str, bool]:
    raw = (text or "").replace("\r", "").strip()
    if max_chars < 5:
        return "", "", False
    if not raw:
        return "", "", False
    if is_dump_line(raw) or is_generic_label(raw):
        return "", "", False
    if is_chopped_draft(raw) and "，" in raw:
        first = raw.split("，", 1)[0].strip()
        kept = _usable_clause(first, max_chars)
        if kept:
            return kept, kept, True
        return "", "", True
    if len(raw) <= max_chars:
        if _hanging_open_tail(raw):
            return "", "", True
        return raw, raw, False
    for sep in ("。", "！", "？", "…"):
        if sep in raw:
            clause = raw.split(sep, 1)[0].strip() + sep
            kept = _usable_clause(clause, max_chars)
            if kept:
                return kept, kept, True
    for sep in ("，", ",", "；"):
        if sep in raw:
            clause = raw.split(sep, 1)[0].strip()
            kept = _usable_clause(clause, max_chars)
            if kept:
                return kept, kept, True
    return "", "", True


def unique_page_texts(page: Page) -> int:
    seen: set[str] = set()
    cleared = 0
    for region in page.regions:
        if region.skip or region.locked:
            continue
        text = (region.text or "").strip()
        if not text:
            continue
        if text in seen:
            region.text = ""
            region.text_short = ""
            region.overflow = False
            cleared += 1
            continue
        seen.add(text)
    return cleared


def preview_pages(project: Project) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    for index, page in enumerate(project.pages):
        scene: SceneCard | None = page.scene
        pages.append(
            {
                "id": page.id,
                "name": page.name or f"第{index + 1}页",
                "order": index + 1,
                "needs_detect": not page.regions,
                "needs_see": scene_needs_see(page, project),
                "where": (scene.where if scene else "") or "",
                "doing": (scene.doing if scene else "") or "",
                "source": (scene.source if scene else "") or "",
            }
        )
    return {
        "pages": pages,
        "need_see": sum(1 for item in pages if item["needs_see"]),
        "need_detect": sum(1 for item in pages if item["needs_detect"]),
        "hint": "组装只往已导入成页嵌字，不生成新页。",
    }
