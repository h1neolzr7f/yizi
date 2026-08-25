from __future__ import annotations

import json
import re
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from yizi.config import ensure_data_dirs, projects_dir, settings_path
from yizi.models import Layer, Page, Project, Settings

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
_UNSAFE_NAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')
MAX_IMAGE_SIDE = 8192
MAX_IMAGE_PIXELS = 40_000_000

_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()
_SETTINGS_LOCK = threading.RLock()
_SETTINGS_CACHE: tuple[float, Settings] | None = None
_PROJECT_CACHE: dict[str, tuple[float, Project]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:10]}"


def safe_id(value: str) -> str:
    token = (value or "").strip()
    if not _SAFE_ID.fullmatch(token) or token in {".", ".."}:
        raise ValueError("invalid id")
    return token


def safe_display_name(name: str, fallback: str = "未命名") -> str:
    text = _UNSAFE_NAME.sub("_", (name or "").strip())
    while ".." in text:
        text = text.replace("..", "_")
    text = text.strip(" .")
    return (text[:80] or fallback)


def safe_filename(name: str, fallback: str = "source.png") -> str:
    token = Path(name or "").name
    if not token or token in {".", ".."} or ".." in token:
        return fallback
    return token


def _contained(root: Path, path: Path) -> Path:
    base = root.resolve()
    target = path.resolve()
    if target != base and base not in target.parents:
        raise ValueError("invalid path")
    return target


def page_file(project_id: str, page_id: str, name: str, fallback: str = "source.png") -> Path:
    return _contained(page_dir(project_id, page_id), page_dir(project_id, page_id) / safe_filename(name, fallback))


def page_source_path(project_id: str, page) -> Path | None:
    path = page_file(project_id, page.id, page.source_file or "source.png")
    return path if path.is_file() else None


def drop_assemble_undo(project_id: str) -> None:
    assemble_undo_path(project_id).unlink(missing_ok=True)


def _project_rlock(project_id: str) -> threading.RLock:
    with _LOCKS_GUARD:
        lock = _LOCKS.get(project_id)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[project_id] = lock
        return lock


@contextmanager
def project_lock(project_id: str) -> Iterator[None]:
    with _project_rlock(project_id):
        yield


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_settings() -> Settings:
    global _SETTINGS_CACHE
    ensure_data_dirs()
    with _SETTINGS_LOCK:
        path = settings_path()
        if not path.exists():
            settings = Settings()
            save_settings(settings)
            return settings.model_copy(deep=True)
        mtime = path.stat().st_mtime
        if _SETTINGS_CACHE and _SETTINGS_CACHE[0] == mtime:
            return _SETTINGS_CACHE[1].model_copy(deep=True)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if "api_key" not in raw and raw.get("deepseek_api_key"):
            raw["api_key"] = raw.get("deepseek_api_key") or ""
            raw["api_base"] = raw.get("deepseek_api_base") or Settings().api_base
            raw["writer_model"] = raw.get("deepseek_model") or Settings().writer_model
            raw["outline_model"] = raw.get("deepseek_outline_model") or Settings().outline_model
        settings = Settings.model_validate({k: v for k, v in raw.items() if k in Settings.model_fields})
        _SETTINGS_CACHE = (mtime, settings)
        return settings.model_copy(deep=True)


def save_settings(settings: Settings) -> Settings:
    global _SETTINGS_CACHE
    ensure_data_dirs()
    with _SETTINGS_LOCK:
        path = settings_path()
        _write_json(path, settings.model_dump())
        try:
            path.chmod(0o600)
        except OSError:
            pass
        _SETTINGS_CACHE = (path.stat().st_mtime, settings.model_copy(deep=True))
        return settings


def public_settings(settings: Settings) -> dict[str, Any]:
    key = (settings.api_key or "").strip()
    masked = ""
    if key:
        masked = key[:3] + "••••••••" + key[-4:] if len(key) > 8 else "••••••••"
    data = settings.model_dump()
    data.pop("api_key", None)
    data["api_key_set"] = bool(key)
    data["api_key_masked"] = masked
    data["deepseek_api_key_set"] = bool(key)
    data["deepseek_api_key_masked"] = masked
    data["deepseek_model"] = settings.writer_model
    data["deepseek_outline_model"] = settings.outline_model
    data["deepseek_api_base"] = settings.api_base
    data["lettering_detect"] = settings.lettering_detect
    return data


def project_dir(project_id: str) -> Path:
    token = safe_id(project_id)
    base = projects_dir().resolve()
    return _contained(base, base / token)


def page_dir(project_id: str, page_id: str) -> Path:
    return project_dir(project_id) / "pages" / safe_id(page_id)


def project_json_path(project_id: str) -> Path:
    return project_dir(project_id) / "project.json"


def load_project(project_id: str) -> Project:
    with project_lock(project_id):
        path = project_json_path(project_id)
        if not path.exists():
            raise FileNotFoundError(project_id)
        mtime = path.stat().st_mtime
        hit = _PROJECT_CACHE.get(project_id)
        if hit and hit[0] == mtime:
            return hit[1].model_copy(deep=True)
        project = Project.model_validate(json.loads(path.read_text(encoding="utf-8")))
        _PROJECT_CACHE[project_id] = (mtime, project)
        return project.model_copy(deep=True)


def save_project(project: Project) -> Project:
    with project_lock(project.id):
        project.updated_at = _now()
        path = project_json_path(project.id)
        _write_json(path, project.model_dump())
        _PROJECT_CACHE[project.id] = (path.stat().st_mtime, project.model_copy(deep=True))
        return project


def assemble_undo_path(project_id: str) -> Path:
    return project_dir(project_id) / "undo.json"


def snapshot_project(project: Project) -> Path:
    path = assemble_undo_path(project.id)
    _write_json(path, {"saved_at": _now(), "project": project.model_dump()})
    return path


def has_assemble_undo(project_id: str) -> bool:
    return assemble_undo_path(project_id).exists()


def restore_project_snapshot(project_id: str) -> Project | None:
    path = assemble_undo_path(project_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("project") if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        return None
    project = Project.model_validate(raw)
    path.unlink(missing_ok=True)
    _PROJECT_CACHE.pop(project_id, None)
    return save_project(project)


def list_projects() -> list[dict[str, Any]]:
    ensure_data_dirs()
    items: list[dict[str, Any]] = []
    for path in projects_dir().glob("*/project.json"):
        try:
            project_id = path.parent.name
            mtime = path.stat().st_mtime
            hit = _PROJECT_CACHE.get(project_id)
            if hit and hit[0] == mtime:
                project = hit[1]
                items.append(
                    {
                        "id": project.id,
                        "name": project.name or "未命名",
                        "updated_at": project.updated_at or "",
                        "pages": len(project.pages),
                    }
                )
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            items.append(
                {
                    "id": data["id"],
                    "name": data.get("name") or "未命名",
                    "updated_at": data.get("updated_at") or "",
                    "pages": len(data.get("pages") or []),
                }
            )
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    items.sort(key=lambda row: row.get("updated_at") or "", reverse=True)
    return items


def create_project(name: str = "未命名剧本") -> Project:
    ensure_data_dirs()
    project = Project(id=new_id("p"), name=safe_display_name(name, "未命名剧本"), created_at=_now(), updated_at=_now())
    project_dir(project.id).mkdir(parents=True, exist_ok=True)
    return save_project(project)


def delete_project(project_id: str) -> None:
    import shutil

    token = safe_id(project_id)
    with project_lock(token):
        _PROJECT_CACHE.pop(token, None)
        root = project_dir(token)
        if root.exists():
            shutil.rmtree(root)


def remove_page(project: Project, page_id: str) -> None:
    import shutil

    token = safe_id(page_id)
    project.pages = [item for item in project.pages if item.id != token]
    drop_assemble_undo(project.id)
    root = page_dir(project.id, token)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)


def get_page(project: Project, page_id: str) -> Page:
    for page in project.pages:
        if page.id == page_id:
            return page
    raise FileNotFoundError(page_id)


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def image_size(data: bytes) -> tuple[int, int]:
    import io

    from PIL import Image

    im = Image.open(io.BytesIO(data))
    return im.size


def add_page_from_image(project: Project, data: bytes, name: str) -> Page:
    import io

    from PIL import Image

    im = Image.open(io.BytesIO(data))
    if im.mode not in {"RGB", "RGBA"}:
        im = im.convert("RGB")
    width, height = im.size
    if width <= 0 or height <= 0:
        raise ValueError("不是图")
    if width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE or width * height > MAX_IMAGE_PIXELS:
        raise ValueError("图太大")
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="PNG")
    page = Page(id=new_id("g"), name=safe_display_name(name, "未命名页"), width=width, height=height, source_file="source.png")
    write_bytes(page_dir(project.id, page.id) / "source.png", buf.getvalue())
    project.pages.append(page)
    return page


def add_empty_page(project: Project, name: str | None = None) -> Page:
    index = len(project.pages) + 1
    page = Page(id=new_id("g"), name=safe_display_name(name or "", f"第{index}页"))
    page_dir(project.id, page.id).mkdir(parents=True, exist_ok=True)
    project.pages.append(page)
    return page


def upsert_layer(project: Project, page: Page, *, role: str, name: str, png: bytes) -> Layer:
    existing = next((layer for layer in page.layers if (layer.extra or {}).get("lettering") == role), None)
    filename = f"{role}.png"
    write_bytes(page_dir(project.id, page.id) / filename, png)
    extra = {"lettering": role}
    if existing:
        existing.name = name
        existing.file = filename
        existing.visible = True
        existing.extra = {**(existing.extra or {}), **extra}
        page.layers = [layer for layer in page.layers if layer.id != existing.id] + [existing]
        return existing
    layer = Layer(id=new_id("l"), name=name, kind="text" if role == "text" else "effect", file=filename, extra=extra)
    page.layers.append(layer)
    return layer
