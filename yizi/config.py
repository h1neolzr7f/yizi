from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("YIZI_DATA", ROOT / "user_data"))
HOST = os.environ.get("YIZI_HOST", "127.0.0.1")
PORT = int(os.environ.get("YIZI_PORT", "8766"))


def projects_dir() -> Path:
    return DATA / "projects"


def settings_path() -> Path:
    return DATA / "settings.json"


def models_dir() -> Path:
    return DATA / "models"


def llama_dir() -> Path:
    return DATA / "bin" / "llama"


_DIRS_READY_FOR: Path | None = None


def ensure_data_dirs() -> None:
    global _DIRS_READY_FOR
    if _DIRS_READY_FOR == DATA and DATA.is_dir() and projects_dir().is_dir():
        return
    DATA.mkdir(parents=True, exist_ok=True)
    projects_dir().mkdir(parents=True, exist_ok=True)
    models_dir().mkdir(parents=True, exist_ok=True)
    llama_dir().mkdir(parents=True, exist_ok=True)
    _DIRS_READY_FOR = DATA
