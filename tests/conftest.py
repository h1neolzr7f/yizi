from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ["YIZI_VLM"] = "0"
os.environ["YIZI_DETECT"] = "0"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("YIZI_VLM", "0")
    monkeypatch.setenv("YIZI_DETECT", "0")
    from yizi import config

    monkeypatch.setattr(config, "DATA", tmp_path / "data")
    config.ensure_data_dirs()
    from yizi.app import create_app

    return TestClient(create_app())
