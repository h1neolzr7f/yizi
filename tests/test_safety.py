from __future__ import annotations

import io
import zipfile

from PIL import Image

from tests.helpers import lettering_fixture_png


def _png(w: int = 80, h: int = 80) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (30, 30, 30)).save(buf, "PNG")
    return buf.getvalue()


def test_project_id_traversal_is_rejected(client):
    import pytest

    from yizi.config import DATA
    from yizi.store import delete_project, safe_id

    with pytest.raises(ValueError):
        safe_id("..")
    with pytest.raises(ValueError):
        delete_project("../")

    marker = DATA / "keep-me.txt"
    marker.write_text("safe", encoding="utf-8")
    for bad in ("..", "../", "...", "p%2e%2e"):
        assert client.delete(f"/api/projects/{bad}").status_code in {404, 405, 422}
        assert client.get(f"/api/projects/{bad}").status_code in {404, 405, 422}
    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == "safe"


def test_static_cannot_read_settings(client):
    client.put("/api/settings", json={"api_key": "sk-secret-static"})
    leaked = client.get("/static/../user_data/settings.json")
    assert leaked.status_code in {404, 400}
    assert b"sk-secret-static" not in leaked.content
    assert b"api_key" not in leaked.content


def test_poisoned_source_file_cannot_leave_page(client):
    from yizi.store import load_project, save_project

    client.put("/api/settings", json={"api_key": "sk-secret-leak"})
    pid = client.post("/api/projects", json={"name": "路径"}).json()["id"]
    page_id = client.post(
        f"/api/projects/{pid}/import",
        files=[("files", ("page.png", lettering_fixture_png(), "image/png"))],
    ).json()["project"]["pages"][0]["id"]
    project = load_project(pid)
    project.pages[0].source_file = "../../../settings.json"
    save_project(project)
    res = client.get(f"/api/projects/{pid}/pages/{page_id}/source.png")
    assert res.status_code == 404
    assert b"sk-secret-leak" not in res.content


def test_export_zip_stays_inside_archive(client):
    pid = client.post("/api/projects", json={"name": "压缩"}).json()["id"]
    page_id = client.post(
        f"/api/projects/{pid}/import",
        files=[("files", ("page.png", lettering_fixture_png(), "image/png"))],
    ).json()["project"]["pages"][0]["id"]
    renamed = client.patch(f"/api/projects/{pid}/pages/{page_id}", json={"name": "../../evil"})
    assert renamed.status_code == 200
    assert ".." not in renamed.json()["name"]
    exported = client.get(f"/api/projects/{pid}/export.zip")
    assert exported.status_code == 200
    with zipfile.ZipFile(io.BytesIO(exported.content)) as zf:
        for name in zf.namelist():
            assert ".." not in name
            assert not name.startswith("/")
            assert "\\" not in name


def test_delete_page_drops_stale_undo(client):
    pid = client.post("/api/projects", json={"name": "撤销过期"}).json()["id"]
    imported = client.post(
        f"/api/projects/{pid}/import",
        files=[
            ("files", ("a.png", lettering_fixture_png(), "image/png")),
            ("files", ("b.png", lettering_fixture_png(), "image/png")),
        ],
    ).json()
    first = imported["project"]["pages"][0]["id"]
    assert client.post(f"/api/projects/{pid}/story/snapshot").status_code == 200
    assert client.get(f"/api/projects/{pid}/story/preview").json()["has_undo"] is True
    left = client.delete(f"/api/projects/{pid}/pages/{first}")
    assert left.status_code == 200
    assert len(left.json()["pages"]) == 1
    assert client.post(f"/api/projects/{pid}/story/undo").status_code == 400
    assert client.get(f"/api/projects/{pid}").json()["pages"][0]["id"] != first


def test_empty_page_detect_and_see_are_400(client):
    pid = client.post("/api/projects", json={"name": "空页"}).json()["id"]
    page_id = client.post(f"/api/projects/{pid}/pages", json={"name": "无图"}).json()["id"]
    assert client.post(f"/api/projects/{pid}/pages/{page_id}/lettering/detect").status_code == 400
    assert client.post(f"/api/projects/{pid}/pages/{page_id}/lettering/see").status_code == 400


def test_oversized_image_is_rejected(client):
    pid = client.post("/api/projects", json={"name": "大图"}).json()["id"]
    res = client.post(
        f"/api/projects/{pid}/import",
        files=[("files", ("huge.png", _png(9000, 80), "image/png"))],
    )
    assert res.status_code == 400
    assert client.get(f"/api/projects/{pid}").json()["pages"] == []


def test_region_box_is_clamped(client):
    pid = client.post("/api/projects", json={"name": "框界"}).json()["id"]
    page_id = client.post(f"/api/projects/{pid}/pages", json={"name": "空"}).json()["id"]
    created = client.post(
        f"/api/projects/{pid}/pages/{page_id}/regions",
        json={"kind": "dialogue", "x": -80, "y": -40, "w": 5, "h": 5},
    ).json()
    assert created["x"] >= 0
    assert created["y"] >= 0
    assert created["w"] >= 24
    assert created["h"] >= 24
    moved = client.patch(
        f"/api/projects/{pid}/pages/{page_id}/regions/{created['id']}",
        json={"x": -10, "w": 99999},
    ).json()
    assert moved["x"] >= 0
    assert moved["w"] <= client.get(f"/api/projects/{pid}").json()["pages"][0]["width"]


def test_merge_one_region_is_400(client):
    pid = client.post("/api/projects", json={"name": "合框"}).json()["id"]
    page_id = client.post(f"/api/projects/{pid}/pages", json={"name": "空"}).json()["id"]
    one = client.post(
        f"/api/projects/{pid}/pages/{page_id}/regions",
        json={"kind": "dialogue", "x": 10, "y": 10, "w": 80, "h": 80},
    ).json()
    res = client.post(f"/api/projects/{pid}/pages/{page_id}/regions/merge", json={"region_ids": [one["id"]]})
    assert res.status_code == 400


def test_js_escapes_page_names(client):
    js = client.get("/static/app.js").text
    assert "escapeHtml(item.name)" in js
    assert "escapeHtml(region.speaker" in js
    assert "escapeHtml(region.text" in js
    assert "refreshUndo" in js
