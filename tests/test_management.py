from __future__ import annotations

import re

from tests.helpers import lettering_fixture_png


def _import(client, pid: str, *names: str):
    files = [("files", (name, lettering_fixture_png(), "image/png")) for name in names]
    res = client.post(f"/api/projects/{pid}/import", files=files)
    assert res.status_code == 200, res.text
    return res.json()


def test_managed_surfaces_have_add_and_remove(client):
    html = client.get("/").text
    js = client.get("/static/app.js").text
    css = client.get("/static/app.css").text
    surfaces = {
        "剧本": ("btnNewProject", "btnDeleteProject"),
        "成页": ("btnPageAdd", "btnPageAddFolder", "btnPageClear"),
        "文字框": ("btnLetterAddBox", "btnRegionsClear", "btnLetterDelBox"),
        "场景卡": ("btnSceneClear",),
        "情节": ("btnStoryClear",),
        "编剧Key": ("btnClearApiKey",),
    }
    for name, ids in surfaces.items():
        for token in ids:
            assert f'id="{token}"' in html, f"{name} 少了 {token}"
            assert f'$("{token}")' in js, f"{name} 的 {token} 没绑上"

    templates = re.findall(r"card\.innerHTML = `([^`]*)`", js)
    assert len(templates) >= 2
    for tpl in templates:
        assert 'data-del="' in tpl
        assert "删" in tpl
    assert "letter-box-del" in js
    assert "letter-box-del" in css
    assert "再导入" in js
    assert "webkitGetAsEntry" in js
    assert "walkEntry" in js
    assert "mergePacked" in js
    assert "if (entry) entries.push(entry)" in js
    assert "filesFromFileList" in js
    assert "还没有框。点「检出」或「手加」" in js


def test_every_html_button_is_wired(client):
    html = client.get("/").text
    js = client.get("/static/app.js").text
    ids = re.findall(r"<button\b[^>]*\bid=\"([^\"]+)\"", html)
    assert len(ids) >= 20
    missing = [item for item in ids if f'$("{item}")' not in js]
    assert missing == [], f"按钮写了但没绑: {missing}"


def test_import_sorts_nested_folder_naturally(client):
    from yizi.app import import_sort_key

    assert import_sort_key("vol/page2.png") < import_sort_key("vol/page10.png")
    pid = client.post("/api/projects", json={"name": "夹层"}).json()["id"]
    imported = client.post(
        f"/api/projects/{pid}/import",
        files=[
            ("files", ("vol/page10.png", lettering_fixture_png(), "image/png")),
            ("files", ("vol/page2.png", lettering_fixture_png(), "image/png")),
            ("files", ("vol/page1.png", lettering_fixture_png(), "image/png")),
        ],
    )
    assert imported.status_code == 200
    names = [page["name"] for page in imported.json()["project"]["pages"]]
    assert names == ["page1", "page2", "page10"]


def test_import_keeps_duplicate_basenames(client):
    pid = client.post("/api/projects", json={"name": "同名"}).json()["id"]
    imported = client.post(
        f"/api/projects/{pid}/import",
        files=[
            ("files", ("page.png", lettering_fixture_png(), "image/png")),
            ("files", ("page.png", lettering_fixture_png(), "image/png")),
        ],
    )
    assert imported.status_code == 200
    assert len(imported.json()["added"]) == 2
    assert len(imported.json()["project"]["pages"]) == 2


def test_missing_ids_are_404(client):
    pid = client.post("/api/projects", json={"name": "空"}).json()["id"]
    assert client.get("/api/projects/pdoesnotexist").status_code == 404
    assert client.delete("/api/projects/pdoesnotexist/pages/gnope").status_code == 404
    assert client.delete(f"/api/projects/{pid}/pages/gnope").status_code == 404
    assert client.delete(f"/api/projects/{pid}/pages/gnope/regions/rnope").status_code == 404
    assert client.get(f"/api/projects/{pid}/pages/gnope/source.png").status_code == 404
    empty = client.post(f"/api/projects/{pid}/pages", json={"name": "无图"}).json()
    assert client.get(f"/api/projects/{pid}/pages/{empty['id']}/source.png").status_code == 404
    assert client.post(f"/api/projects/{pid}/lettering/write", json={"scope": "page"}).status_code == 400
    assert client.post(f"/api/projects/{pid}/lettering/write", json={"scope": "region", "page_id": empty["id"]}).status_code == 400


def test_delete_project_drops_files_and_keeps_others(client):
    from yizi.store import project_dir

    keep = client.post("/api/projects", json={"name": "留下"}).json()
    gone = client.post("/api/projects", json={"name": "丢掉"}).json()
    _import(client, gone["id"], "a.png")
    root = project_dir(gone["id"])
    assert (root / "project.json").exists()
    assert client.delete(f"/api/projects/{gone['id']}").status_code == 200
    assert client.get(f"/api/projects/{gone['id']}").status_code == 404
    assert not root.exists()
    names = {item["name"] for item in client.get("/api/projects").json()}
    assert "丢掉" not in names
    assert "留下" in names
    assert client.get(f"/api/projects/{keep['id']}").status_code == 200


def test_delete_one_page_keeps_the_rest(client):
    from yizi.store import page_dir

    pid = client.post("/api/projects", json={"name": "两页"}).json()["id"]
    imported = _import(client, pid, "E01P01.png", "E01P02.png")
    first, second = imported["project"]["pages"]
    first_dir = page_dir(pid, first["id"])
    assert (first_dir / "source.png").exists()
    left = client.delete(f"/api/projects/{pid}/pages/{first['id']}")
    assert left.status_code == 200
    ids = [page["id"] for page in left.json()["pages"]]
    assert ids == [second["id"]]
    assert not first_dir.exists()
    assert client.get(f"/api/projects/{pid}/pages/{first['id']}/source.png").status_code == 404
    assert client.get(f"/api/projects/{pid}/pages/{second['id']}/source.png").status_code == 200
    listed = client.get("/api/projects").json()
    row = next(item for item in listed if item["id"] == pid)
    assert row["pages"] == 1


def test_clear_all_pages_then_import_again(client):
    pid = client.post("/api/projects", json={"name": "清空再导"}).json()["id"]
    imported = _import(client, pid, "a.png", "b.png")
    assert len(imported["project"]["pages"]) == 2
    for page in imported["project"]["pages"]:
        gone = client.delete(f"/api/projects/{pid}/pages/{page['id']}")
        assert gone.status_code == 200
    empty = client.get(f"/api/projects/{pid}").json()
    assert empty["pages"] == []
    assert empty["id"] == pid
    again = _import(client, pid, "c.png")
    assert len(again["project"]["pages"]) == 1
    assert again["project"]["pages"][0]["name"] == "c"


def test_clear_story_scene_and_regions(client):
    pid = client.post("/api/projects", json={"name": "清空字段"}).json()["id"]
    page_id = _import(client, pid, "page.png")["project"]["pages"][0]["id"]
    client.patch(
        f"/api/projects/{pid}",
        json={"premise": "港口认桩", "cast": "甲", "tone": "抬杠", "style": "短句", "expanded": "扩写", "outline": "大纲"},
    )
    client.put(
        f"/api/projects/{pid}/pages/{page_id}/scene",
        json={"where": "码头", "who": "两人", "doing": "指桩", "notes": "上格", "source": "human"},
    )
    client.patch(f"/api/projects/{pid}/pages/{page_id}", json={"page_summary": "认系船柱"})
    detected = client.post(f"/api/projects/{pid}/pages/{page_id}/lettering/detect", json={"replace": True}).json()
    assert detected["page"]["regions"]

    cleared = client.patch(
        f"/api/projects/{pid}",
        json={"premise": "", "cast": "", "tone": "", "style": "", "expanded": "", "outline": ""},
    ).json()
    assert cleared["premise"] == ""
    assert cleared["outline"] == ""
    assert cleared["pages"][0]["id"] == page_id

    scene = client.put(
        f"/api/projects/{pid}/pages/{page_id}/scene",
        json={"where": "", "who": "", "doing": "", "mood": "", "jump_from_prev": "", "notes": "", "intimacy": "none", "source": "cv"},
    ).json()
    assert scene["where"] == ""
    assert scene["who"] == ""
    blank = client.patch(f"/api/projects/{pid}/pages/{page_id}", json={"page_summary": ""}).json()
    assert blank["page_summary"] == ""

    for region in detected["page"]["regions"]:
        res = client.delete(f"/api/projects/{pid}/pages/{page_id}/regions/{region['id']}")
        assert res.status_code == 200
    page = client.get(f"/api/projects/{pid}").json()["pages"][0]
    assert page["regions"] == []
    assert page["scene"]["where"] == ""


def test_see_does_not_write_or_erase_lines(client):
    pid = client.post("/api/projects", json={"name": "看图不写词"}).json()["id"]
    page_id = _import(client, pid, "page.png")["project"]["pages"][0]["id"]
    detected = client.post(f"/api/projects/{pid}/pages/{page_id}/lettering/detect", json={"replace": True}).json()
    before = [(item["id"], item.get("text") or "") for item in detected["page"]["regions"]]
    seen = client.post(f"/api/projects/{pid}/pages/{page_id}/lettering/see")
    assert seen.status_code == 200
    after = [(item["id"], item.get("text") or "") for item in seen.json()["page"]["regions"]]
    assert after == before
    assert seen.json()["scene"]["where"]
    assert seen.json()["page"].get("text_changed") in {None, False}


def test_settings_omit_key_keeps_it(client):
    client.put("/api/settings", json={"api_key": "sk-keep-lettering"})
    kept = client.put("/api/settings", json={"writer_model": "deepseek-chat"}).json()
    assert kept["api_key_set"] is True
    assert "keep" not in (kept.get("api_key_masked") or "")
    cleared = client.put("/api/settings", json={"api_key": ""}).json()
    assert cleared["api_key_set"] is False


def test_delete_missing_project_is_idempotent(client):
    assert client.delete("/api/projects/pdoesnotexist").status_code == 200
    assert client.get("/api/projects/pdoesnotexist").status_code == 404
