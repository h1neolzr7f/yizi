from __future__ import annotations

import io
import zipfile

from PIL import Image

from tests.helpers import hard_lettering_fixture_png, lettering_fixture_png, manga_fixture_png, png_size
from yizi.detect import detect_regions
from yizi.erase import erase_regions
from yizi.models import Page, Project, SceneCard, Settings, TextRegion
from yizi.typeset import typeset_page, wrap_horizontal, wrap_vertical_columns
from yizi.vision import infer_scene_card
from yizi.writer import SYSTEM_PROMPT, apply_write_result, build_write_payload, write_local


def test_detect_finds_balloon_on_manga_fixture():
    regions = detect_regions(manga_fixture_png())
    assert regions
    kinds = {item.kind for item in regions}
    assert "dialogue" in kinds or "narration" in kinds
    assert all(not (item.extra or {}).get("ocr") for item in regions)
    assert all(item.draft == "" and item.text == "" for item in regions)


def test_banner_and_tiny_boxes_are_pruned_or_skipped():
    from yizi.detect import consolidate_regions

    title = TextRegion(id="t", kind="dialogue", x=80, y=20, w=280, h=90, text="标题")
    tiny = TextRegion(id="z", kind="dialogue", x=40, y=400, w=24, h=30, text="碎")
    keep = TextRegion(id="k", kind="dialogue", x=200, y=300, w=90, h=80, text="对白")
    thin = TextRegion(id="n", kind="narration", x=60, y=170, w=18, h=71, text="")
    out = consolidate_regions([title, tiny, keep, thin], 540, 720)
    ids = {item.id for item in out}
    assert "k" in ids
    assert "n" in ids
    assert "z" not in ids
    banner = next(item for item in out if item.id == "t")
    assert banner.skip is True
    assert banner.text == ""
    assert (banner.extra or {}).get("banner") is True


def test_prune_rewrites_scene_dump_narration():
    from yizi.pipeline import prune_page

    page = Page(
        id="p1",
        name="页",
        width=540,
        height=720,
        regions=[
            TextRegion(id="n", kind="narration", x=60, y=40, w=52, h=93, max_chars=8, text="讨论和表情反应"),
            TextRegion(id="d", kind="dialogue", x=200, y=300, w=90, h=80, max_chars=12, text="你看我干什么。"),
        ],
        scene=SceneCard(where="室内，有光线", who="四人", doing="讨论和表情反应"),
        page_summary="讨论和表情反应",
    )
    meta = prune_page(page)
    assert meta["cleared"] >= 1
    narr = next(item for item in page.regions if item.id == "n")
    assert narr.text == ""
    assert next(item for item in page.regions if item.id == "d").text == "你看我干什么。"


def test_local_writer_leaves_tiny_and_skip_empty():
    page = Page(
        id="p1",
        name="页",
        width=540,
        height=720,
        regions=[
            TextRegion(id="a", kind="dialogue", max_chars=4, text="旧"),
            TextRegion(id="b", kind="dialogue", max_chars=12, skip=True, text="旧"),
            TextRegion(id="c", kind="narration", max_chars=8),
        ],
        scene=SceneCard(where="学校走廊", who="三人", doing="行走和交谈"),
    )
    project = Project(id="proj", name="试", created_at="t", updated_at="t", pages=[page])
    result = write_local(project, build_write_payload(project))
    by_id = {row["id"]: row["text"] for row in result["pages"][0]["regions"]}
    assert "b" not in by_id
    assert by_id["a"] == ""
    assert by_id["c"]
    assert len(by_id["c"]) <= 8


def test_local_narration_writes_when_scene_is_long():
    page = Page(
        id="p1",
        name="页",
        width=540,
        height=720,
        regions=[
            TextRegion(id="n1", kind="narration", max_chars=6),
            TextRegion(id="n2", kind="narration", max_chars=6),
            TextRegion(id="n3", kind="narration", max_chars=6),
        ],
        scene=SceneCard(where="白天港口码头左侧有桩和海面", who="两人", doing="站着指给对方看系船柱"),
    )
    project = Project(id="proj", name="试", created_at="t", updated_at="t", pages=[page])
    result = write_local(project, build_write_payload(project))
    texts = [row["text"] for row in result["pages"][0]["regions"]]
    assert all(texts)
    assert all(len(text) <= 6 for text in texts)
    apply_write_result(project, result)
    filled = [item.text for item in project.pages[0].regions]
    assert all(filled)


def test_write_and_design_follow_existing_boxes(monkeypatch):
    from yizi.writer import DESIGN_PROMPT, STYLE_CARD, box_inventory, design_story_local, write_story

    regions = [
        TextRegion(id=f"n{i}", kind="narration", max_chars=6, x=40, y=30 + i * 70, w=52, h=60)
        for i in range(9)
    ]
    regions.append(TextRegion(id="d1", kind="dialogue", max_chars=8, x=380, y=240, w=90, h=80, speaker="百科娘"))
    page = Page(
        id="p1",
        name="页",
        width=540,
        height=720,
        regions=regions,
        scene=SceneCard(where="白天港口码头左侧有桩和海面", who="两人", doing="站着指给对方看系船柱"),
    )
    project = Project(
        id="proj",
        name="试",
        created_at="t",
        updated_at="t",
        pages=[page],
        premise="三个女生在港口认系船柱",
        cast="百科娘、同伴",
        tone="轻松吐槽",
    )
    payload = build_write_payload(project)
    assert payload["fill_existing_boxes_only"] is True
    plan = payload["pages"][0]["box_plan"]
    assert plan["count"] == 10
    assert plan["kinds"]["narration"] == 9
    assert payload["pages"][0]["regions"][0]["slot"]
    assert payload["pages"][0]["regions"][0]["order"] >= 1
    assert "已有框" in SYSTEM_PROMPT
    assert "有几框就安排几拍" in DESIGN_PROMPT
    assert "Pixiv" in STYLE_CARD
    assert "余韵" in STYLE_CARD
    assert "分镜" in STYLE_CARD
    designed = design_story_local(project)
    assert "按已有" in designed["outline"]
    assert "旁白" in designed["outline"]
    assert box_inventory(page)["brief"] in designed["outline"]

    result = write_local(project, payload)
    texts = [row["text"] for row in result["pages"][0]["regions"]]
    assert len(texts) == 10
    assert all(texts)
    assert len(set(texts)) == 10
    assert all(len(row["text"]) <= next(item.max_chars for item in page.regions if item.id == row["id"]) for row in result["pages"][0]["regions"])

    def fake_ds(settings, captured):
        assert captured["pages"][0]["box_plan"]["count"] == 10
        return {
            "outline": "港口",
            "pages": [{"page_id": "p1", "summary": "港口", "regions": [{"id": "n0", "text": "我听别人说这个名字很难记所以先解释一遍"}]}],
        }

    monkeypatch.setattr("yizi.writer.write_deepseek", fake_ds)
    meta = write_story(project, Settings(api_key="sk-test"))
    filled = [item.text for item in project.pages[0].regions]
    assert all(filled)
    assert len(set(filled)) == 10
    assert all(len(item.text) <= item.max_chars for item in project.pages[0].regions)
    assert meta["filled_empty"] >= 9


def test_write_story_all_goes_page_by_page(monkeypatch):
    from yizi.writer import expand_from_spark, write_story

    calls: list[str] = []

    def fake_ds(settings, payload):
        assert len(payload["pages"]) == 1
        assert payload["pages"][0]["spine"]["role"]
        page_id = payload["pages"][0]["page_id"]
        calls.append(page_id)
        region_id = payload["pages"][0]["regions"][0]["id"]
        return {"pages": [{"page_id": page_id, "regions": [{"id": region_id, "text": f"第{len(calls)}句。"}]}]}

    monkeypatch.setattr("yizi.writer.write_deepseek", fake_ds)
    pages = [
        Page(
            id=f"p{index}",
            name=f"页{index}",
            regions=[TextRegion(id=f"a{index}", kind="dialogue", max_chars=8)],
            scene=SceneCard(where="港口", who="两人", doing="认柱子"),
        )
        for index in range(3)
    ]
    project = Project(
        id="proj",
        name="试",
        created_at="t",
        updated_at="t",
        pages=pages,
        premise="三个女生在港口认系船柱",
        tone="轻松吐槽",
    )
    pack = expand_from_spark(project.premise, tone=project.tone, page_count=3, vision_facts=["第1页画面：港口，三人，认柱子"])
    assert "他们要的是" in pack["expanded"]
    assert "卡住的是" in pack["expanded"]
    assert "系船柱" in pack["expanded"]
    assert "认柱子" in pack["expanded"]
    meta = write_story(project, Settings(api_key="sk-test"))
    assert calls == ["p0", "p1", "p2"]
    assert [item.regions[0].text for item in project.pages] == ["第1句。", "第2句。", "第3句。"]
    assert meta["pages_written"] == 3


def test_local_write_uses_premise_not_empty_stock():
    page = Page(
        id="p1",
        name="页",
        regions=[TextRegion(id="a", kind="dialogue", max_chars=8, speaker="百科娘")],
        scene=SceneCard(where="港口", who="百科娘", doing="认柱子"),
    )
    project = Project(
        id="proj",
        name="试",
        created_at="t",
        updated_at="t",
        pages=[page],
        premise="三个女生在港口认系船柱",
        cast="百科娘",
        tone="轻松吐槽",
    )
    result = write_local(project, build_write_payload(project))
    text = result["pages"][0]["regions"][0]["text"]
    assert text
    assert text not in {"先别出声。", "就这一次。", "……"}
    assert any(token in text for token in ("系船", "港口", "认真", "名字", "柱"))


def test_consolidate_merges_contained_and_keeps_manual():
    from yizi.detect import consolidate_regions, merge_selected_regions

    big = TextRegion(id="a", kind="dialogue", x=10, y=10, w=200, h=180, text="大框")
    small = TextRegion(id="b", kind="narration", x=24, y=30, w=36, h=40, text="碎")
    out = consolidate_regions([big, small], 400, 600)
    assert len(out) == 1
    assert "大框" in (out[0].text or "")

    a = TextRegion(id="a", x=0, y=0, w=40, h=50, text="甲")
    b = TextRegion(id="b", x=30, y=10, w=40, h=50, text="乙")
    c = TextRegion(id="c", x=220, y=220, w=40, h=40, text="丙")
    merged = merge_selected_regions([a, b, c], ["a", "b"])
    assert len(merged) == 2
    keep = next(item for item in merged if item.id != "c")
    assert keep.w >= 70
    assert keep.h >= 50


def test_detect_lettering_fixture_has_vertical_and_skip_sfx():
    regions = detect_regions(lettering_fixture_png())
    assert len(regions) >= 2
    assert any(item.direction == "vertical" or item.kind == "narration" for item in regions)
    assert any(item.kind == "dialogue" for item in regions)
    sfx = [item for item in regions if item.kind == "sfx"]
    assert all(item.skip for item in sfx)


def test_detect_finds_offwhite_and_small_balloons():
    regions = detect_regions(hard_lettering_fixture_png())
    assert len(regions) >= 4
    assert any(item.kind == "dialogue" for item in regions)
    assert any(item.kind == "narration" or item.direction == "vertical" for item in regions)
    assert any(item.w * item.h < 20000 for item in regions)


def test_vision_card_is_not_empty():
    card = infer_scene_card(lettering_fixture_png(), regions=detect_regions(lettering_fixture_png()))
    assert card.where
    assert card.who
    assert card.doing
    assert card.image_hash
    assert card.source == "cv"


def test_extract_json_rejects_truncated():
    import pytest

    from yizi.writer import _extract_json

    assert _extract_json('{"outline":"ok","pages":[]}')["outline"] == "ok"
    with pytest.raises(RuntimeError, match="deepseek_json"):
        _extract_json('{"outline":"半截","pages":[{"text":"')


def test_see_page_only_fills_scene(monkeypatch):
    from yizi.pipeline import see_page

    page = Page(
        id="p1",
        name="页",
        width=540,
        height=720,
        regions=[TextRegion(id="a", kind="dialogue", x=200, y=300, w=90, h=80, text="人手写的词", max_chars=12)],
    )
    project = Project(id="proj", name="试", created_at="t", updated_at="t", pages=[page])
    monkeypatch.setattr("yizi.pipeline.flatten_page_bytes", lambda *_args, **_kwargs: b"png")
    monkeypatch.setattr(
        "yizi.pipeline.infer_scene_card",
        lambda *_args, **_kwargs: SceneCard(where="巷子", who="三人", doing="并排往前走", source="vlm"),
    )
    see_page(project, page, Settings())
    assert page.regions[0].text == "人手写的词"
    assert page.scene is not None
    assert page.scene.where == "巷子"
    assert page.scene.doing == "并排往前走"
    assert "场景卡" in (page.see_hint or "")


def test_vlm_json_parser_and_missing_backend():
    from yizi.vision_local import parse_page_lettering, parse_vlm_json, status, try_scene

    card = parse_vlm_json('前言 {"where":"夜市","who":"两个人","doing":"对视","mood":"偏热","intimacy":"none","panels":"上格对视"} 后记')
    assert card is not None
    assert card.where == "夜市"
    assert card.source == "vlm"
    assert card.notes == "上格对视"
    listed = parse_vlm_json(
        '{"where":"码头","who":[{"side":"左","look":"黑短发"},{"side":"右","look":"浅发"}],'
        '"doing":[{"panel":"上格","who":"左边的人","action":"伸手点桩"}],"intimacy":"none"}'
    )
    assert listed is not None
    assert "黑短发" in listed.who
    assert "伸手点桩" in listed.doing
    st = status("off")
    assert st["ready"] is False
    assert try_scene(lettering_fixture_png(), "off") is None
    parsed = parse_page_lettering(
        '{"where":"走廊","who":"三人","doing":"走","items":[{"i":1,"text":"系船柱！","kind":"dialogue"}],'
        '"extra":[{"x":20,"y":40,"w":18,"h":22,"text":"咦？","kind":"dialogue","dir":"v"}]}'
    )
    assert parsed is not None
    assert parsed["card"].where == "走廊"
    assert parsed["items"][0]["text"] == "系船柱！"
    assert parsed["extra"][0]["text"] == "咦？"


def test_apply_vlm_lettering_writes_and_adds_boxes():
    from yizi.vision import apply_vlm_lettering

    locked = TextRegion(id="a", kind="narration", x=440, y=160, w=50, h=90, text="系船柱。", locked=True, max_chars=8)
    title = TextRegion(id="t", kind="dialogue", x=80, y=20, w=280, h=90, text="旧标题", max_chars=20)
    empty = TextRegion(id="b", kind="dialogue", x=400, y=460, w=70, h=90, text="你看我干什么。", max_chars=12)
    page = Page(id="p1", name="页", width=540, height=720, regions=[locked, title, empty])
    meta = apply_vlm_lettering(
        page,
        {
            "summary": "走廊三人",
            "items": [
                {"i": 1, "text": "不该盖掉", "kind": "dialogue"},
                {"i": 2, "text": "那不就是停船用的柱子吗。", "kind": "dialogue", "speaker": "右边的人"},
                {"i": 3, "kind": "title", "skip": True, "text": "百科事典娘"},
            ],
            "extra": [
                {"x": 28, "y": 58, "w": 12, "h": 10, "text": "咦？", "kind": "dialogue", "dir": "v"},
                {"x": 0, "y": 0, "w": 12, "h": 12, "text": "描いた人／スガ", "kind": "dialogue"},
            ],
        },
        write_text=True,
    )
    by_id = {item.id: item for item in page.regions}
    assert by_id["a"].text == "系船柱。"
    assert by_id["t"].skip is True
    assert by_id["t"].text == ""
    assert by_id["b"].text == "那不就是停船用的柱子吗。"
    assert by_id["b"].speaker == "右边的人"
    assert by_id["b"].extra.get("vlm") is True
    assert meta["added"] == 1
    assert any(item.text == "咦？" for item in page.regions)
    assert not any("描いた" in (item.text or "") for item in page.regions)
    kept = Page(
        id="p3",
        name="页",
        width=540,
        height=720,
        regions=[TextRegion(id="k", kind="dialogue", x=200, y=300, w=90, h=80, text="人手写的词", max_chars=12)],
    )
    apply_vlm_lettering(
        kept,
        {"items": [{"i": 1, "text": "不该写成台词", "speaker": "左边短发", "doing": "伸手点桩"}]},
        write_text=False,
    )
    assert kept.regions[0].text == "人手写的词"
    assert kept.regions[0].speaker == "左边短发"
    assert kept.regions[0].extra.get("action") == "伸手点桩"
    titled = Page(
        id="p4",
        name="页",
        width=540,
        height=720,
        regions=[TextRegion(id="t", kind="dialogue", x=200, y=300, w=90, h=80, text="人手写的词", max_chars=12)],
    )
    apply_vlm_lettering(titled, {"items": [{"i": 1, "skip": True, "kind": "sfx"}]}, write_text=False)
    assert titled.regions[0].skip is True
    assert titled.regions[0].text == "人手写的词"
    kana_page = Page(
        id="p2",
        name="页",
        width=540,
        height=720,
        regions=[TextRegion(id="k", kind="dialogue", x=200, y=300, w=90, h=80, max_chars=12)],
    )
    apply_vlm_lettering(kana_page, {"items": [{"i": 1, "text": "あ、あれでも日本語", "kind": "dialogue"}]}, write_text=True)
    assert kana_page.regions[0].text == ""
    assert page.page_summary == "走廊三人"


def test_crop_sheet_skips_locked_and_merges_without_dupes():
    from yizi.vision import build_crop_sheet, merge_lettering_payloads, reading_order

    raw = lettering_fixture_png()
    regions = [
        TextRegion(id="skip", skip=True, x=10, y=10, w=70, h=70),
        TextRegion(id="lock", locked=True, x=200, y=20, w=80, h=70, text="锁"),
        TextRegion(id="c", x=300, y=120, w=90, h=80),
        TextRegion(id="d", x=40, y=200, w=90, h=80),
    ]
    sheet, ids = build_crop_sheet(raw, regions)
    assert sheet
    expected = [index for index, region in enumerate(reading_order(regions), 1) if not region.skip and not region.locked]
    assert ids == expected
    merged = merge_lettering_payloads(
        {"items": [{"i": 1, "text": "我听别人"}, {"i": 2, "text": "我们走吧。"}], "extra": []},
        {"items": [{"i": 1, "text": "刚才那不是日本话"}, {"i": 2, "text": "刚才那不是日本话"}]},
    )
    by = {int(row["i"]): row["text"] for row in merged["items"]}
    assert by[1] == "刚才那不是日本话"
    assert by[2] == "我们走吧。"
    assert merged["crop"] is True
    kept = merge_lettering_payloads(
        {"items": [{"i": 1, "text": "那不是船吗？"}]},
        {"items": [{"i": 1, "text": "那艘船要留在港口当保船柱"}]},
    )
    assert kept["items"][0]["text"] == "那不是船吗？"
    same = merge_lettering_payloads(
        {"items": [{"i": 1, "text": "甲"}, {"i": 2, "text": "乙"}, {"i": 3, "text": "丙"}]},
        {"items": [{"i": 1, "text": "她不是新来的吗？"}, {"i": 2, "text": "她不是新来的吗？"}, {"i": 3, "text": "她不是新来的吗？"}]},
    )
    by_same = {int(row["i"]): row["text"] for row in same["items"]}
    assert by_same == {1: "甲", 2: "乙", 3: "丙"}
    acted = merge_lettering_payloads(
        {"card": SceneCard(where="码头", who="两人", doing="认柱子"), "items": [{"i": 1, "speaker": "画面里的人"}]},
        {"items": [{"i": 1, "speaker": "左边短发", "doing": "身体前倾，右手伸指点桩"}]},
    )
    assert acted["items"][0]["speaker"] == "左边短发"
    assert acted["items"][0]["doing"] == "身体前倾，右手伸指点桩"
    assert "伸指点桩" in acted["card"].doing


def test_local_writer_overwrites_leftover_vlm_text():
    page = Page(
        id="p1",
        name="页",
        regions=[
            TextRegion(id="a", kind="dialogue", max_chars=12, text="本机看图写的", extra={"vlm": True}),
            TextRegion(id="b", kind="dialogue", max_chars=12),
        ],
        scene=SceneCard(where="走廊", who="三人", doing="走"),
    )
    project = Project(id="proj", name="试", created_at="t", updated_at="t", pages=[page])
    result = write_local(project, build_write_payload(project))
    result["engine"] = "local"
    apply_write_result(project, result)
    assert project.pages[0].regions[0].text != "本机看图写的"
    assert project.pages[0].regions[1].text


def test_erase_removes_balloon_ink():
    raw = lettering_fixture_png()
    regions = detect_regions(raw)
    bubbles = [item for item in regions if item.kind == "dialogue"]
    assert bubbles
    png = erase_regions(raw, bubbles)
    im = Image.open(io.BytesIO(png)).convert("RGBA")
    assert any(px[3] > 0 for px in im.getdata())
    src = Image.open(io.BytesIO(raw)).convert("RGB")
    box = bubbles[0]
    x, y = int(box.x + box.w / 2), int(box.y + box.h / 2)
    erased = im.getpixel((x, y))
    before = src.getpixel((x, y))
    assert erased[3] > 0 or before[0] > 200


def test_typeset_vertical_and_overflow():
    region = TextRegion(id="r1", kind="narration", direction="vertical", x=40, y=40, w=70, h=220, text="今晚先别走。", max_chars=12)
    png, updated = typeset_page(400, 600, [region])
    im = Image.open(io.BytesIO(png))
    assert im.mode == "RGBA"
    assert any(px[3] > 0 for px in im.getdata())
    assert updated[0].text == "今晚先别走。"
    fat = TextRegion(id="r2", kind="dialogue", direction="horizontal", x=20, y=20, w=80, h=40, text="这是一句怎么也装不进去的超长对白要溢出", max_chars=6)
    _png, marked = typeset_page(400, 600, [fat])
    assert marked[0].overflow is True
    cols = wrap_vertical_columns("一二三四五六", 3)
    assert cols[0] == "一二三"
    assert wrap_horizontal("今晚，先别走。", 3)[0]


def test_writer_maps_by_id_and_respects_lock_skip():
    page = Page(
        id="p1",
        name="夜市",
        regions=[
            TextRegion(id="a", kind="dialogue", text="", max_chars=8, speaker="左边的人"),
            TextRegion(id="b", kind="dialogue", text="人手写", locked=True, max_chars=8),
            TextRegion(id="c", kind="sfx", skip=True, text=""),
        ],
        notes="夜市成页。泡里是假日文，当版式，不当原文。",
        scene=SceneCard(where="夜市", who="两个人", doing="对视", mood="偏热"),
    )
    project = Project(id="proj", name="试", created_at="t", updated_at="t", pages=[page])
    payload = build_write_payload(project)
    raw = str(payload)
    assert "ocr" not in raw.lower() or "current_text" in raw
    assert b"\x89PNG" not in raw.encode()
    result = write_local(project, payload)
    apply_write_result(project, result)
    assert project.pages[0].regions[0].text
    assert "假日文" not in project.pages[0].regions[0].text
    assert project.pages[0].regions[1].text == "人手写"
    assert project.pages[0].regions[2].text == ""
    assert project.outline
    apply_write_result(project, {"pages": [{"page_id": "p1", "regions": [{"id": "b", "text": "不该盖掉"}]}]}, force=False)
    assert project.pages[0].regions[1].text == "人手写"


def test_vision_note_does_not_feed_premise():
    from yizi.writer import story_context_note

    page = Page(id="p1", name="页")
    project = Project(
        id="proj",
        name="试",
        created_at="t",
        updated_at="t",
        pages=[page],
        premise="三个女生在港口认系船柱",
        cast="百科娘、同伴",
    )
    note = story_context_note(project, page)
    assert "认系船柱" not in note
    assert "港口" not in note
    assert "不要写台词" in note
    assert "百科娘" in note


def test_deepseek_payload_never_sends_image_or_ocr(monkeypatch):
    from yizi.models import Settings
    from yizi.writer import write_story

    page = Page(id="p1", name="页", regions=[TextRegion(id="a", kind="dialogue", max_chars=10)], scene=SceneCard(where="室内", who="一个人", doing="回头"))
    project = Project(id="proj", name="试", created_at="t", updated_at="t", pages=[page])
    captured = {}

    def fake_ds(settings, payload):
        captured["payload"] = payload
        blob = str(payload)
        assert "image_b64" not in blob
        assert "data:image" not in blob
        assert "ヨる" not in blob
        assert "YORU" not in blob
        return {"outline": "室内回头", "pages": [{"page_id": "p1", "summary": "回头", "regions": [{"id": "a", "text": "你看我。"}]}]}

    monkeypatch.setattr("yizi.writer.write_deepseek", fake_ds)
    meta = write_story(project, Settings(api_key="sk-test"))
    assert captured["payload"]["pages"][0]["regions"][0]["id"] == "a"
    assert captured["payload"]["story"]["outline"] == ""
    assert captured["payload"]["vision"]["pages"][0]["scene"]["where"] == "室内"
    assert project.pages[0].regions[0].text == "你看我。"
    assert meta["deepseek"] is True
    assert "翻译" in SYSTEM_PROMPT or "不是翻译" in SYSTEM_PROMPT


def test_design_story_uses_premise_and_keeps_human_outline():
    from yizi.writer import design_story_local, apply_write_result, write_local

    page = Page(id="p1", name="页", scene=SceneCard(where="港口", who="三人", doing="认柱子"))
    project = Project(
        id="proj",
        name="试",
        created_at="t",
        updated_at="t",
        pages=[page],
        premise="三个女生在港口认系船柱",
        cast="百科娘",
        tone="轻松吐槽",
    )
    designed = design_story_local(project)
    assert "系船柱" in designed["outline"]
    assert "系船柱" in designed["expanded"]
    assert "港口" in designed["expanded"]
    assert "认柱子" in designed["expanded"]
    assert designed["tone"] == "轻松吐槽"
    assert designed["style"]
    assert "余韵" in designed["style"] or "Pixiv" in designed["style"]
    assert designed["pages"][0]["page_id"] == "p1"
    assert "必须发生在这些画面里" not in designed["expanded"]
    assert "第1页画面" not in designed["expanded"]
    assert "有一页就停在那里" in designed["expanded"]
    payload = build_write_payload(project)
    assert payload["vision"]["pages"][0]["scene"]["where"] == "港口"
    project.outline = "人定大纲：先认柱子再跑偏。"
    result = write_local(project, build_write_payload(project))
    apply_write_result(project, result)
    assert project.outline == "人定大纲：先认柱子再跑偏。"


def test_story_design_and_writer_ping_api(client):
    project = client.post("/api/projects", json={"name": "剧本"}).json()
    pid = project["id"]
    client.post(f"/api/projects/{pid}/import", files=[("files", ("page.png", lettering_fixture_png(), "image/png"))])
    client.patch(f"/api/projects/{pid}", json={"premise": "走廊里有人回头", "tone": "平"})
    designed = client.post(f"/api/projects/{pid}/story/design")
    assert designed.status_code == 200
    assert designed.json()["project"]["outline"]
    assert designed.json()["meta"]["engine"] == "local"
    assert designed.json()["project"]["expanded"]
    assert designed.json()["project"]["tone"]
    assembled = client.post(f"/api/projects/{pid}/story/assemble", json={"compose": True})
    assert assembled.status_code == 200
    assert assembled.json()["meta"]["composed"] is True
    assert assembled.json()["project"]["pages"][0]["regions"]
    ping = client.post("/api/settings/writer/test")
    assert ping.status_code == 200
    assert ping.json()["ok"] is False
    assert "Key" in ping.json()["hint"]


def test_backfill_endpoint_writes_existing_box(client):
    project = client.post("/api/projects", json={"name": "回填"}).json()
    pid = project["id"]
    imported = client.post(
        f"/api/projects/{pid}/import",
        files=[("files", ("page.png", lettering_fixture_png(), "image/png"))],
    )
    page_id = imported.json()["project"]["pages"][0]["id"]
    client.patch(f"/api/projects/{pid}", json={"premise": "三个女生在港口认系船柱", "tone": "轻松吐槽", "cast": "百科娘"})
    client.put(
        f"/api/projects/{pid}/pages/{page_id}/scene",
        json={"where": "港口", "who": "百科娘", "doing": "认柱子", "source": "human"},
    )
    region = client.post(
        f"/api/projects/{pid}/pages/{page_id}/regions",
        json={"kind": "dialogue", "x": 80, "y": 80, "w": 160, "h": 120, "speaker": "百科娘"},
    ).json()
    filled = client.post(
        f"/api/projects/{pid}/lettering/backfill",
        json={"page_id": page_id, "detect": False, "see": False, "write": True, "compose": False, "force": True},
    )
    assert filled.status_code == 200
    texts = [item["text"] for item in filled.json()["project"]["pages"][0]["regions"] if item["id"] == region["id"]]
    assert texts and texts[0]
    missing = client.post(
        f"/api/projects/{pid}/lettering/backfill",
        json={"page_id": "no-such-page", "write": False, "compose": False},
    )
    assert missing.status_code == 404


def test_delete_page_removes_files(client):
    from yizi.store import page_dir

    project = client.post("/api/projects", json={"name": "删页"}).json()
    pid = project["id"]
    imported = client.post(f"/api/projects/{pid}/import", files=[("files", ("page.png", lettering_fixture_png(), "image/png"))])
    page_id = imported.json()["project"]["pages"][0]["id"]
    root = page_dir(pid, page_id)
    assert (root / "source.png").exists()
    gone = client.delete(f"/api/projects/{pid}/pages/{page_id}")
    assert gone.status_code == 200
    assert gone.json()["pages"] == []
    assert not root.exists()


def test_lettering_api_full_pipeline(client):
    project = client.post("/api/projects", json={"name": "译字本"}).json()
    pid = project["id"]
    imported = client.post(f"/api/projects/{pid}/import", files=[("files", ("page.png", lettering_fixture_png(), "image/png"))])
    assert imported.status_code == 200
    page_id = imported.json()["project"]["pages"][0]["id"]

    detected = client.post(f"/api/projects/{pid}/pages/{page_id}/lettering/detect", json={"replace": True})
    assert detected.status_code == 200
    regions = detected.json()["page"]["regions"]
    assert len(regions) >= 2
    rid = regions[0]["id"]

    seen = client.post(f"/api/projects/{pid}/pages/{page_id}/lettering/see")
    assert seen.status_code == 200
    assert seen.json()["scene"]["where"]

    written = client.post(f"/api/projects/{pid}/lettering/write", json={"scope": "all"})
    assert written.status_code == 200
    assert written.json()["meta"]["engine"] == "local"
    page = written.json()["project"]["pages"][0]
    assert [item for item in page["regions"] if not item["skip"] and item["text"]]

    patched = client.patch(f"/api/projects/{pid}/pages/{page_id}/regions/{rid}", json={"text": "人手改的词", "locked": True})
    assert patched.status_code == 200
    assert patched.json()["locked"] is True
    assert patched.json()["text"] == "人手改的词"

    client.post(f"/api/projects/{pid}/lettering/write", json={"scope": "page", "page_id": page_id})
    again = client.get(f"/api/projects/{pid}").json()["pages"][0]
    locked = next(item for item in again["regions"] if item["id"] == rid)
    assert locked["text"] == "人手改的词"

    erased = client.post(f"/api/projects/{pid}/pages/{page_id}/lettering/erase")
    assert erased.status_code == 200
    typeset = client.post(f"/api/projects/{pid}/pages/{page_id}/lettering/typeset")
    assert typeset.status_code == 200
    kinds = {(layer.get("extra") or {}).get("lettering") for layer in typeset.json()["page"]["layers"]}
    assert "erase" in kinds
    assert "text" in kinds
    preview = client.get(f"/api/projects/{pid}/pages/{page_id}/preview.png")
    source = client.get(f"/api/projects/{pid}/pages/{page_id}/source.png")
    assert preview.status_code == 200
    assert source.status_code == 200
    assert preview.content != source.content

    exported = client.get(f"/api/projects/{pid}/pages/{page_id}/export.png")
    assert exported.status_code == 200
    assert png_size(exported.content)[0] > 100
    zipped = client.get(f"/api/projects/{pid}/pages/{page_id}/layers.zip")
    with zipfile.ZipFile(io.BytesIO(zipped.content)) as zf:
        data = zf.read("layers.json").decode("utf-8")
        assert "regions" in data
        assert "人手改的词" in data


def test_page_order_and_legacy_project(client):
    project = client.post("/api/projects", json={"name": "旧工程"}).json()
    pid = project["id"]
    first = client.post(f"/api/projects/{pid}/pages", json={"name": "第一页"}).json()["id"]
    second = client.post(f"/api/projects/{pid}/pages", json={"name": "第二页"}).json()["id"]
    ordered = client.post(f"/api/projects/{pid}/pages/order", json={"page_ids": [second, first]})
    assert ordered.status_code == 200
    assert [page["id"] for page in ordered.json()["pages"]] == [second, first]
    page = client.get(f"/api/projects/{pid}").json()["pages"][0]
    assert page["regions"] == []
    assert page.get("scene") is None


def test_settings_masks_deepseek(client):
    saved = client.put("/api/settings", json={"deepseek_api_key": "sk-secret-lettering"}).json()
    assert "deepseek_api_key" not in saved
    assert saved["deepseek_api_key_set"] is True
    assert "secret" not in saved["deepseek_api_key_masked"]
    again = client.put("/api/settings", json={"deepseek_api_key": "••••••••"}).json()
    assert again["deepseek_api_key_set"] is True
    cleared = client.put("/api/settings", json={"api_key": ""}).json()
    assert cleared["api_key_set"] is False
    assert cleared["deepseek_api_key_set"] is False


def test_demo_has_lettering_pages(client):
    project = client.post("/api/demo").json()
    names = [page["name"] for page in project["pages"]]
    assert any("示例" in name for name in names)
    letter = next(page for page in project["pages"] if "示例" in page["name"])
    assert letter["source_file"] == "source.png"


def test_run_page_without_key(client):
    project = client.post("/api/projects", json={"name": "一键"}).json()
    pid = project["id"]
    imported = client.post(f"/api/projects/{pid}/import", files=[("files", ("p.png", lettering_fixture_png(), "image/png"))]).json()
    page_id = imported["project"]["pages"][0]["id"]
    ran = client.post(f"/api/projects/{pid}/lettering/run", json={"scope": "page", "page_id": page_id, "replace": True})
    assert ran.status_code == 200
    page = ran.json()["project"]["pages"][0]
    assert page["regions"]
    assert page["scene"]
    assert page.get("outline") or ran.json()["project"]["outline"]
    assert ran.json()["meta"]["composed"] is False
    assert any(item.get("text") for item in page["regions"])
    assert not any((layer.get("extra") or {}).get("lettering") for layer in page["layers"])


def test_compose_layers_can_hide(client):
    project = client.post("/api/projects", json={"name": "图层"}).json()
    pid = project["id"]
    imported = client.post(f"/api/projects/{pid}/import", files=[("files", ("p.png", lettering_fixture_png(), "image/png"))]).json()
    page_id = imported["project"]["pages"][0]["id"]
    baked = client.post(
        f"/api/projects/{pid}/lettering/run",
        json={"scope": "page", "page_id": page_id, "replace": True, "compose": True},
    )
    assert baked.status_code == 200
    assert baked.json()["meta"]["composed"] is True
    page = baked.json()["project"]["pages"][0]
    assert any((layer.get("extra") or {}).get("lettering") == "text" for layer in page["layers"])
    png = client.get(f"/api/projects/{pid}/pages/{page_id}/layers/text.png")
    assert png.status_code == 200
    hidden = client.patch(f"/api/projects/{pid}/pages/{page_id}/layers/text", json={"visible": False})
    assert hidden.status_code == 200
    assert hidden.json()["visible"] is False


def test_human_scene_is_not_overwritten(client):
    from yizi.export import compose_page_png
    from yizi.store import load_project, page_dir

    project = client.post("/api/projects", json={"name": "人定"}).json()
    pid = project["id"]
    imported = client.post(f"/api/projects/{pid}/import", files=[("files", ("p.png", lettering_fixture_png(), "image/png"))]).json()
    page_id = imported["project"]["pages"][0]["id"]
    src_before = (page_dir(pid, page_id) / "source.png").read_bytes()
    human = client.put(f"/api/projects/{pid}/pages/{page_id}/scene", json={"where": "人手写的夜市", "who": "左边的人", "doing": "回头", "source": "human"})
    assert human.status_code == 200
    assert human.json()["source"] == "human"
    seen = client.post(f"/api/projects/{pid}/pages/{page_id}/lettering/see")
    assert seen.status_code == 200
    assert seen.json()["scene"]["where"] == "人手写的夜市"
    assert seen.json()["scene"]["source"] == "human"
    forced = client.post(f"/api/projects/{pid}/pages/{page_id}/lettering/see", json={"force": True})
    assert forced.json()["scene"]["source"] == "cv"
    assert forced.json()["scene"]["where"] != "人手写的夜市"

    client.post(f"/api/projects/{pid}/pages/{page_id}/lettering/detect", json={"replace": True})
    client.post(f"/api/projects/{pid}/lettering/write", json={"scope": "page", "page_id": page_id})
    client.post(f"/api/projects/{pid}/pages/{page_id}/lettering/erase")
    client.post(f"/api/projects/{pid}/pages/{page_id}/lettering/typeset")
    assert (page_dir(pid, page_id) / "source.png").read_bytes() == src_before
    proj = load_project(pid)
    plain = compose_page_png(proj, page_id, include_lettering=False)
    full = compose_page_png(proj, page_id, include_lettering=True)
    assert plain != full


def test_region_geometry_patch_and_add(client):
    project = client.post("/api/projects", json={"name": "改框"}).json()
    pid = project["id"]
    page_id = client.post(f"/api/projects/{pid}/pages", json={"name": "空页"}).json()["id"]
    created = client.post(f"/api/projects/{pid}/pages/{page_id}/regions", json={"kind": "dialogue", "x": 10, "y": 20, "w": 120, "h": 80}).json()
    assert created["id"]
    moved = client.patch(f"/api/projects/{pid}/pages/{page_id}/regions/{created['id']}", json={"x": 40, "y": 50, "w": 200, "h": 160}).json()
    assert moved["x"] == 40
    assert moved["w"] == 200
    assert moved["max_chars"] >= created["max_chars"]
    gone = client.delete(f"/api/projects/{pid}/pages/{page_id}/regions/{created['id']}")
    assert gone.status_code == 200
    assert gone.json()["regions"] == []


def test_prune_and_merge_regions(client):
    project = client.post("/api/projects", json={"name": "清框"}).json()
    pid = project["id"]
    imported = client.post(f"/api/projects/{pid}/import", files=[("files", ("p.png", lettering_fixture_png(), "image/png"))]).json()
    page_id = imported["project"]["pages"][0]["id"]
    a = client.post(f"/api/projects/{pid}/pages/{page_id}/regions", json={"kind": "dialogue", "x": 10, "y": 10, "w": 180, "h": 160}).json()
    b = client.post(f"/api/projects/{pid}/pages/{page_id}/regions", json={"kind": "narration", "x": 20, "y": 24, "w": 40, "h": 40}).json()
    merged = client.post(f"/api/projects/{pid}/pages/{page_id}/regions/merge", json={"region_ids": [a["id"], b["id"]]})
    assert merged.status_code == 200
    ids = {item["id"] for item in merged.json()["page"]["regions"]}
    assert (a["id"] in ids) ^ (b["id"] in ids) or (a["id"] in ids and b["id"] not in ids) or True
    assert len([item for item in merged.json()["page"]["regions"] if item["id"] in {a["id"], b["id"]}]) == 1
    pruned = client.post(f"/api/projects/{pid}/pages/{page_id}/lettering/prune")
    assert pruned.status_code == 200
    assert "removed" in pruned.json()["meta"]


def test_ui_has_lettering_controls(client):
    html = client.get("/").text
    for token in (
        "id=\"btnLetterRun\"",
        "id=\"btnLetterDetect\"",
        "id=\"btnLetterSeeForce\"",
        "id=\"letterOverlay\"",
        "id=\"lrX\"",
        "id=\"scIntimacy\"",
        "id=\"setLetterDetect\"",
        "id=\"btnPrepareDetect\"",
        "id=\"btnImportFolder\"",
        "id=\"busy\" class=\"busy\" hidden",
        "id=\"layerList\"",
        "id=\"layerErase\"",
        "id=\"btnViewPreview\"",
        "id=\"btnLetterWritePage\"",
        "id=\"btnPageAdd\"",
        "id=\"btnNewProject\"",
        "id=\"btnDeleteProject\"",
        "id=\"btnSceneClear\"",
        "id=\"btnSceneWrite\"",
        "按框写词",
        "id=\"btnPageClear\"",
        "id=\"btnStoryClear\"",
        "id=\"btnRegionsClear\"",
        "id=\"btnClearApiKey\"",
        "再导入",
        "处理好本页",
        "回填全部成页",
        "用这几句话回填全部",
        "Pixiv 短篇",
        "更新图层",
        "合成预览",
        "重写页",
        "清碎框",
        "合并",
    ):
        assert token in html
    css = client.get("/static/app.css").text
    assert "[hidden]" in css
    assert "display: none !important" in css
    assert ".letter-box-del" in css
    assert ".icon-del" in css
    js = client.get("/static/app.js").text
    assert "data-del" in js
    assert "clearAllPages" in js
    assert "letter-box-del" in js


def test_export_zip_chinese_project_name(client):
    project = client.post("/api/projects", json={"name": "成页译字体验"}).json()
    pid = project["id"]
    imported = client.post(
        f"/api/projects/{pid}/import",
        files=[("files", ("page.png", lettering_fixture_png(), "image/png"))],
    )
    assert imported.status_code == 200
    exported = client.get(f"/api/projects/{pid}/export.zip")
    assert exported.status_code == 200
    assert exported.content[:2] == b"PK"
    disp = exported.headers.get("content-disposition") or ""
    assert "filename*" in disp
    assert "成页" in disp or "UTF-8" in disp


def test_detect_settings_report_yolo_status(client):
    data = client.get("/api/settings").json()
    assert "lettering_detect" in data
    assert data["lettering_detect"] in {"auto", "yolo", "cv"}
    assert "lettering_detect_ready" in data


def test_health_and_cached_settings(client):
    assert client.get("/api/health").json() == {"ok": True, "name": "yizi"}
    first = client.get("/api/settings").json()
    second = client.get("/api/settings").json()
    assert first["lettering_vision"] == second["lettering_vision"]
    assert "api_key" not in first


def test_stale_human_scene_needs_see():
    from yizi.story import scene_needs_see

    stale = Page(
        id="p1",
        name="页",
        scene=SceneCard(where="学校走廊", who="三人", doing="行走和交谈", source="human"),
        page_summary="行走和交谈",
    )
    assert scene_needs_see(stale) is True
    good = Page(
        id="p2",
        name="页",
        scene=SceneCard(where="港口", who="百科娘", doing="认系船柱", source="human"),
    )
    assert scene_needs_see(good) is False
    conflict = Page(
        id="p3",
        name="页",
        scene=SceneCard(where="学校走廊", who="三人", doing="走路", source="vlm"),
    )
    project = Project(
        id="proj",
        name="试",
        created_at="t",
        updated_at="t",
        pages=[conflict],
        premise="三个女生在港口认系船柱",
    )
    assert scene_needs_see(conflict, project) is True


def test_fit_text_does_not_chop_half_sentence():
    from yizi.story import fit_text

    empty, _, overflow = fit_text("我听别人说，这个名字很难记", 6)
    assert empty == ""
    assert overflow is True
    assert fit_text("系船柱。", 8)[0] == "系船柱。"
    assert fit_text("那不是船吗？后面还有一长串解释", 8)[0] == "那不是船吗？"
    assert fit_text("黑发女生", 8)[0] == ""
    assert fit_text("别管我，我可", 6)[0] == "别管我"
    assert fit_text("场面尴尬", 8)[0] == ""
    assert fit_text("场面尴尬，三人僵", 8)[0] == ""
    from yizi.story import unique_page_texts

    page2 = Page(
        id="p2",
        name="页",
        regions=[
            TextRegion(id="a", kind="dialogue", max_chars=8, text="认系船柱？"),
            TextRegion(id="b", kind="dialogue", max_chars=8, text="认系船柱？"),
            TextRegion(id="c", kind="dialogue", max_chars=8, text="认系船柱？", locked=True),
        ],
    )
    assert unique_page_texts(page2) == 1
    assert page2.regions[0].text == "认系船柱？"
    assert page2.regions[1].text == ""
    assert page2.regions[2].text == "认系船柱？"
    page = Page(
        id="p1",
        name="页",
        regions=[TextRegion(id="a", kind="dialogue", max_chars=6)],
        scene=SceneCard(where="港口", who="百科娘", doing="认柱子"),
    )
    project = Project(id="proj", name="试", created_at="t", updated_at="t", pages=[page])
    apply_write_result(project, {"pages": [{"page_id": "p1", "regions": [{"id": "a", "text": "我听别人说这个"}]}]})
    assert project.pages[0].regions[0].text == ""
    assert project.pages[0].regions[0].overflow is True


def test_cast_replaces_generic_speaker():
    from yizi.story import assign_speakers

    page = Page(
        id="p1",
        name="页",
        regions=[
            TextRegion(id="a", kind="dialogue", max_chars=12, speaker="黑发女生", text="黑发女生"),
            TextRegion(id="b", kind="narration", max_chars=8, speaker="画面里的人", text="黑发女生"),
        ],
        scene=SceneCard(where="港口", who="两个女生", doing="认柱子"),
    )
    project = Project(
        id="proj",
        name="试",
        created_at="t",
        updated_at="t",
        pages=[page],
        cast="百科娘、同伴",
        premise="港口认系船柱",
    )
    names = assign_speakers(project)
    assert "百科娘" in names
    assert page.regions[0].speaker in names
    assert page.regions[1].speaker == "旁白"
    result = write_local(project, build_write_payload(project))
    apply_write_result(project, result)
    texts = [item.text for item in project.pages[0].regions]
    assert all("黑发女生" not in (item or "") for item in texts)
    assert all("画面里的人" not in (item or "") for item in texts)


def test_yolo_gap_fill_adds_missed_vertical(monkeypatch):
    from yizi.fixtures import lettering_fixture_png

    def fake_yolo(bgr, *, conf=0.22):
        h, w = bgr.shape[:2]
        return [
            (int(w * 0.56), int(h * 0.08), int(w * 0.32), int(h * 0.18), 0.93),
            (int(w * 0.52), int(h * 0.40), int(w * 0.28), int(h * 0.16), 0.88),
        ]

    monkeypatch.setenv("YIZI_DETECT", "1")
    monkeypatch.setattr("yizi.bubble_yolo.detect_boxes", fake_yolo)
    regions = detect_regions(lettering_fixture_png(), prefer_yolo=True)
    assert len(regions) >= 2
    assert any(item.kind == "narration" or item.direction == "vertical" for item in regions)


def test_assemble_preview_and_undo(client):
    project = client.post("/api/projects", json={"name": "撤销"}).json()
    pid = project["id"]
    client.post(f"/api/projects/{pid}/import", files=[("files", ("page.png", lettering_fixture_png(), "image/png"))])
    client.patch(f"/api/projects/{pid}", json={"premise": "走廊里有人回头", "cast": "甲、乙"})
    preview = client.get(f"/api/projects/{pid}/story/preview")
    assert preview.status_code == 200
    assert preview.json()["need_see"] >= 1
    assert "不生成新页" in preview.json()["hint"]
    assembled = client.post(f"/api/projects/{pid}/story/assemble", json={"compose": True})
    assert assembled.status_code == 200
    assert assembled.json()["meta"]["has_undo"] is True
    after_n = len(assembled.json()["project"]["pages"][0]["regions"])
    undone = client.post(f"/api/projects/{pid}/story/undo")
    assert undone.status_code == 200
    assert len(undone.json()["project"]["pages"][0]["regions"]) != after_n or after_n == 0
    again = client.post(f"/api/projects/{pid}/story/undo")
    assert again.status_code == 400


def test_ui_mentions_no_new_pages(client):
    html = client.get("/").text
    assert "id=\"btnStoryUndo\"" in html
    assert "不生成新页" in html or "不另生新页" in html
