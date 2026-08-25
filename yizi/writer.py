from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from yizi.models import Page, Project, Settings, TextRegion
from yizi.vision import scene_as_prompt

log = logging.getLogger("yizi.writer")

# 文笔卡：吸收公开轻小说写作台的手艺（感官、表里、禁说明书），按 Pixiv 投稿口吻重写，不整段抄别人的提示词。
STYLE_CARD = """你现在的身份是 Pixiv 投稿向的轻小说/漫画剧情作者，不是分镜助理，不是翻译。
剧情说明（expanded、outline、summary、style）必须能当短篇来读。
写法：
- 短段，留白，收在余韵上。读者看完还该停一下。
- 用身体写场：风、盐、袖口、桩的凉、谁先移开视线。不要「地点／人物／动作」三栏。
- 每人有表和里：嘴上在讲什么，手里其实在躲什么。
- 对话口语、有口癖；旁白是心象，不是解说。
- 每页只写一件事，加一个没说完的感觉，钩住下一页。
禁用：
- 说明书句：「第N页画面」「对上画面」「基调是」「必须发生在这些画面里」「本页要」这种编辑备注
- AI腔：不禁、宛如、仿佛、一抹、那一刻、眼神交汇、空气仿佛凝固
- 破折号——。用句号、逗号或省略号
事实锁：地点、人数、动作听画面，一个都不能改。文笔只负责怎么说，不负责另编一场。
"""

SYSTEM_PROMPT = """你是「AI漫画编剧大师」，不是翻译。
vision.pages 是本机看图事实：地点、谁、每人在干什么、分镜。看图只出场景卡，不写台词。
story 是用户几句话扩出来的同一条情节。二者必须织在一起，禁止各写各的。
冲突时：地点、人数、动作听画面；口气和因果听 story。台词由你来写。
框里原来的假日文/火星文已经作废，不要当原文。
只返回 JSON 对象，不要 markdown。格式：
{"outline":"全书或这一组的大纲","pages":[{"page_id":"...","summary":"本页一句","regions":[{"id":"...","text":"台词","text_short":"更短备选","speaker":"说话人"}]}]}
规则：
- 先看 spine：want（他们要什么）、role（起承转合）、prev.last_lines（上一页说到哪）。这一页必须接上，不要重新开场
- 先数这一页 box_plan：有几框、各是什么、各能装几个字，再写词。有几框就写几拍，禁止按小说篇幅写再硬塞
- 只给已有 region_id 写词，不要发明新框，也不要因为「还想多写一句」而空着已有框
- 每个可写框都必须有一句完整短句，text 不超过该框 max_chars。先按框长设计，不要写长了再砍
- 阅读顺序：上到下，同行从右到左。regions[].order / slot 是版式位置
- 对白框写说话，旁白框写场面短注，内心写心里话，拟声写拟声。不要把长场面句抄进窄旁白
- 九个小旁白就写成九句能连起来的短注，不要写成一段再拆
- 每页台词必须发生在该页 scene 里，不要把别页的事写进来
- scene.doing 是画面动作，不要抄进框当台词
- locked 或 skip 的框不要改
- 图不会发给你，也不要向用户索要
- 如果场景卡 intimacy 是 explicit，用成人对白，但不要写违法内容
- 说话人必须用 story.cast 里的名字，禁止「黑发女生」「画面里的人」
- 装不下就换更短的完整句，不要半句，不要「别管我，我可」这种砍断
- 同一页每个框必须不同，禁止整页都写「认系船柱？」
- 旁白禁止写「场面尴尬」「讨论和表情反应」「行走和交谈」
- vision.pages[].trusted 为 false 时，不要用该页 scene 的地点，只听用户原话
- 禁止每页都用「先别出声」「就这一次」这种空套话；词要从 premise / want / 本页拍里长出来
- pages[].summary 用投稿小说一句，不要写成「港口，三人，认柱子」这种分镜表
"""

DESIGN_PROMPT = """你是「AI漫画编剧大师」。用户只写了几句话。
vision.pages 是本机看图结果，是事实。你要把用户的几句话织进这些画面，不能另写地点、人数、动作。
先定一条能读的情节，再按每页已有的框切开：有几框就安排几拍，框短就写短拍。不要设计需要新泡、新页才能装下的对白。
不要翻译假字。不要索要原图。不要生成图片。
只返回 JSON，不要 markdown。格式：
{"want":"他们要什么","obstacle":"卡住什么","turn":"哪一页翻面","payoff":"收回哪一句","expanded":"Pixiv短篇口吻的一段故事，140到220字，有风有人有余韵，有开头中间收束","tone":"二字到六字基调","style":"一句写法，像给投稿小说定调","cast":"谁","outline":"每行能读：第N页｜起/承/转/合｜这一页看见的一件事｜没说完的那一点｜按几框写","pages":[{"page_id":"...","role":"起","summary":"投稿小说一句，带画面和余韵，顺便点明框怎么分工"}]}
规则：
- expanded、outline、summary 都按文笔卡来写，必须是故事，不要罗列看图结果
- 每页 summary 必须能对上该页 scene.where/who/doing，但要写成能读的一句，不要三栏清单
- 后一页必须接前一页没说完的事，禁止每页重复同一句玩笑
- 看图只有场景卡，没有台词草稿。boxes / box_plan 是版式事实
- 用户如果已经给了 tone/cast，就沿用
- page_id 必须对上，不要发明新 id
- 还没看图的页标「待看图」，不要编画面
- 人物名跟 story.cast 走，不要用发色外貌当名字
- trusted 为 false 的页标「待重看」，大纲里不要写成学校走廊套话
"""


_KIND_ZH = {"dialogue": "对白", "narration": "旁白", "thought": "内心", "sfx": "拟声"}

_STOCK = {
    "narration": (
        "先别出声。",
        "就这一次。",
        "看着那边。",
        "别回头。",
        "先等等。",
        "还没完。",
        "就是这儿。",
        "别出声。",
        "先别动。",
        "就这样。",
        "先记下。",
        "有人来了。",
        "先别问。",
        "到这儿。",
        "停一下。",
        "别让人看。",
        "在这儿。",
        "先看着。",
        "夜里。",
        "白天。",
        "先走。",
        "等等。",
        "……",
        "其一。",
        "其二。",
        "其三。",
        "其四。",
        "其五。",
        "其六。",
        "其七。",
        "其八。",
        "其九。",
    ),
    "thought": (
        "别让人看出来。",
        "就这一次。",
        "……先别出声。",
        "别回头。",
        "先别问。",
        "我知道。",
        "别出声。",
        "先等等。",
    ),
    "sfx": ("碰", "嘶", "咚", "哈", "哗", "喀", "呼", "嗒"),
    "dialogue": (
        "你认真的？",
        "那是名字。",
        "不是吧。",
        "等一下。",
        "我知道了。",
        "走这边。",
        "你说什么？",
        "真的假的。",
        "我没听清。",
        "就这儿？",
        "你先说。",
        "别装了。",
        "行了。",
        "看着我。",
        "靠近一点。",
        "……我知道。",
        "你看我干什么。",
        "别吵。",
        "先别问。",
        "那不就是名字吗。",
    ),
}


def box_slot(region: TextRegion, page: Page) -> str:
    cx = (region.x + region.w / 2) / max(page.width, 1)
    cy = (region.y + region.h / 2) / max(page.height, 1)
    row = "上" if cy < 0.34 else ("下" if cy > 0.66 else "中")
    col = "右" if cx > 0.55 else ("左" if cx < 0.45 else "中")
    return f"{row}{col}"


def ordered_regions(page: Page, regions: list[TextRegion] | None = None) -> list[TextRegion]:
    items = list(regions if regions is not None else page.regions)
    return sorted(
        items,
        key=lambda region: (
            int(((region.y + region.h / 2) / max(page.height, 1)) * 6),
            -(region.x + region.w / 2),
            region.id,
        ),
    )


def box_inventory(page: Page) -> dict[str, Any]:
    writable = [region for region in page.regions if not region.skip]
    kinds: dict[str, int] = {}
    for region in writable:
        kinds[region.kind] = kinds.get(region.kind, 0) + 1
    parts = [f"{kinds[kind]}个{_KIND_ZH[kind]}" for kind in ("dialogue", "narration", "thought", "sfx") if kinds.get(kind)]
    open_boxes = [region for region in writable if not region.locked and region.max_chars >= 5]
    return {
        "count": len(writable),
        "locked": sum(1 for region in writable if region.locked),
        "open": len(open_boxes),
        "kinds": kinds,
        "brief": "、".join(parts) if parts else "没有可写框",
        "hard_max": min((region.max_chars for region in open_boxes), default=0),
    }


_ARC = (
    ("起", "把这件事抛到台面上"),
    ("承", "有人接话，认真和抬杠分开"),
    ("转", "解释越认真越歪，或者忽然近一层"),
    ("合", "收回开头那件事，留一句没说完"),
)

_NOUN_STOP = {
    "三个",
    "几个",
    "我们",
    "他们",
    "她们",
    "自己",
    "什么",
    "这个",
    "那个",
    "不是",
    "没有",
    "已经",
    "还是",
    "一个",
    "女生",
    "男生",
    "有人",
    "基调",
    "画面",
    "对白",
    "旁白",
    "在场",
}

_META_MARKERS = ("基调是", "对上画面", "必须发生", "他们要的是", "卡住的是", "按已有", "本页要")


def page_arc(index: int, count: int) -> tuple[str, str]:
    if count <= 1:
        return "合", "在这一页把事说清楚，留一句"
    if index <= 0:
        return _ARC[0]
    if index >= count - 1:
        return _ARC[3]
    inner = count - 2
    pos = (index - 1) / max(inner - 1, 1) if inner > 1 else 0
    return _ARC[1] if pos < 0.5 else _ARC[2]


def _story_nouns(text: str) -> list[str]:
    raw = text or ""
    found: list[str] = []
    for match in re.finditer(r"[认看找叫讲说]([\u4e00-\u9fff]{2,4})", raw):
        token = match.group(1)
        if token not in _NOUN_STOP and token not in found:
            found.append(token)
    for key in ("系船柱", "港口", "码头", "海边", "学校", "教室", "走廊", "夜市"):
        if key in raw and key not in found:
            found.append(key)
    return found[:6]


def _story_clauses(story: dict[str, Any], page: Page) -> list[str]:
    from yizi.story import is_dump_line, is_generic_label

    chunks = [
        story.get("premise") or "",
        story.get("expanded") or "",
        page.page_summary or "",
    ]
    if page.scene:
        chunks.extend((page.scene.where or "", page.scene.doing or ""))
    parts = re.split(r"[。！？\n；;]", "。".join(item for item in chunks if item))
    out: list[str] = []
    for part in parts:
        bit = part.strip(" ，,")
        if any(marker in bit for marker in _META_MARKERS):
            bit = bit.split("：")[-1].strip() if "：" in bit else ""
        if not bit or is_dump_line(bit) or is_generic_label(bit):
            continue
        if bit not in out:
            out.append(bit)
    return out


def _page_recap(page: Page) -> dict[str, Any]:
    lines: list[dict[str, str]] = []
    for region in ordered_regions(page):
        text = (region.text or "").strip()
        if region.skip or not text:
            continue
        lines.append({"id": region.id, "kind": region.kind, "speaker": region.speaker, "text": text})
    return {
        "page_id": page.id,
        "name": page.name,
        "summary": page.page_summary,
        "last_lines": lines[-3:],
    }


def page_spine(project: Project, page: Page) -> dict[str, Any]:
    idx = project.pages.index(page)
    role, job = page_arc(idx, len(project.pages))
    return {
        "want": _first_clause(project.premise or project.expanded or ""),
        "tone": project.tone,
        "role": role,
        "job": job,
        "prev": _page_recap(project.pages[idx - 1]) if idx else None,
        "next_scene": scene_as_prompt(project.pages[idx + 1].scene, project.pages[idx + 1].name) if idx + 1 < len(project.pages) else None,
    }


def _region_payload(region: TextRegion, page: Page | None = None, order: int = 0) -> dict[str, Any]:
    data = {
        "id": region.id,
        "kind": region.kind,
        "direction": region.direction,
        "speaker": region.speaker,
        "tone": region.tone,
        "max_chars": region.max_chars,
        "skip": region.skip,
        "locked": region.locked,
        "current_text": region.text if region.locked else "",
    }
    if page is not None:
        data["order"] = order
        data["slot"] = box_slot(region, page)
        data["w"] = int(region.w)
        data["h"] = int(region.h)
    return data


def build_write_payload(
    project: Project,
    *,
    scope: str = "all",
    page_id: str | None = None,
    region_id: str | None = None,
) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    for index, page in enumerate(project.pages):
        if scope == "page" and page_id and page.id != page_id:
            continue
        regions = page.regions
        if scope == "region" and region_id:
            regions = [r for r in regions if r.id == region_id]
            if not regions:
                continue
        writable = ordered_regions(page, [r for r in regions if not r.skip])
        pages.append(
            {
                "page_id": page.id,
                "name": page.name,
                "order": index + 1,
                "notes": page.notes,
                "summary": page.page_summary,
                "scene": scene_as_prompt(page.scene, page.name, page.notes),
                "box_plan": box_inventory(page),
                "spine": page_spine(project, page),
                "fill_existing_boxes_only": True,
                "regions": [_region_payload(region, page, order) for order, region in enumerate(writable, start=1)],
            }
        )
    focus = page_id if scope in {"page", "region"} else None
    return {
        "title": project.name,
        "story": story_brief(project),
        "vision": visual_board(project, focus_page_id=focus),
        "existing_outline": project.outline,
        "fill_existing_boxes_only": True,
        "pages": pages,
    }


def visual_board(project: Project, *, focus_page_id: str | None = None) -> dict[str, Any]:
    from yizi.story import scene_is_usable

    focus_ids: set[str] | None = None
    if focus_page_id:
        ids = [item.id for item in project.pages]
        try:
            idx = ids.index(focus_page_id)
        except ValueError:
            idx = -1
        if idx >= 0:
            focus_ids = {ids[item] for item in (idx - 1, idx, idx + 1) if 0 <= item < len(ids)}
    pages: list[dict[str, Any]] = []
    for index, page in enumerate(project.pages):
        scene = page.scene
        trusted = scene_is_usable(page, project)
        boxes = [
            _region_payload(region, page, order)
            for order, region in enumerate(ordered_regions(page, [item for item in page.regions if not item.skip]), start=1)
        ]
        if focus_ids is not None and page.id not in focus_ids:
            boxes = []
        elif len(boxes) > 12:
            boxes = boxes[:8]
        pages.append(
            {
                "page_id": page.id,
                "order": index + 1,
                "name": page.name,
                "seen": bool(scene and scene.source in {"vlm", "human"}),
                "trusted": trusted,
                "source": scene.source if scene else "",
                "summary": page.page_summary if trusted else "",
                "see_hint": page.see_hint,
                "scene": scene_as_prompt(scene, page.name, page.notes) if trusted else {"where": "待重看", "who": "", "doing": "场景卡不可信", "notes": "地点听用户原话，不要写学校走廊套话"},
                "box_plan": box_inventory(page),
                "boxes": boxes,
            }
        )
    return {
        "page_count": len(pages),
        "seen_count": sum(1 for item in pages if item["seen"]),
        "pages": pages,
    }


def vision_facts(project: Project) -> list[str]:
    from yizi.story import scene_is_usable

    facts: list[str] = []
    for index, page in enumerate(project.pages):
        scene = page.scene
        if not scene_is_usable(page, project):
            facts.append(f"第{index + 1}页待重看，地点先听用户原话，不要写学校走廊套话")
            continue
        bit = f"第{index + 1}页画面：{scene.where}，{scene.who}，{scene.doing}"
        if scene.notes:
            bit += f"。分镜：{scene.notes}"
        if page.page_summary:
            bit += f"。看图摘要：{page.page_summary}"
        facts.append(bit)
    return facts


def story_context_note(project: Project, page: Page) -> str:
    note = (
        "只认这一页画面上看得见的角色和动作，不要写台词，不要翻译泡里的字。"
        "不要用全书情节或上一页的地点来猜。图上没有的东西不要写。"
    )
    if project.cast:
        note += f"人物表仅供对名：{project.cast}。对不上就写外貌，不要按名字编动作。"
    return note


def story_brief(project: Project) -> dict[str, Any]:
    from yizi.story import has_foreign_place

    expanded = project.expanded or ""
    outline = project.outline or ""
    premise = project.premise or ""
    if has_foreign_place(expanded, premise):
        expanded = ""
    if has_foreign_place(outline, premise):
        outline = ""
    return {
        "premise": premise,
        "cast": project.cast or "",
        "tone": project.tone or "",
        "style": project.style or "",
        "expanded": expanded,
        "outline": outline,
    }


_TONE_RULES = (
    ("轻松吐槽", ("吐槽", "搞笑", "跑偏", "百科", "抬杠", "系船", "借口")),
    ("甜", ("甜", "喜欢", "心动", "告白")),
    ("热血", ("热血", "战斗", "冲", "赢")),
    ("悬疑", ("悬疑", "谜", "不对劲")),
    ("偏热", ("靠近", "呼吸", "今晚")),
)

_STYLE_LINE = {
    "轻松吐槽": "Pixiv短篇口吻：短句抬杠，认真的人先翻车，收在没说完的那一声里",
    "甜": "话说一半，袖口比称呼近，留一句没说完",
    "热血": "短促，收在动作上，余韵留给下一页",
    "悬疑": "先藏一句，下一页再漏，不要解释谜底",
    "偏热": "近、慢、少称呼，风先碰到皮肤",
    "平": "一句一事，口语，像投稿小说那样留白",
    "轻松": "口语，别旁白说教，收在余韵上",
}

_EXPAND_TAIL = {
    "轻松吐槽": "有人把名字当正经事讲，旁边的人只抬杠。盐风一过，认真的那张脸先垮。",
    "甜": "两人话越说越近，最后停在没说完的那一句，袖口还没放下。",
    "热血": "先被堵住，再顶回去，最后往前冲一步，尘还没落。",
    "悬疑": "表面在闲聊，底下有一句对不上。谁先听见，谁先不说话。",
    "偏热": "话说到一半就停。风比台词更近，有人忘了把手收回来。",
    "平": "从小切口进，中间生出岔路，最后收回开头那件事，还留一点没讲完。",
    "轻松": "正事被人带跑。笑完以后，桩还在原来的地方。",
}

_PAGE_ROLES = ("起：把事抛出来", "承：有人接话或抬杠", "转：发现不对或跑偏", "合：收回正题，留一句")


def infer_tone(text: str) -> str:
    blob = text or ""
    for tone, keys in _TONE_RULES:
        if any(key in blob for key in keys):
            return tone
    return "轻松"


def style_for(tone: str) -> str:
    for key, line in _STYLE_LINE.items():
        if key in (tone or ""):
            return line
    return _STYLE_LINE["轻松"]


def _vision_fact_prose(fact: str) -> str:
    raw = (fact or "").strip()
    if not raw or "待看图" in raw or "待重看" in raw:
        return ""
    scene = raw
    if "画面：" in raw:
        scene = raw.split("画面：", 1)[1].strip()
    elif "：" in raw:
        scene = raw.split("：", 1)[1].strip()
    scene = scene.strip("。")
    if not scene:
        return ""
    return f"有一页就停在那里：{scene}。"


def _scene_prose(where: str, who: str, doing: str, role: str = "") -> str:
    where = (where or "").strip()
    who = (who or "").strip()
    doing = (doing or "").strip()
    if where and who and doing:
        line = f"{where}这边，{who}还在{doing}"
    else:
        line = "，".join(part for part in (where, who, doing) if part)
    if role:
        return f"{role}：{line}" if line else role
    return line


def expand_from_spark(
    spark: str,
    *,
    cast: str = "",
    tone: str = "",
    page_count: int = 1,
    vision_facts: list[str] | None = None,
) -> dict[str, Any]:
    spark = (spark or "").strip() or "按画面往下讲"
    if spark[-1] not in "。！？…":
        spark += "。"
    facts = [item for item in (vision_facts or []) if item and "待看图" not in item]
    tone = (tone or "").strip() or infer_tone(spark + "".join(facts))
    style = style_for(tone)
    who = f"{cast}在场。" if (cast or "").strip() else ""
    tail = _EXPAND_TAIL["轻松"]
    for key, line in _EXPAND_TAIL.items():
        if key in tone:
            tail = line
            break
    want = _first_clause(spark)
    obstacle = {
        "轻松吐槽": "认真解释的人被抬杠带跑，名字比东西难记",
        "甜": "话到嘴边又停住",
        "热血": "先被堵住，再顶回去",
        "悬疑": "表面闲聊，底下有一句对不上",
        "偏热": "空气比台词更近，话说一半",
        "平": "从小切口进，中间生出岔路",
        "轻松": "旁边的人抬杠，正事被带跑",
    }.get(next((key for key in ("轻松吐槽", "甜", "热血", "悬疑", "偏热", "平", "轻松") if key in tone), "轻松"))
    expanded = f"{spark}{who}他们要的是：{want}。卡住的是：{obstacle}。{tail}"
    if facts:
        expanded += "".join(_vision_fact_prose(item) for item in facts)
    count = max(1, int(page_count or 1))
    raw_facts = list(vision_facts or [])
    beats = []
    for index in range(count):
        role, job = page_arc(index, count)
        if index < len(raw_facts) and raw_facts[index] and "待看图" not in raw_facts[index]:
            beats.append(f"第{index + 1}页｜{role}｜{_vision_fact_prose(raw_facts[index]).strip('。')}。余韵：{job}。")
        else:
            old = _PAGE_ROLES[index] if index < 4 else _PAGE_ROLES[index % 4]
            beats.append(f"第{index + 1}页｜{role}｜{old}。余韵：{job}。")
    outline = f"{expanded}\n" + "\n".join(beats)
    return {
        "expanded": expanded,
        "tone": tone,
        "style": style,
        "cast": (cast or "").strip(),
        "outline": outline,
        "beats": beats,
    }


def _clip(text: str, max_chars: int) -> tuple[str, str, bool]:
    from yizi.story import fit_text

    return fit_text(text, max_chars)


def _first_clause(text: str) -> str:
    raw = (text or "").strip()
    for sep in ("，", ",", "。", "；"):
        if sep in raw:
            raw = raw.split(sep, 1)[0]
            break
    return raw.strip()


def _pick_fit(lines: tuple[str, ...], max_chars: int, start: int, used: set[str] | None = None) -> str:
    if max_chars < 5:
        return ""
    taken = used or set()
    ordered = [lines[(start + i) % len(lines)] for i in range(len(lines))]
    for line in (*ordered, *lines):
        if line and len(line) <= max_chars and line not in taken:
            return line
    return ""


def _local_line(
    page: Page,
    region: TextRegion,
    page_index: int,
    slot: int = 0,
    story: dict[str, Any] | None = None,
    used: set[str] | None = None,
) -> str:
    from yizi.story import STALE_PHRASES, fit_text, is_dump_line, is_generic_label

    if region.skip or region.max_chars < 5:
        return ""
    story = story or {}
    taken = used or set()
    scene = page.scene
    where = _first_clause((scene.where if scene else "") or "")
    doing = _first_clause((scene.doing if scene else "") or "")
    if any(marker in doing for marker in STALE_PHRASES):
        doing = ""
    beat = (page.page_summary or "").strip()
    if any(marker in beat for marker in STALE_PHRASES):
        beat = ""
    mood = (story.get("tone") or (scene.mood if scene else "") or "")
    who = (region.speaker or "").strip()
    if not who or is_generic_label(who) or who == "旁白":
        who = _first_clause((story.get("cast") or (scene.who if scene else "") or "").split("、")[0] if (story.get("cast") or (scene.who if scene else "")) else "") or "有人"
    if is_generic_label(who):
        who = "有人"
    start = page_index + slot
    nouns = _story_nouns("。".join(item for item in ((story.get("premise") or ""), (story.get("expanded") or ""), beat, where, doing) if item))
    clauses: list[str] = []
    for cand in (*_story_clauses(story, page), where, beat, doing):
        if not cand or is_generic_label(cand) or is_dump_line(cand):
            continue
        if len(cand) <= region.max_chars:
            clauses.append(cand)
            continue
        fitted, _, _ = fit_text(cand, region.max_chars)
        if fitted:
            clauses.append(fitted)
    if region.kind == "narration":
        picks = list(clauses)
        for noun in nouns:
            picks.append(f"{noun}。")
        picks.extend(("先看见了。", "就是这儿。", "有人在指。", "越讲越歪。", "还没完。", "先记下。", "到这儿。", "先停住。"))
        picks.extend(_STOCK["narration"])
        return _pick_fit(tuple(picks), region.max_chars, start, taken)
    if region.kind == "thought":
        return _pick_fit(tuple([*clauses, *(_STOCK["thought"])]), region.max_chars, start, taken)
    if region.kind == "sfx":
        return _pick_fit(_STOCK["sfx"], region.max_chars, start, taken)
    pool = list(clauses)
    for noun in nouns:
        pool.extend((f"这是{noun}？", f"认{noun}？", f"{noun}吗。"))
    pool.extend(
        item
        for item in (
            f"{doing}？" if doing else "",
            f"这是{where}。" if where else "",
            f"{who}还在这儿。" if who and who != "有人" else "",
        )
        if item and not is_generic_label(item)
    )
    if "吐槽" in mood:
        pool.extend(("你认真的？", "那是名字吧。", "那不就是名字吗。", "越讲越歪。"))
    if mood in {"偏热", "偏沉"}:
        pool.extend(("靠近一点。", "看着我。"))
    pool.extend(_STOCK["dialogue"])
    if not pool:
        pool = ["你看我干什么。", "……我知道。"]
    return _pick_fit(tuple(pool), region.max_chars, start, taken)


def write_local(project: Project, payload: dict[str, Any]) -> dict[str, Any]:
    pages_out: list[dict[str, Any]] = []
    beats: list[str] = []
    story = payload.get("story") or story_brief(project)
    for item in payload.get("pages") or []:
        page = next((p for p in project.pages if p.id == item["page_id"]), None)
        if not page:
            continue
        idx = project.pages.index(page)
        scene = item.get("scene") or {}
        plan = item.get("box_plan") or box_inventory(page)
        beat = f"第{idx + 1}页：{scene.get('where') or page.name}，{scene.get('who') or ''}。{scene.get('doing') or ''}"
        if plan.get("count"):
            beat += f"。按已有{plan.get('brief') or '框'}写"
        beats.append(beat.strip("，"))
        used = {(region.text or "").strip() for region in page.regions if region.locked and (region.text or "").strip()}
        regions = []
        raw_rows = list(item.get("regions") or [])
        raw_rows.sort(key=lambda row: int(row.get("order") or 0))
        for slot, raw in enumerate(raw_rows):
            region = next((r for r in page.regions if r.id == raw["id"]), None)
            if not region:
                continue
            text = _local_line(page, region, idx, slot, story, used)
            if text:
                used.add(text)
            regions.append({"id": region.id, "text": text, "text_short": text, "speaker": region.speaker})
        pages_out.append(
            {
                "page_id": page.id,
                "summary": page.page_summary or (scene.get("doing") or page.name),
                "regions": regions,
            }
        )
    outline = (project.outline or "").strip() or _local_outline(project, beats)
    if len(outline) > 400:
        outline = outline[:399] + "…"
    return {"outline": outline, "pages": pages_out, "engine": "local", "deepseek": False}


def _local_outline(project: Project, beats: list[str]) -> str:
    head = (project.premise or "").strip() or "按画面往下讲"
    if (project.cast or "").strip():
        head += f"。出场：{project.cast.strip()}"
    if (project.tone or "").strip():
        head += f"。基调：{project.tone.strip()}"
    body = "。".join(beats)
    return f"{head}。{body}" if body else head


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("deepseek_json") from exc
    if not isinstance(data, dict):
        raise RuntimeError("deepseek_json")
    return data


def write_deepseek(settings: Settings, payload: dict[str, Any]) -> dict[str, Any]:
    return call_writer(settings, STYLE_CARD + "\n" + SYSTEM_PROMPT, payload)


def call_writer(
    settings: Settings,
    system: str,
    payload: dict[str, Any],
    *,
    model: str | None = None,
    temperature: float = 0.7,
    timeout: float = 90.0,
) -> dict[str, Any]:
    key = (settings.api_key or "").strip()
    if not key:
        raise RuntimeError("no_key")
    base = (settings.api_base or "https://api.deepseek.com/v1").rstrip("/")
    model = model or settings.writer_model or "deepseek-chat"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        with httpx.Client(timeout=timeout) as client:
            res = client.post(f"{base}/chat/completions", headers=headers, json=body)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"deepseek_http:{exc}") from exc
    if res.status_code == 401:
        raise RuntimeError("deepseek_auth")
    data: dict[str, Any] = {}
    try:
        data = res.json()
    except ValueError:
        data = {}
    err = str(data.get("error") or data.get("code") or "")
    if res.status_code >= 400:
        if "content_filter" in err or (res.status_code == 400 and "filter" in str(data).lower()):
            raise RuntimeError("content_filter")
        if res.status_code == 402:
            raise RuntimeError("deepseek_402")
        raise RuntimeError(f"deepseek_{res.status_code}")
    choice = (data.get("choices") or [{}])[0]
    finish = choice.get("finish_reason") or ""
    if finish in {"content_filter", "content-filter"}:
        raise RuntimeError("content_filter")
    content = (choice.get("message") or {}).get("content") or ""
    parsed = _extract_json(content)
    parsed["engine"] = "deepseek"
    parsed["deepseek"] = True
    parsed["model"] = model
    return parsed


def apply_write_result(
    project: Project,
    result: dict[str, Any],
    *,
    scope: str = "all",
    page_id: str | None = None,
    region_id: str | None = None,
    force: bool = False,
) -> Project:
    if result.get("outline") and not (project.outline or "").strip():
        project.outline = str(result["outline"])
    by_page = {item.get("page_id"): item for item in (result.get("pages") or []) if isinstance(item, dict)}
    for page in project.pages:
        if scope == "page" and page_id and page.id != page_id:
            continue
        item = by_page.get(page.id)
        if not item:
            continue
        if item.get("summary") and scope != "region":
            page.page_summary = str(item["summary"])
        by_id = {row.get("id"): row for row in (item.get("regions") or []) if isinstance(row, dict)}
        updated: list[TextRegion] = []
        for region in page.regions:
            if scope == "region" and region_id and region.id != region_id:
                updated.append(region)
                continue
            if region.skip or (region.locked and not force):
                updated.append(region)
                continue
            row = by_id.get(region.id)
            if not row:
                updated.append(region)
                continue
            text, short, overflow = _clip(str(row.get("text") or ""), region.max_chars)
            if row.get("text_short"):
                extra, extra_short, extra_ov = _clip(str(row["text_short"]), region.max_chars)
                if extra:
                    short = extra_short or extra
                    overflow = overflow or extra_ov
            next_region = region.model_copy(deep=True)
            next_region.text = text
            next_region.text_short = short
            next_region.overflow = overflow or len(text) > region.max_chars
            if row.get("speaker"):
                from yizi.story import is_generic_label

                speaker = str(row["speaker"]).strip()
                if speaker and not is_generic_label(speaker):
                    next_region.speaker = speaker
            updated.append(next_region)
        page.regions = updated
        from yizi.story import unique_page_texts

        unique_page_texts(page)
    return project


def fill_unwritten_boxes(
    project: Project,
    *,
    scope: str = "all",
    page_id: str | None = None,
    region_id: str | None = None,
) -> int:
    story = story_brief(project)
    filled = 0
    for idx, page in enumerate(project.pages):
        if scope == "page" and page_id and page.id != page_id:
            continue
        used = {(region.text or "").strip() for region in page.regions if (region.text or "").strip()}
        for slot, region in enumerate(ordered_regions(page)):
            if scope == "region" and region_id and region.id != region_id:
                continue
            if region.skip or region.locked:
                continue
            if (region.text or "").strip():
                continue
            text = _local_line(page, region, idx, slot, story, used)
            if not text:
                continue
            region.text = text
            region.text_short = text
            region.overflow = False
            used.add(text)
            filled += 1
    return filled


def _write_once(
    project: Project,
    settings: Settings,
    *,
    scope: str,
    page_id: str | None,
    region_id: str | None,
    force: bool,
) -> dict[str, Any]:
    payload = build_write_payload(project, scope=scope, page_id=page_id, region_id=region_id)
    meta: dict[str, Any] = {"payload_pages": len(payload["pages"]), "deepseek": False, "engine": "local"}
    if settings.api_key.strip():
        try:
            result = write_deepseek(settings, payload)
            meta["deepseek"] = True
            meta["engine"] = "deepseek"
        except RuntimeError as exc:
            code = str(exc)
            meta["error"] = code
            result = write_local(project, payload)
            meta["engine"] = "local"
            meta["fallback"] = code
            if code == "content_filter":
                meta["hint"] = "DeepSeek 拒写。改场景卡再试，或手写这一框。"
            elif code == "no_key":
                meta["hint"] = "没填 Key，本机按已有框写了草稿。"
            elif code == "deepseek_auth":
                meta["hint"] = "Key 无效。本机先按已有框写了草稿。"
            elif code == "deepseek_402":
                meta["hint"] = "编剧 API 余额不足，这次用的是本机按框写的草稿，不是接口写成的。"
            else:
                meta["hint"] = "编剧 API 没走成，这次用的是本机按框写的草稿，不是接口写成的。"
    else:
        result = write_local(project, payload)
        meta["hint"] = "没填 Key，本机已按这一页已有的框写了草稿。不会假装接口写成了。"
    apply_write_result(
        project,
        result,
        scope=scope,
        page_id=page_id,
        region_id=region_id,
        force=force,
    )
    filled = fill_unwritten_boxes(project, scope=scope, page_id=page_id, region_id=region_id)
    if filled:
        meta["filled_empty"] = filled
        if not meta.get("hint"):
            meta["hint"] = "已按已有框补上还没写的词。"
    meta["outline"] = project.outline
    return meta


def write_story(
    project: Project,
    settings: Settings,
    *,
    scope: str = "all",
    page_id: str | None = None,
    region_id: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    from yizi.story import assign_speakers

    assign_speakers(project)
    if scope == "all" and not region_id and len(project.pages) > 1:
        metas: list[dict[str, Any]] = []
        for page in project.pages:
            metas.append(_write_once(project, settings, scope="page", page_id=page.id, region_id=None, force=force))
        fallbacks = [item.get("fallback") for item in metas if item.get("fallback")]
        used_ds = bool(settings.api_key.strip()) and all(item.get("deepseek") for item in metas)
        hint = "已按页回填全部成页。每页按已有框写，前后页接上。"
        if fallbacks:
            hint += " 有页用了本机草稿。"
        elif not settings.api_key.strip():
            hint = "没填 Key，本机已按页、按已有框写完全部成页。"
        return {
            "payload_pages": len(project.pages),
            "deepseek": used_ds,
            "engine": "deepseek" if used_ds else "local",
            "fallback": fallbacks[0] if fallbacks else "",
            "pages_written": len(project.pages),
            "filled_empty": sum(int(item.get("filled_empty") or 0) for item in metas),
            "hint": hint,
            "outline": project.outline,
        }
    return _write_once(project, settings, scope=scope, page_id=page_id, region_id=region_id, force=force)


def _writer_hint(code: str, *, empty_key: str, failed: str) -> str:
    return {
        "no_key": empty_key,
        "deepseek_auth": "Key 无效。检查设置里粘贴的是不是完整 Key。",
        "deepseek_402": "编剧 API 余额不足。本机先扩写定调，不是接口写成的。",
        "content_filter": "接口拒写。把想法写得普通一点再试。",
    }.get(code, failed)


def design_story_local(project: Project) -> dict[str, Any]:
    pack = expand_from_spark(
        project.premise,
        cast=project.cast,
        tone=project.tone,
        page_count=max(1, len(project.pages)),
        vision_facts=vision_facts(project),
    )
    pages_out: list[dict[str, Any]] = []
    beats: list[str] = []
    for index, page in enumerate(project.pages):
        scene = page.scene
        role, job = page_arc(index, max(1, len(project.pages)))
        if scene and (scene.where or scene.doing or scene.who):
            summary = _scene_prose(scene.where, scene.who, scene.doing, role)
        else:
            raw = pack["beats"][index] if index < len(pack["beats"]) else f"第{index + 1}页往下讲"
            summary = raw.split("｜", 1)[-1].strip("。") if "｜" in raw else raw.split("：", 1)[-1].strip("。")
        summary = summary[:42]
        plan = box_inventory(page)
        pages_out.append({"page_id": page.id, "summary": summary, "role": role, "box_plan": plan["brief"]})
        line = f"第{index + 1}页｜{role}｜{summary}。余韵：{job}"
        if plan["count"]:
            line += f"。按已有{plan['brief']}写。"
        beats.append(line)
    outline = pack["expanded"]
    if beats:
        outline = outline + "\n" + "\n".join(beats)
    return {
        "expanded": pack["expanded"],
        "tone": pack["tone"],
        "style": pack["style"],
        "cast": pack["cast"] or project.cast,
        "outline": outline,
        "pages": pages_out,
        "engine": "local",
        "deepseek": False,
    }


def apply_design_result(project: Project, result: dict[str, Any]) -> Project:
    from yizi.story import has_foreign_place

    for key in ("expanded", "tone", "style", "cast", "outline"):
        value = result.get(key)
        if value:
            setattr(project, key, str(value).strip())
    extra = "。".join(f"{label}{result[key]}" for label, key in (("他们要的是：", "want"), ("卡住的是：", "obstacle")) if result.get(key))
    if extra and extra not in (project.expanded or ""):
        project.expanded = f"{extra}。{project.expanded}".strip("。")
    if has_foreign_place(project.outline, project.premise) or has_foreign_place(project.expanded, project.premise):
        local = design_story_local(project)
        project.expanded = str(local.get("expanded") or project.expanded)
        project.outline = str(local.get("outline") or project.outline)
        result = {**result, **local}
    by_page = {item.get("page_id"): item for item in (result.get("pages") or []) if isinstance(item, dict)}
    for page in project.pages:
        item = by_page.get(page.id)
        if item and item.get("summary"):
            summary = str(item["summary"]).strip()
            if not has_foreign_place(summary, project.premise):
                page.page_summary = summary
    return project


def design_story(project: Project, settings: Settings) -> dict[str, Any]:
    payload = {
        "title": project.name,
        "story": story_brief(project),
        "vision": visual_board(project),
        "fill_existing_boxes_only": True,
        "pages": [
            {
                "page_id": page.id,
                "name": page.name,
                "order": index + 1,
                "scene": scene_as_prompt(page.scene, page.name, page.notes),
                "summary": page.page_summary,
                "box_plan": box_inventory(page),
                "boxes": [
                    _region_payload(region, page, order)
                    for order, region in enumerate(ordered_regions(page, [item for item in page.regions if not item.skip]), start=1)
                ][:8],
            }
            for index, page in enumerate(project.pages)
        ],
    }
    meta: dict[str, Any] = {"deepseek": False, "engine": "local", "pages": len(payload["pages"])}
    if settings.api_key.strip():
        try:
            result = call_writer(settings, STYLE_CARD + "\n" + DESIGN_PROMPT, payload, temperature=0.65)
            meta["deepseek"] = True
            meta["engine"] = "deepseek"
            meta["hint"] = "已按 Pixiv 短篇口吻织好情节。扩写和大纲可改，有几框就安排几拍。"
        except RuntimeError as exc:
            code = str(exc)
            meta["error"] = code
            meta["fallback"] = code
            result = design_story_local(project)
            meta["hint"] = _writer_hint(code, empty_key="没填 Key，本机先扩写定调。", failed="编剧没写成，本机先扩写定调。")
    else:
        result = design_story_local(project)
        meta["hint"] = "没填 Key，本机已按 Pixiv 短篇口吻扩写定调。设置里贴上编剧 Key 会更像投稿小说。"
    apply_design_result(project, result)
    if not any(page.scene and page.scene.source in {"vlm", "human"} for page in project.pages):
        meta["hint"] = (meta.get("hint") or "") + " 还没看图。组装时会先看各页再织情节。"
    meta["outline"] = project.outline
    meta["tone"] = project.tone
    meta["expanded"] = project.expanded
    return meta


def ping_writer(settings: Settings) -> dict[str, Any]:
    if not (settings.api_key or "").strip():
        return {"ok": False, "hint": "还没填 Key。先粘贴编剧 Key，不填也能用本机草稿。"}
    model = settings.writer_model or "deepseek-chat"
    try:
        parsed = call_writer(
            settings,
            '只返回 JSON：{"ok":true}',
            {"ask": "ping"},
            model=model,
            temperature=0,
            timeout=20.0,
        )
        return {
            "ok": True,
            "model": parsed.get("model") or model,
            "hint": f"编剧已连通，模型 {parsed.get('model') or model}。",
        }
    except (RuntimeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        code = str(exc)
        return {
            "ok": False,
            "error": code,
            "model": model,
            "hint": _writer_hint(code, empty_key="还没填 Key。", failed=f"没连通（{code}）。检查地址、模型名是否小写、Key 是否有效。"),
        }
