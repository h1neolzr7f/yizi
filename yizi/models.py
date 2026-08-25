from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RegionKind = Literal["dialogue", "narration", "thought", "sfx"]
TextDirection = Literal["vertical", "horizontal"]
IntimacyLevel = Literal["none", "suggestive", "explicit"]
LetteringWriteScope = Literal["all", "page", "region"]
SceneSource = Literal["cv", "vlm", "human"]


class TextRegion(BaseModel):
    id: str
    kind: RegionKind = "dialogue"
    direction: TextDirection = "horizontal"
    x: float = 0
    y: float = 0
    w: float = 120
    h: float = 80
    skip: bool = False
    speaker: str = ""
    max_chars: int = 24
    draft: str = ""
    text: str = ""
    text_short: str = ""
    locked: bool = False
    overflow: bool = False
    tone: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


class SceneCard(BaseModel):
    where: str = ""
    who: str = ""
    doing: str = ""
    mood: str = ""
    jump_from_prev: str = ""
    intimacy: IntimacyLevel = "none"
    image_hash: str = ""
    source: SceneSource = "cv"
    notes: str = ""


class Layer(BaseModel):
    id: str
    name: str
    kind: str = "image"
    visible: bool = True
    file: str
    extra: dict[str, Any] = Field(default_factory=dict)


class Page(BaseModel):
    id: str
    name: str
    width: int = 832
    height: int = 1216
    source_file: str | None = "source.png"
    layers: list[Layer] = Field(default_factory=list)
    notes: str = ""
    regions: list[TextRegion] = Field(default_factory=list)
    scene: SceneCard | None = None
    page_summary: str = ""
    see_hint: str = ""


class Project(BaseModel):
    id: str
    name: str
    created_at: str
    updated_at: str
    pages: list[Page] = Field(default_factory=list)
    schema_version: int = 1
    premise: str = ""
    cast: str = ""
    tone: str = ""
    style: str = ""
    expanded: str = ""
    outline: str = ""


LetteringVision = Literal["qwen3vl_4b", "qwen3vl_2b", "joycaption_q3", "off"]
LetteringDetect = Literal["auto", "yolo", "cv"]


class Settings(BaseModel):
    api_key: str = ""
    api_base: str = "https://api.deepseek.com/v1"
    writer_model: str = "deepseek-chat"
    outline_model: str = "deepseek-chat"
    lettering_vision: LetteringVision = "qwen3vl_4b"
    lettering_detect: LetteringDetect = "auto"


class ProjectCreate(BaseModel):
    name: str = "未命名剧本"


class ProjectPatch(BaseModel):
    name: str | None = None
    premise: str | None = None
    cast: str | None = None
    tone: str | None = None
    style: str | None = None
    expanded: str | None = None
    outline: str | None = None


class PagePatch(BaseModel):
    name: str | None = None
    notes: str | None = None
    page_summary: str | None = None


class PageOrder(BaseModel):
    page_ids: list[str]


class RegionCreate(BaseModel):
    kind: RegionKind = "dialogue"
    direction: TextDirection | None = None
    x: float = 80
    y: float = 80
    w: float = 160
    h: float = 120
    skip: bool = False
    speaker: str = ""


class RegionPatch(BaseModel):
    kind: RegionKind | None = None
    direction: TextDirection | None = None
    x: float | None = None
    y: float | None = None
    w: float | None = None
    h: float | None = None
    skip: bool | None = None
    speaker: str | None = None
    max_chars: int | None = None
    draft: str | None = None
    text: str | None = None
    text_short: str | None = None
    locked: bool | None = None
    tone: str | None = None


class SceneCardPatch(BaseModel):
    where: str | None = None
    who: str | None = None
    doing: str | None = None
    mood: str | None = None
    jump_from_prev: str | None = None
    intimacy: IntimacyLevel | None = None
    notes: str | None = None
    source: SceneSource | None = None


class LetteringDetectRequest(BaseModel):
    replace: bool = True


class LetteringSeeRequest(BaseModel):
    force: bool = False


class LetteringWriteRequest(BaseModel):
    scope: LetteringWriteScope = "all"
    page_id: str | None = None
    region_id: str | None = None
    force: bool = False


class LetteringRunRequest(BaseModel):
    scope: LetteringWriteScope = "all"
    page_id: str | None = None
    replace: bool = True
    force: bool = False
    compose: bool = False


class StoryAssembleRequest(BaseModel):
    compose: bool = True
    detect_missing: bool = True
    see_missing: bool = True
    snapshot: bool = True


class StoryDesignRequest(BaseModel):
    detect_missing: bool = True
    see_missing: bool = True


class LetteringBackfillRequest(BaseModel):
    page_id: str
    detect: bool = True
    see: bool = True
    write: bool = True
    compose: bool = True
    force: bool = False


class RegionMergeRequest(BaseModel):
    region_ids: list[str]


class LayerPatch(BaseModel):
    visible: bool | None = None


class SettingsUpdate(BaseModel):
    api_key: str | None = None
    deepseek_api_key: str | None = None
    api_base: str | None = None
    deepseek_api_base: str | None = None
    writer_model: str | None = None
    deepseek_model: str | None = None
    outline_model: str | None = None
    deepseek_outline_model: str | None = None
    lettering_vision: LetteringVision | None = None
    lettering_detect: LetteringDetect | None = None
