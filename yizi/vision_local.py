"""本机 Qwen3-VL / JoyCaption：llama.cpp 看图，用完卸模。图不上网。"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import ssl
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from yizi.config import ensure_data_dirs, llama_dir, models_dir
from yizi.hw import release_cuda
from yizi.models import LetteringVision, SceneCard

log = logging.getLogger("yizi.vision")

LLAMA_TAG = "b10603"
LLAMA_ZIP = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/"
    f"llama-{LLAMA_TAG}-bin-win-cuda-13.3-x64.zip"
)
CUDART_ZIP = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/"
    f"cudart-llama-bin-win-cuda-13.3-x64.zip"
)

HF_ENDPOINTS = (
    os.environ.get("HF_ENDPOINT", "").rstrip("/"),
    "https://hf-mirror.com",
    "https://huggingface.co",
)

BACKENDS: dict[str, dict[str, Any]] = {
    "qwen3vl_4b": {
        "label": "Qwen3-VL-4B Q4",
        "repo": "Qwen/Qwen3-VL-4B-Instruct-GGUF",
        "llm": (
            "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
            "Qwen3-VL-4B-Instruct-Q4_K_M.gguf",
        ),
        "mmproj": (
            "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf",
            "mmproj-Qwen3-VL-4B-Instruct-Q8_0.gguf",
            "mmproj-Qwen3VL-4B-Instruct-F16.gguf",
            "mmproj-Qwen3-VL-4B-Instruct-F16.gguf",
        ),
        "max_side": 768,
    },
    "qwen3vl_2b": {
        "label": "Qwen3-VL-2B Q4",
        "repo": "Qwen/Qwen3-VL-2B-Instruct-GGUF",
        "llm": (
            "Qwen3VL-2B-Instruct-Q4_K_M.gguf",
            "Qwen3-VL-2B-Instruct-Q4_K_M.gguf",
        ),
        "mmproj": (
            "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
            "mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf",
            "mmproj-Qwen3VL-2B-Instruct-F16.gguf",
            "mmproj-Qwen3-VL-2B-Instruct-F16.gguf",
        ),
        "max_side": 640,
    },
    "joycaption_q3": {
        "label": "JoyCaption Q3",
        "repo": "",
        "llm": ("joycaption-q3_k_m.gguf",),
        "mmproj": ("mmproj-joycaption.gguf",),
        "max_side": 768,
    },
}

SCENE_PROMPT = (
    "你是漫画分镜看图助手。只根据画面写 JSON。识别角色和画面，不要写台词，不要翻译泡里的字。"
    "where：地点、天气、前后景、左右环境，写具体。"
    "who：每个能看见的人分别写发型、衣服、站位，不要用「大约两个人」糊弄。"
    "doing：按格、按人写动作。每人必须写清面向哪、手在干什么、脚在哪、表情。不要只写「认柱子」「走路」。"
    "panels：从上到下每格画面一句，写构图和人物关系，不写对白。"
    "mood：画面气氛。intimacy：none|suggestive|explicit。"
    '示例：{"where":"白天室外街道，右侧有电线杆和木屋","who":"左：浅色双马尾校服女生；中：戴帽提包的女生；右：黑短发校服女生",'
    '"doing":"上格浅色双马尾女生面向画面举手竖食指；中格三人并排往前走，中间的人双手抱书包；下格双马尾女生回头张开双手解释，另外两人看着她",'
    '"panels":"上：一人举手说明；中：三人走在巷子里；下：一人回头解释","mood":"轻快","intimacy":"none"}'
)

PAGE_PROMPT = (
    "你是成页漫画看图助手。图上红框只是泡的位置，用来判断谁靠近这框，不是让你写台词。"
    "只根据画面识别角色和动作。禁止写台词、禁止翻译泡里的字、禁止输出日文。"
    "只输出一个 JSON，不要 markdown。"
    '格式：{"where":"地点+环境+前后景","who":"每人发型衣服和左右站位","doing":"按格按人写面向、手势、脚步、表情",'
    '"panels":"从上到下每格一句构图","mood":"","intimacy":"none","summary":"这一页画面在发生什么，不要引用台词",'
    '"items":[{"i":1,"skip":false,"kind":"dialogue","speaker":"靠近这框的角色外貌或站位","doing":"这个人此刻的动作"}]}'
    "kind 只能是 dialogue/narration/thought/sfx/title。"
    "标题、作者署名、许可证 skip=true。items 必须覆盖所有编号，text 一律留空字符串。"
    "speaker 写画面里的人，用外貌或左右站位，不要编台词。"
    "doing 不要四个字套话。图上没有的东西不要写。不要复述示例。不要默认写成学校走廊，除非能看见教室门或鞋柜。"
)

CROP_PROMPT = (
    "这些是漫画格子/泡附近的特写，左上角红底白字是编号。"
    "不要读泡里的字，不要写台词。只看泡旁边是谁、这个人面向哪、手和表情在做什么。"
    '只输出 JSON：{"items":[{"i":1,"speaker":"框左侧短发女生","doing":"侧过脸看同伴，一只手抓着书包带"}]}'
    "每个编号不同。竖排从右往左看人，不看字。"
)


def _enabled() -> bool:
    flag = os.environ.get("YIZI_VLM", os.environ.get("KOMALAYER_VLM", "1")).strip().lower()
    return flag not in {"0", "off", "false", "no"}


_LLAMA_EXE: Path | None = None
_STATUS_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def llama_exe() -> Path | None:
    global _LLAMA_EXE
    if _LLAMA_EXE and _LLAMA_EXE.exists():
        return _LLAMA_EXE
    root = llama_dir()
    for name in ("llama-mtmd-cli.exe", "llama-mtmd-cli"):
        found = root / name
        if found.exists():
            _LLAMA_EXE = found
            return found
        nested = next(root.rglob(name), None)
        if nested:
            _LLAMA_EXE = nested
            return nested
    return None


def _first_existing(folder: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = folder / name
        if path.exists() and path.stat().st_size > 1024:
            return path
    for path in folder.glob("*.gguf"):
        low = path.name.lower()
        if "mmproj" in low:
            continue
        if any(token in low for token in ("q4_k_m", "q4_k", "q4")):
            return path
    return None


def _first_mmproj(folder: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = folder / name
        if path.exists() and path.stat().st_size > 1024:
            return path
    found = next(folder.glob("*mmproj*.gguf"), None)
    if found and found.stat().st_size > 1024:
        return found
    return None


def backend_dir(name: str) -> Path:
    return models_dir() / name


def resolve_files(name: str) -> dict[str, Path | None]:
    spec = BACKENDS.get(name) or {}
    folder = backend_dir(name)
    return {
        "llm": _first_existing(folder, tuple(spec.get("llm") or ())),
        "mmproj": _first_mmproj(folder, tuple(spec.get("mmproj") or ())),
    }


def status(name: LetteringVision | str = "qwen3vl_4b") -> dict[str, Any]:
    import time

    now = time.monotonic()
    hit = _STATUS_CACHE.get(str(name))
    if hit and now - hit[0] < 2.5:
        return hit[1]
    info = _status_uncached(name)
    _STATUS_CACHE[str(name)] = (now, info)
    return info


def _status_uncached(name: LetteringVision | str = "qwen3vl_4b") -> dict[str, Any]:
    ensure_data_dirs()
    if name == "off":
        return {"ready": False, "backend": "off", "note": "已关闭本机视觉，只用几何反推。"}
    spec = BACKENDS.get(str(name))
    if not spec:
        return {"ready": False, "backend": name, "note": f"未知后端 {name}"}
    exe = llama_exe()
    files = resolve_files(str(name))
    missing = []
    if exe is None:
        missing.append("llama-mtmd-cli")
    if files["llm"] is None:
        missing.append("语言模型 GGUF")
    if files["mmproj"] is None:
        missing.append("mmproj")
    ready = not missing
    note = (
        f"{spec['label']} 已就绪，看完卸模。"
        if ready
        else f"{spec['label']} 未就绪：缺 {'、'.join(missing)}。设置里点「准备本机视觉」。"
    )
    return {
        "ready": ready,
        "backend": name,
        "label": spec["label"],
        "exe": str(exe) if exe else "",
        "llm": str(files["llm"] or ""),
        "mmproj": str(files["mmproj"] or ""),
        "missing": missing,
        "note": note,
    }


def _parse_json_obj(text: str) -> dict[str, Any] | None:
    blob = (text or "").strip()
    if not blob:
        return None
    match = re.search(r"\{.*\}", blob, flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _as_scene_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        order = ("panel", "格", "where", "地点", "side", "站位", "who", "谁", "speaker", "look", "外貌", "doing", "action", "动作")
        bits = [str(value[key]).strip() for key in order if value.get(key)]
        if not bits:
            bits = [str(item).strip() for item in value.values() if item]
        return "：".join(bit for bit in bits if bit)
    if isinstance(value, list):
        bits = [_as_scene_text(item) for item in value]
        return "；".join(bit for bit in bits if bit)
    return str(value).strip()


def parse_vlm_json(text: str) -> SceneCard | None:
    data = _parse_json_obj(text)
    if not data:
        return None
    intimacy = str(data.get("intimacy") or "none").strip().lower()
    if intimacy not in {"none", "suggestive", "explicit"}:
        intimacy = "none"
    where = _as_scene_text(data.get("where"))
    who = _as_scene_text(data.get("who"))
    doing = _as_scene_text(data.get("doing") or data.get("actions"))
    panels = _as_scene_text(data.get("panels") or data.get("notes"))
    if not (where or who or doing or panels):
        return None
    return SceneCard(
        where=where or "画面地点不明",
        who=who or "人数不明",
        doing=doing or "动作不明",
        mood=str(data.get("mood") or "").strip(),
        jump_from_prev=str(data.get("jump_from_prev") or "").strip(),
        intimacy=intimacy,  # type: ignore[arg-type]
        notes=panels,
        source="vlm",
    )


def parse_page_lettering(text: str) -> dict[str, Any] | None:
    data = _parse_json_obj(text)
    if not data:
        return None
    items = data.get("items")
    extra = data.get("extra")
    if items is None and isinstance(data.get("regions"), list):
        items = data["regions"]
    if not isinstance(items, list):
        items = []
    if not isinstance(extra, list):
        extra = []
    card = parse_vlm_json(json.dumps(data, ensure_ascii=False))
    if card is None and not items and not extra:
        return None
    return {
        "card": card,
        "items": items,
        "extra": extra,
        "summary": str(data.get("summary") or "").strip(),
    }


def _write_preview(image_bytes: bytes, dest: Path, max_side: int) -> None:
    import cv2
    import numpy as np

    bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        dest.write_bytes(image_bytes)
        return
    h, w = bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    dest.write_bytes(buf.tobytes() if ok else image_bytes)


def _run_mtmd(
    image_bytes: bytes,
    backend: LetteringVision | str,
    prompt: str,
    *,
    n_predict: int = 256,
    ctx: int = 2048,
    temp: str = "0.2",
    timeout: int = 240,
    annotated: bytes | None = None,
    max_side: int | None = None,
) -> str:
    info = status(backend)
    if not info["ready"]:
        return ""
    spec = BACKENDS[str(backend)]
    tmp = Path(tempfile.mkdtemp(prefix="yizi-vlm-"))
    image_path = tmp / "page.jpg"
    try:
        raw = annotated if annotated else image_bytes
        _write_preview(raw, image_path, int(max_side or spec["max_side"]))
        cmd = [
            info["exe"],
            "-m",
            info["llm"],
            "--mmproj",
            info["mmproj"],
            "--image",
            str(image_path),
            "-p",
            prompt,
            "-n",
            str(n_predict),
            "--temp",
            temp,
            "--ctx-size",
            str(ctx),
            "-ngl",
            "99",
        ]
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path(info["exe"]).parent),
        )
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            log.warning("vlm exit %s: %s", proc.returncode, text[-800:])
        return proc.stdout or text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        release_cuda()


def try_scene(image_bytes: bytes, backend: LetteringVision | str = "qwen3vl_4b", *, story_note: str = "") -> SceneCard | None:
    if not _enabled() or backend == "off":
        return None
    info = status(backend)
    if not info["ready"]:
        log.info("vlm skip: %s", info["note"])
        return None
    prompt = SCENE_PROMPT
    if story_note:
        prompt = f"{prompt}\n{story_note}"
    try:
        text = _run_mtmd(image_bytes, backend, prompt, n_predict=768, ctx=2048, timeout=240)
        card = parse_vlm_json(text)
        if card is None:
            log.warning("vlm parse miss: %s", text[-800:])
        return card
    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired) as exc:
        log.warning("vlm failed: %s", exc)
        return None


def try_page_lettering(
    image_bytes: bytes,
    backend: LetteringVision | str = "qwen3vl_4b",
    *,
    annotated: bytes | None = None,
    box_note: str = "",
    story_note: str = "",
) -> dict[str, Any] | None:
    if not _enabled() or backend == "off":
        return None
    info = status(backend)
    if not info["ready"]:
        log.info("vlm skip: %s", info["note"])
        return None
    prompt = PAGE_PROMPT
    if box_note:
        prompt = f"{prompt}\n编号框：{box_note}"
    if story_note:
        prompt = f"{prompt}\n{story_note}"
    try:
        text = _run_mtmd(
            image_bytes,
            backend,
            prompt,
            n_predict=1024,
            ctx=3072,
            temp="0.25",
            timeout=300,
            annotated=annotated,
        )
        parsed = parse_page_lettering(text)
        if parsed is None:
            log.warning("vlm lettering parse miss: %s", text[-800:])
        return parsed
    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired) as exc:
        log.warning("vlm lettering failed: %s", exc)
        return None


def try_crop_lettering(
    sheet_bytes: bytes,
    backend: LetteringVision | str = "qwen3vl_4b",
    *,
    ids: list[int] | None = None,
    story_note: str = "",
) -> dict[str, Any] | None:
    if not _enabled() or backend == "off":
        return None
    info = status(backend)
    if not info["ready"]:
        return None
    note = f"编号：{ids}" if ids else ""
    prompt = CROP_PROMPT
    if note:
        prompt = f"{prompt}\n{note}"
    if story_note:
        prompt = f"{prompt}\n{story_note}"
    try:
        text = _run_mtmd(
            sheet_bytes,
            backend,
            prompt,
            n_predict=640,
            ctx=2560,
            temp="0.2",
            timeout=300,
            annotated=sheet_bytes,
            max_side=1024,
        )
        parsed = parse_page_lettering(text)
        if parsed is None:
            log.warning("vlm crop parse miss: %s", text[-800:])
        return parsed
    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired) as exc:
        log.warning("vlm crop failed: %s", exc)
        return None


def _urlopen(req: Request, timeout: int = 45):
    socket.setdefaulttimeout(timeout)
    ctx = ssl._create_unverified_context()
    try:
        return urlopen(req, timeout=timeout, context=ctx)
    except URLError:
        return urlopen(req, timeout=timeout)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(8):
        try:
            existing = tmp.stat().st_size if tmp.exists() else 0
            headers = {"User-Agent": "yizi-lettering/0.1"}
            if existing:
                headers["Range"] = f"bytes={existing}-"
            req = Request(url, headers=headers)
            with _urlopen(req, timeout=45) as resp, open(tmp, "ab") as fh:
                status = int(getattr(resp, "status", 200) or 200)
                if existing and status == 200:
                    fh.seek(0)
                    fh.truncate()
                    existing = 0
                got = existing
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    fh.write(chunk)
                    got += len(chunk)
                    if got and got % (32 * 1024 * 1024) < 256 * 1024:
                        log.info("download %s %sMB", dest.name, got // (1024 * 1024))
            if not tmp.exists() or tmp.stat().st_size < 1024:
                raise RuntimeError(f"下载不完整：{url}")
            tmp.replace(dest)
            return
        except (OSError, URLError, TimeoutError, RuntimeError) as exc:
            last_err = exc
            log.warning("download retry %s %s: %s", attempt + 1, dest.name, exc)
    if last_err:
        raise last_err
    raise RuntimeError(f"下不到 {url}")


def _download_hf(repo: str, filename: str, dest: Path) -> None:
    last_err: Exception | None = None
    for url in _hf_file_urls(repo, filename):
        try:
            log.info("download %s", url)
            _download(url, dest)
            if dest.exists() and dest.stat().st_size > 1024:
                return
        except OSError as exc:
            last_err = exc
            dest.unlink(missing_ok=True)
            dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)
    if last_err:
        raise last_err
    raise RuntimeError(f"下不到 {repo}/{filename}")


def _unzip(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)


def _has_cudart(root: Path) -> bool:
    return any(root.glob("cudart64_*.dll"))


def ensure_llama() -> Path:
    ensure_data_dirs()
    root = llama_dir()
    zip_path = root / f"llama-{LLAMA_TAG}-cuda13.zip"
    cudart_path = root / f"cudart-{LLAMA_TAG}.zip"
    exe = llama_exe()
    if exe is None:
        if not zip_path.exists():
            log.info("download llama.cpp %s", LLAMA_ZIP)
            _download(LLAMA_ZIP, zip_path)
        _unzip(zip_path, root)
        exe = llama_exe()
    if not _has_cudart(root):
        if not cudart_path.exists():
            log.info("download cudart %s", CUDART_ZIP)
            try:
                _download(CUDART_ZIP, cudart_path)
            except OSError as exc:
                log.warning("cudart download skipped: %s", exc)
        if cudart_path.exists():
            _unzip(cudart_path, root)
    if not exe:
        raise RuntimeError("llama-mtmd-cli 解压后仍找不到")
    return exe


def _hf_file_urls(repo: str, filename: str) -> list[str]:
    urls: list[str] = []
    for base in HF_ENDPOINTS:
        if not base:
            continue
        urls.append(f"{base}/{repo}/resolve/main/{filename}")
    if not urls:
        urls.append(f"https://huggingface.co/{repo}/resolve/main/{filename}")
    urls.append(f"https://www.modelscope.cn/models/{repo}/resolve/master/{filename}")
    return urls


def ensure_backend(name: str = "qwen3vl_4b") -> dict[str, Any]:
    if name == "off":
        return status(name)
    spec = BACKENDS.get(name)
    if not spec:
        raise ValueError(name)
    ensure_llama()
    folder = backend_dir(name)
    folder.mkdir(parents=True, exist_ok=True)
    files = resolve_files(name)
    if files["llm"] is None:
        if not spec.get("repo"):
            raise RuntimeError(f"{spec['label']} 需要手动把 GGUF 放到 {folder}")
        last_err: Exception | None = None
        for filename in spec["llm"]:
            dest = folder / filename
            try:
                _download_hf(spec["repo"], filename, dest)
                if dest.exists() and dest.stat().st_size > 1024:
                    break
            except OSError as exc:
                last_err = exc
                dest.unlink(missing_ok=True)
        else:
            if last_err:
                raise last_err
    if resolve_files(name)["mmproj"] is None:
        last_err = None
        for filename in spec["mmproj"]:
            dest = folder / filename
            try:
                _download_hf(spec["repo"], filename, dest)
                if dest.exists() and dest.stat().st_size > 1024:
                    break
            except OSError as exc:
                last_err = exc
                dest.unlink(missing_ok=True)
        else:
            if last_err:
                raise last_err
    _STATUS_CACHE.clear()
    info = status(name)
    if not info["ready"]:
        raise RuntimeError(info["note"])
    return info
