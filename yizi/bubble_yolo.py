"""本机漫画泡框检出：HuggingFace Kiuyha/Manga-Bubble-YOLO（Apache-2.0）。

不抄 GPL 的 manga-image-translator / comic-text-detector 代码。
权重约 6MB，ONNX 本机推理，图不上网。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import cv2
import numpy as np

from yizi.config import ensure_data_dirs, models_dir

MODEL_ID = "Kiuyha/Manga-Bubble-YOLO"
MODEL_NAME = "yolo26n.onnx"
MODEL_URLS = (
    f"https://hf-mirror.com/{MODEL_ID}/resolve/main/onnx/{MODEL_NAME}",
    f"https://huggingface.co/{MODEL_ID}/resolve/main/onnx/{MODEL_NAME}",
)
INPUT_SIZE = 1280
CONF_THRES = 0.22

_SESSION = None


def _enabled() -> bool:
    flag = os.environ.get("YIZI_DETECT", "1").strip().lower()
    return flag not in {"0", "off", "false", "no"}


def model_path() -> Path:
    return models_dir() / "bubble_yolo" / MODEL_NAME


def status() -> dict[str, Any]:
    ensure_data_dirs()
    path = model_path()
    ready = path.exists() and path.stat().st_size > 1024 * 1024
    return {
        "ready": ready,
        "backend": "manga-bubble-yolo",
        "label": "Manga-Bubble-YOLO",
        "file": str(path) if ready else "",
        "note": (
            "本机 YOLO 泡框已就绪（Apache-2.0，图不上网）。"
            if ready
            else "框检出模型未就绪。设置里点「准备框检出」，约 6MB。"
        ),
    }


def ensure_model() -> Path:
    ensure_data_dirs()
    path = model_path()
    if path.exists() and path.stat().st_size > 1024 * 1024:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    last_error: Exception | None = None
    for url in MODEL_URLS:
        try:
            req = Request(url, headers={"User-Agent": "yizi-detect/0.1"})
            with urlopen(req, timeout=120) as res, tmp.open("wb") as out:
                while True:
                    chunk = res.read(1024 * 256)
                    if not chunk:
                        break
                    out.write(chunk)
            if tmp.stat().st_size < 1024 * 1024:
                tmp.unlink(missing_ok=True)
                raise RuntimeError("框检出模型下载不完整")
            tmp.replace(path)
            return path
        except Exception as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
    raise RuntimeError(f"框检出模型下载失败：{last_error}")


def _session():
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    import onnxruntime as ort

    path = model_path()
    if not path.exists():
        return None
    providers = ["CPUExecutionProvider"]
    avail = ort.get_available_providers()
    if "CUDAExecutionProvider" in avail:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    _SESSION = ort.InferenceSession(str(path), providers=providers)
    return _SESSION


def _letterbox(bgr: np.ndarray, size: int = INPUT_SIZE) -> tuple[np.ndarray, float, int, int]:
    h, w = bgr.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, scale, left, top


def _parse_output(raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        return np.zeros((0, 6), np.float32)
    if arr.shape[0] == 6 and arr.shape[1] != 6:
        arr = arr.T
    if arr.shape[1] < 5:
        return np.zeros((0, 6), np.float32)
    if arr.shape[1] == 5:
        pad = np.zeros((arr.shape[0], 1), arr.dtype)
        arr = np.concatenate([arr, pad], axis=1)
    return arr[:, :6]


def detect_boxes(bgr: np.ndarray, *, conf: float = CONF_THRES) -> list[tuple[int, int, int, int, float]]:
    if not _enabled():
        return []
    sess = _session()
    if sess is None:
        return []
    h, w = bgr.shape[:2]
    canvas, scale, left, top = _letterbox(bgr)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    inp = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    inp = np.expand_dims(inp, 0)
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: inp})[0]
    preds = _parse_output(out)
    boxes: list[tuple[int, int, int, int, float]] = []
    for row in preds:
        x1, y1, x2, y2, score = (float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]))
        if score < conf:
            continue
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
            x1, x2 = x1 * INPUT_SIZE, x2 * INPUT_SIZE
            y1, y2 = y1 * INPUT_SIZE, y2 * INPUT_SIZE
        x1 = (x1 - left) / scale
        x2 = (x2 - left) / scale
        y1 = (y1 - top) / scale
        y2 = (y2 - top) / scale
        bx = int(max(0, min(w - 1, round(min(x1, x2)))))
        by = int(max(0, min(h - 1, round(min(y1, y2)))))
        bw = int(max(8, min(w - bx, round(abs(x2 - x1)))))
        bh = int(max(8, min(h - by, round(abs(y2 - y1)))))
        if bw < 16 or bh < 16:
            continue
        boxes.append((bx, by, bw, bh, score))
    boxes.sort(key=lambda item: item[4], reverse=True)
    return boxes[:40]
