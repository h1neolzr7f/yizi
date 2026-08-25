from __future__ import annotations

import cv2
import numpy as np


def manga_fixture_png() -> bytes:
    img = np.full((900, 700, 3), (176, 168, 158), np.uint8)
    cv2.ellipse(img, (260, 560), (150, 230), 0, 0, 360, (36, 42, 150), -1)
    cv2.ellipse(img, (260, 310), (88, 96), 0, 0, 360, (50, 60, 170), -1)
    cv2.ellipse(img, (230, 300), (18, 22), 0, 0, 360, (20, 20, 20), -1)
    cv2.ellipse(img, (290, 300), (18, 22), 0, 0, 360, (20, 20, 20), -1)
    cv2.rectangle(img, (430, 70), (670, 270), (255, 255, 255), -1)
    cv2.rectangle(img, (430, 70), (670, 270), (25, 25, 25), 4)
    cv2.putText(img, "NAI", (470, 190), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (30, 30, 30), 4, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def lettering_fixture_png() -> bytes:
    img = np.full((1216, 832, 3), (214, 204, 190), np.uint8)
    cv2.ellipse(img, (260, 780), (140, 220), 0, 0, 360, (48, 58, 168), -1)
    cv2.ellipse(img, (260, 500), (90, 110), 0, 0, 360, (62, 70, 182), -1)
    cv2.rectangle(img, (40, 80), (130, 640), (252, 252, 248), -1)
    cv2.rectangle(img, (40, 80), (130, 640), (28, 28, 28), 3)
    cv2.putText(img, "I", (70, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (30, 30, 30), 3, cv2.LINE_AA)
    cv2.putText(img, "I", (70, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (30, 30, 30), 3, cv2.LINE_AA)
    cv2.rectangle(img, (470, 70), (790, 300), (255, 255, 255), -1)
    cv2.rectangle(img, (470, 70), (790, 300), (28, 28, 28), 5)
    cv2.putText(img, "YORU", (500, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (30, 30, 30), 4, cv2.LINE_AA)
    cv2.circle(img, (690, 930), 55, (28, 28, 30), -1)
    cv2.putText(img, "DON", (650, 940), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def hard_lettering_fixture_png() -> bytes:
    """偏黄纸、浅色泡、小框、圆角旁白：专门打当前几何检出的漏框。"""
    img = np.full((1216, 832, 3), (188, 176, 164), np.uint8)
    cv2.ellipse(img, (210, 820), (120, 190), 0, 0, 360, (40, 48, 150), -1)
    cv2.ellipse(img, (220, 216), (118, 86), 0, 0, 360, (228, 222, 208), -1)
    cv2.ellipse(img, (220, 216), (118, 86), 0, 0, 360, (32, 32, 32), 3)
    cv2.putText(img, "aa", (176, 228), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (36, 36, 36), 3, cv2.LINE_AA)
    cv2.ellipse(img, (640, 180), (70, 52), 0, 0, 360, (236, 232, 224), -1)
    cv2.ellipse(img, (640, 180), (70, 52), 0, 0, 360, (28, 28, 28), 2)
    cv2.putText(img, "un", (610, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.rectangle(img, (500, 430), (790, 620), (220, 216, 206), -1)
    cv2.rectangle(img, (500, 430), (790, 620), (24, 24, 24), 4)
    cv2.putText(img, "YORU", (530, 540), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (32, 32, 32), 3, cv2.LINE_AA)
    cv2.rectangle(img, (36, 90), (118, 560), (226, 222, 212), -1)
    cv2.rectangle(img, (36, 90), (118, 560), (30, 30, 30), 3)
    cv2.putText(img, "I", (62, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 3, cv2.LINE_AA)
    cv2.circle(img, (700, 980), 48, (214, 210, 204), -1)
    cv2.circle(img, (700, 980), 48, (40, 40, 40), 2)
    cv2.putText(img, "..", (682, 988), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()
