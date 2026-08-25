from __future__ import annotations

import io

from PIL import Image

from yizi.fixtures import hard_lettering_fixture_png, lettering_fixture_png, manga_fixture_png

__all__ = ["hard_lettering_fixture_png", "lettering_fixture_png", "manga_fixture_png", "png_size"]


def png_size(data: bytes) -> tuple[int, int]:
    im = Image.open(io.BytesIO(data))
    return im.size
