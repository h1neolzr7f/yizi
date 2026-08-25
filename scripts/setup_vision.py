"""下载 llama.cpp 与 Qwen3-VL GGUF。图仍然只在本机推理。"""

from __future__ import annotations

import argparse
import sys

from yizi.vision_local import ensure_backend


def main() -> int:
    parser = argparse.ArgumentParser(description="准备本机视觉")
    parser.add_argument("--backend", default="qwen3vl_4b", choices=["qwen3vl_4b", "qwen3vl_2b"])
    args = parser.parse_args()
    info = ensure_backend(args.backend)
    print(info["note"])
    print(info["exe"])
    print(info["llm"])
    print(info["mmproj"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
