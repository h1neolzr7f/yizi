"""Resume-friendly HTTP download for GGUF / llama zips."""

from __future__ import annotations

import ssl
import sys
import urllib.request
from pathlib import Path


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    ctx = ssl._create_unverified_context()
    got = tmp.stat().st_size if tmp.exists() else 0
    headers = {"User-Agent": "yizi/1.0"}
    if got:
        headers["Range"] = f"bytes={got}-"
        print(f"resume {dest.name} from {got // (1024 * 1024)}MB", flush=True)
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=45, context=ctx) as resp, open(tmp, "ab") as fh:
        status = int(getattr(resp, "status", 200) or 200)
        total = resp.headers.get("Content-Length")
        if got and status == 200:
            fh.seek(0)
            fh.truncate()
            got = 0
            print("server ignored range, restarting", flush=True)
        print(f"status {status} length {total}", flush=True)
        last = got
        while True:
            chunk = resp.read(256 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            if got - last >= 32 * 1024 * 1024:
                print(f"{dest.name} {got // (1024 * 1024)}MB", flush=True)
                last = got
    if not tmp.exists() or tmp.stat().st_size < 1024:
        raise SystemExit(f"incomplete {url}")
    tmp.replace(dest)
    print(f"done {dest} {dest.stat().st_size // (1024 * 1024)}MB", flush=True)


if __name__ == "__main__":
    download(sys.argv[1], Path(sys.argv[2]))
