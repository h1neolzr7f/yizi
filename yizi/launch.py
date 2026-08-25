"""桌面启动器：已在跑就直接打开浏览器，否则后台拉起服务。"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import webbrowser
from urllib.error import URLError
from urllib.request import urlopen

from yizi.config import DATA, HOST, PORT, ROOT, ensure_data_dirs

HEALTH = f"http://{HOST}:{PORT}/api/health"
HOME = f"http://{HOST}:{PORT}/"


def _alive(timeout: float = 0.35) -> bool:
    try:
        with urlopen(HEALTH, timeout=timeout) as res:
            return res.status == 200
    except (URLError, OSError, TimeoutError):
        return False


def _alert(message: str) -> None:
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, message, "AI漫画编剧大师", 0x10)
            return
        except Exception:
            pass
    print(message, file=sys.stderr)


def _python() -> str:
    exe = sys.executable
    if exe.lower().endswith("pythonw.exe"):
        sibling = os.path.join(os.path.dirname(exe), "python.exe")
        if os.path.isfile(sibling):
            return sibling
    return exe


def _spawn_server() -> None:
    ensure_data_dirs()
    log_path = DATA / "yizi.log"
    flags = 0
    if os.name == "nt":
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n---- launch {time.strftime('%Y-%m-%d %H:%M:%S')} ----\n")
        log.flush()
        subprocess.Popen(
            [_python(), "-m", "yizi"],
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=flags,
            close_fds=True,
        )


def main() -> int:
    if _alive():
        webbrowser.open(HOME)
        return 0
    _spawn_server()
    deadline = time.time() + 25
    while time.time() < deadline:
        if _alive(timeout=0.6):
            webbrowser.open(HOME)
            return 0
        time.sleep(0.25)
    _alert(f"AI漫画编剧大师没有启动起来。看 {DATA / 'yizi.log'}，或再点一次桌面快捷方式。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
