"""分块下载 SSGD。清华云长连接会中途断开，所以按 Range 续传。

在仓库根目录：
    python src/download_ssgd.py
输出：
    data/raw/ssgd/SSGD.zip
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

SHARE = "https://cloud.tsinghua.edu.cn/f/720250d21e1b4887abf7/?dl=1"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "ssgd" / "SSGD.zip"
EXPECTED = 1_062_084_791
CHUNK = 8 * 1024 * 1024
PROXY = os.environ.get("HTTPS_PROXY", "http://127.0.0.1:17897")


def opener():
    return build_opener(ProxyHandler({"http": PROXY, "https": PROXY}))


def file_url() -> str:
    result = subprocess.run(
        [
            "curl.exe",
            "-sI",
            "-L",
            "--max-time",
            "40",
            "-x",
            PROXY,
            SHARE,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    locations = [
        line.split(":", 1)[1].strip()
        for line in result.stdout.splitlines()
        if line.lower().startswith("location:")
    ]
    if not locations:
        raise RuntimeError(result.stdout)
    return locations[-1]


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    have = DEST.stat().st_size if DEST.exists() else 0
    if have > EXPECTED:
        DEST.unlink()
        have = 0
    url = file_url()
    while have < EXPECTED:
        end = min(have + CHUNK - 1, EXPECTED - 1)
        request = Request(url, headers={"Range": f"bytes={have}-{end}"})
        try:
            with opener().open(request, timeout=120) as response:
                data = response.read()
        except Exception as exc:
            print(f"retry at {have}: {exc}", flush=True)
            url = file_url()
            continue
        if not data:
            url = file_url()
            continue
        with DEST.open("ab") as handle:
            handle.write(data)
        have += len(data)
        print(f"{have}/{EXPECTED}", flush=True)
    print(DEST)


if __name__ == "__main__":
    main()
