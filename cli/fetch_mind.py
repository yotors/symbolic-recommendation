"""Download the public, streamable MIND-small benchmark projection."""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

from ..paths import DATASET_DIR


URL = "https://huggingface.co/datasets/reczoo/MIND_small_x1/resolve/main/MIND_small_x1.zip"
SHA256 = "5429bae7263208b8e68c1edc31b5a09b8cbea9473df2b1e61d1dc4d8c96586b7"
LICENSE = "https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-license", action="store_true",
                        help="confirm the Microsoft Research License terms")
    parser.add_argument("--output", type=Path,
                        default=DATASET_DIR / "MIND_small_x1.zip")
    args = parser.parse_args()
    if not args.accept_license:
        parser.error(f"read {LICENSE}, then rerun with --accept-license")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and digest(output) == SHA256:
        print(f"Already verified: {output}")
        return
    partial = output.with_suffix(output.suffix + ".part")
    request = urllib.request.Request(URL, headers={"User-Agent": "recommendation-lab/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response, partial.open("wb") as target:
            total = int(response.headers.get("Content-Length", 0)); copied = 0
            while block := response.read(1024 * 1024):
                target.write(block); copied += len(block)
                if copied % (32 * 1024 * 1024) < len(block):
                    suffix = f"/{total // (1024 * 1024)} MiB" if total else " MiB"
                    print(f"Downloaded {copied // (1024 * 1024)}{suffix}", flush=True)
    except Exception:
        print(f"Download interrupted; partial file retained at {partial}", file=sys.stderr)
        raise
    actual = digest(partial)
    if actual != SHA256:
        raise RuntimeError(f"SHA-256 mismatch: expected {SHA256}, got {actual}")
    os.replace(partial, output)
    print(f"Verified MIND-small archive: {output}")


if __name__ == "__main__":
    main()
