#!/usr/bin/env python3
"""Download RoboDojo Assets into a persistent cache directory (idempotent).

Layout: ``<cache>/Assets`` inside a sparse git clone ``<cache>`` of the HF dataset
repo. Git LFS is NOT required: the clone keeps LFS pointer files, which are then
replaced by their content downloaded over plain HTTPS from ``<endpoint>/datasets/
<repo>/resolve/<revision>/<path>`` and verified against the pointer's sha256.
This works through HF mirrors (e.g. ``https://hf-mirror.com``) and resumes after
interruption. When everything is in place a marker file is written, and later
runs return immediately.

Usage:
  python3 scripts/ci/download_assets.py --assets-dir /cache/robodojo-assets/Assets \
      [--endpoint https://hf-mirror.com] [--workers 16] [--check-only]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.parse
import urllib.request

REPO_ID = "RoboDojo-Benchmark/RoboDojo"
REQUIRED_SUBDIRS = ("Robots", "Object", "Material", "Eval_Layout")
MARKER = ".robodojo_assets_complete"
POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
NO_LFS = [
    "-c",
    "filter.lfs.smudge=",
    "-c",
    "filter.lfs.process=",
    "-c",
    "filter.lfs.required=false",
]


def log(msg: str) -> None:
    print(f"[download_assets] {msg}", flush=True)


def git(cwd: Path, *args: str) -> None:
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    subprocess.run(["git", *NO_LFS, *args], cwd=cwd, env=env, check=True)


def read_pointer(path: Path) -> tuple[str, int] | None:
    """Return (sha256, size) if ``path`` is a git-lfs pointer file, else None."""
    try:
        if path.is_symlink() or path.stat().st_size > 1024:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if not data.startswith(POINTER_PREFIX):
        return None
    oid = size = None
    for line in data.decode("utf-8", "replace").splitlines():
        if line.startswith("oid sha256:"):
            oid = line.split(":", 1)[1].strip()
        elif line.startswith("size "):
            size = int(line.split()[1])
    return (oid, size) if oid and size is not None else None


def find_pointers(assets: Path) -> list[tuple[Path, str, int]]:
    out = []
    for root, _dirs, files in os.walk(assets):
        for name in files:
            p = Path(root) / name
            ptr = read_pointer(p)
            if ptr:
                out.append((p, *ptr))
    return out


def is_complete(assets: Path) -> bool:
    return (assets.parent / MARKER).is_file() and all((assets / d).is_dir() for d in REQUIRED_SUBDIRS)


def fetch(url: str, dest: Path, oid: str, retries: int = 6) -> str | None:
    tmp = dest.with_name(dest.name + ".part")
    err = "unknown error"
    for attempt in range(retries):
        try:
            h = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
                while chunk := resp.read(1 << 20):
                    h.update(chunk)
                    fh.write(chunk)
            if h.hexdigest() != oid:
                raise ValueError("sha256 mismatch")
            os.replace(tmp, dest)
            return None
        except Exception as exc:  # noqa: BLE001 - retried, reported at the end
            err = f"{exc}"
            time.sleep(min(60, 3 * (attempt + 1)))
    tmp.unlink(missing_ok=True)
    return f"{dest}: {err}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets-dir", required=True, help="Target Assets dir (basename must be 'Assets')")
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    ap.add_argument("--revision", default="main")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--check-only", action="store_true", help="Exit 0 if complete, 1 otherwise; never download")
    args = ap.parse_args()

    assets = Path(args.assets_dir).resolve()
    if assets.name != "Assets":
        log(f"--assets-dir must end with /Assets (got {assets})")
        return 2
    cache = assets.parent
    endpoint = args.endpoint.rstrip("/")

    if is_complete(assets):
        log(f"cache hit: {assets} is complete; skipping download")
        return 0
    if args.check_only:
        log(f"cache miss: {assets} is not complete")
        return 1

    cache.mkdir(parents=True, exist_ok=True)
    if not (cache / ".git").is_dir():
        if any(cache.iterdir()):
            log(f"{cache} exists, is not empty, and is not a git clone; refusing to overwrite")
            return 2
        log(f"cloning {endpoint}/datasets/{REPO_ID} (sparse, pointers only) into {cache}")
        git(
            cache.parent,
            "clone",
            "--depth",
            "1",
            "--sparse",
            "--branch",
            args.revision,
            f"{endpoint}/datasets/{REPO_ID}",
            str(cache),
        )
    git(cache, "sparse-checkout", "set", "Assets")
    if not assets.is_dir():
        git(cache, "checkout", "-f", "HEAD", "--", "Assets")

    pending = find_pointers(assets)
    log(f"{len(pending)} LFS file(s) to download")
    base = f"{endpoint}/datasets/{REPO_ID}/resolve/{args.revision}/"
    errors: list[str] = []
    done = 0

    def job(item: tuple[Path, str, int]) -> str | None:
        path, oid, _size = item
        rel = path.relative_to(cache).as_posix()
        return fetch(base + urllib.parse.quote(rel), path, oid)

    with cf.ThreadPoolExecutor(max(1, args.workers)) as pool:
        for res in pool.map(job, pending):
            done += 1
            if res:
                errors.append(res)
            if done % 500 == 0 or done == len(pending):
                log(f"{done}/{len(pending)} downloaded, {len(errors)} error(s)")

    if errors:
        for e in errors[:20]:
            log(f"ERROR {e}")
        log("incomplete; re-run to resume")
        return 1

    missing = [d for d in REQUIRED_SUBDIRS if not (assets / d).is_dir()]
    if missing:
        log(f"missing required subdirs: {missing}")
        return 1

    # Generate curobo.yml etc. with absolute paths for this location.
    script = Path(__file__).resolve().parents[2] / "utils" / "update_embodiment_config_path.py"
    subprocess.run([sys.executable, str(script)], cwd=cache, check=True)
    (cache / MARKER).write_text(f"endpoint={endpoint}\nrevision={args.revision}\n", encoding="utf-8")
    log(f"assets ready: {assets}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
