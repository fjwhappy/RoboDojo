#!/usr/bin/env python3
"""Download a published RoboDojo policy checkpoint into a persistent cache (idempotent).

Files come from the Hugging Face dataset ``RoboDojo-Benchmark/RoboDojo`` under
``ckpt/RoboDojo/<policy>/<run>/`` over plain HTTPS. This works with HF mirrors such as
``https://hf-mirror.com`` and needs no git-lfs. Each file is checked against the sha256 the
Hub reports, downloads resume after interruption, and a marker file makes later runs return
immediately.

By default only what inference needs is fetched: everything except ``train_state/``
(optimizer state, ~half the size). Pass ``--include-train-state`` to get it all.

Usage:
  python3 scripts/ci/download_checkpoint.py --policy Pi_05 --run RoboDojo-sim-arx_x5-joint-0 \
      --dest /cache/checkpoints/Pi_05 [--endpoint https://hf-mirror.com] [--check-only]
Result: <dest>/<run>/... and <dest>/<run>/.robodojo_ckpt_complete
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

# Some HF mirrors reject the default Python-urllib User-Agent.
USER_AGENT = "robodojo-ci/1.0"


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


REPO_ID = "RoboDojo-Benchmark/RoboDojo"
MARKER = ".robodojo_ckpt_complete"


def log(msg: str) -> None:
    print(f"[download_checkpoint] {msg}", flush=True)


def list_files(endpoint: str, revision: str, prefix: str) -> list[dict]:
    url = f"{endpoint}/api/datasets/{REPO_ID}/tree/{revision}/{urllib.parse.quote(prefix)}?recursive=true"
    for attempt in range(6):
        try:
            with urllib.request.urlopen(_request(url), timeout=60) as resp:
                entries = json.load(resp)
            return [e for e in entries if e.get("type") == "file"]
        except Exception as exc:  # noqa: BLE001
            log(f"listing failed ({exc}); retry {attempt + 1}/6")
            time.sleep(5 * (attempt + 1))
    raise SystemExit(f"cannot list {url}")


def expected_sha256(entry: dict) -> str | None:
    lfs = entry.get("lfs") or {}
    return lfs.get("oid") or lfs.get("sha256")


def file_ok(path: Path, entry: dict) -> bool:
    return path.is_file() and path.stat().st_size == int(entry.get("size", -1))


def fetch(url: str, dest: Path, entry: dict, retries: int = 8) -> str | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    sha = expected_sha256(entry)
    err = "unknown error"
    for attempt in range(retries):
        try:
            h = hashlib.sha256()
            with urllib.request.urlopen(_request(url), timeout=120) as resp, open(tmp, "wb") as fh:
                while chunk := resp.read(4 << 20):
                    h.update(chunk)
                    fh.write(chunk)
            if tmp.stat().st_size != int(entry.get("size", -1)):
                raise ValueError(f"size mismatch ({tmp.stat().st_size} != {entry.get('size')})")
            if sha and h.hexdigest() != sha:
                raise ValueError("sha256 mismatch")
            os.replace(tmp, dest)
            return None
        except Exception as exc:  # noqa: BLE001 - retried, reported at the end
            err = str(exc)
            time.sleep(min(60, 5 * (attempt + 1)))
    tmp.unlink(missing_ok=True)
    return f"{dest}: {err}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="Pi_05", help="Remote folder under ckpt/RoboDojo/")
    ap.add_argument("--run", required=True, help="Checkpoint run directory, e.g. RoboDojo-sim-arx_x5-joint-0")
    ap.add_argument("--dest", required=True, help="Local dir; files land in <dest>/<run>/")
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    ap.add_argument("--revision", default="main")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--include-train-state", action="store_true")
    ap.add_argument("--check-only", action="store_true", help="Exit 0 if complete, 1 otherwise")
    args = ap.parse_args()

    run_dir = Path(args.dest).expanduser().resolve() / args.run
    marker = run_dir / MARKER
    if marker.is_file():
        log(f"cache hit: {run_dir} is complete; skipping download")
        return 0
    if args.check_only:
        log(f"cache miss: {run_dir}")
        return 1

    endpoint = args.endpoint.rstrip("/")
    prefix = f"ckpt/RoboDojo/{args.policy}/{args.run}"
    entries = list_files(endpoint, args.revision, prefix)
    if not args.include_train_state:
        entries = [e for e in entries if "/train_state/" not in e["path"]]
    if not entries:
        log(f"no files under {prefix} on {endpoint}")
        return 2

    def local(entry: dict) -> Path:
        return run_dir / entry["path"][len(prefix) + 1 :]

    todo = [e for e in entries if not file_ok(local(e), e)]
    total_gb = sum(int(e.get("size", 0)) for e in entries) / 1e9
    todo_gb = sum(int(e.get("size", 0)) for e in todo) / 1e9
    log(f"{len(entries)} files ({total_gb:.1f} GB); {len(todo)} to download ({todo_gb:.1f} GB) -> {run_dir}")

    base = f"{endpoint}/datasets/{REPO_ID}/resolve/{args.revision}/"
    errors: list[str] = []
    done = 0
    with cf.ThreadPoolExecutor(max(1, args.workers)) as pool:
        futures = {pool.submit(fetch, base + urllib.parse.quote(e["path"]), local(e), e): e for e in todo}
        for fut in cf.as_completed(futures):
            done += 1
            if res := fut.result():
                errors.append(res)
            log(f"{done}/{len(todo)} files, {len(errors)} error(s)")

    if errors:
        for e in errors[:20]:
            log(f"ERROR {e}")
        log("incomplete; re-run to resume")
        return 1
    marker.write_text(
        json.dumps({"endpoint": endpoint, "revision": args.revision, "prefix": prefix, "files": len(entries)}),
        encoding="utf-8",
    )
    log(f"checkpoint ready: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
