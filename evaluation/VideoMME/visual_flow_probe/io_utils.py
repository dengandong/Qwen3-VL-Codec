from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def add_videomme_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_videomme_samples(data_path: str, *, duration: str | None = None) -> list[dict[str, Any]]:
    path = Path(data_path)
    if path.suffix == ".parquet" and path.exists():
        import pandas as pd

        rows = pd.read_parquet(path).to_dict("records")
        if duration is not None:
            rows = [row for row in rows if row.get("duration") == duration]
        return rows

    add_videomme_to_path()
    from dataset_utils import load_videomme_dataset

    if duration is not None:
        return list(load_videomme_dataset(data_path, duration=duration))
    out: list[dict[str, Any]] = []
    for dur in ("short", "medium", "long"):
        out.extend(load_videomme_dataset(data_path, duration=dur))
    return out


def build_prompt(sample: dict[str, Any], args: Any):
    add_videomme_to_path()
    from dataset_utils import build_videomme_prompt

    return build_videomme_prompt(
        sample,
        args.data_path,
        use_subtitle=args.use_subtitle,
        fps=args.fps,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        total_pixels=args.total_pixels,
        video_dir=args.video_dir,
    )


def completed_question_ids(results_path: Path) -> set[str]:
    done: set[str] = set()
    if not results_path.exists():
        return done
    with results_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = rec.get("question_id")
            if qid is not None:
                done.add(str(qid))
    return done


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(path)


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    normalized = {}
    for key, value in arrays.items():
        if isinstance(value, torch.Tensor):
            normalized[key] = value.detach().cpu().numpy()
        else:
            normalized[key] = np.asarray(value)
    with tmp.open("wb") as f:
        np.savez_compressed(f, **normalized)
    tmp.replace(path)


def git_info(repo_root: Path) -> dict[str, Any]:
    def run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, cwd=repo_root, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    status = run(["git", "status", "--short"])
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_short": status,
    }
