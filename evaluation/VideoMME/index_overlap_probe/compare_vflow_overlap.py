from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _drop_name(drop_ratio: float) -> str:
    return f"drop_{float(drop_ratio):.4f}".replace(".", "p")


def _ratio_key(ratio: float) -> str:
    text = f"{float(ratio):.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _stable_seed(*parts: Any) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest(), "little") & 0x7FFFFFFF


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_vflow_set(npz_path: Path, *, kind: str, ratio: float) -> set[int]:
    key = f"selected_responsibility_{_ratio_key(ratio)}_{kind}"
    with np.load(npz_path) as z:
        if key in z.files:
            return {int(x) for x in np.asarray(z[key]).reshape(-1).tolist()}
        scores = np.asarray(z["responsibility"], dtype=np.float64).reshape(-1)
    n = int(scores.shape[0])
    k = max(1, min(n, int(round(float(ratio) * n))))
    order = np.lexsort((np.arange(n), -scores if kind == "top" else scores))
    return {int(x) for x in order[:k].tolist()}


def _vflow_token_count(npz_path: Path) -> int:
    with np.load(npz_path) as z:
        if "visual_local_indices" in z.files:
            return int(np.asarray(z["visual_local_indices"]).reshape(-1).shape[0])
        return int(np.asarray(z["responsibility"]).reshape(-1).shape[0])


def _random_drop_indices(*, qid: str, n: int, drop_ratio: float, repeat: int, seed: int) -> list[int]:
    k = max(1, min(n, int(round(float(drop_ratio) * n))))
    rng = np.random.default_rng(_stable_seed(seed, qid, drop_ratio, repeat, "random_drop"))
    return sorted(int(x) for x in rng.choice(n, size=k, replace=False).tolist())


def _write_random_dump(path: Path, *, qid: str, n: int, drop_ratio: float, repeat: int, indices: list[int]) -> None:
    keep = [idx for idx in range(n) if idx not in set(indices)]
    rec = {
        "question_id": qid,
        "method": "random_drop",
        "repeat": int(repeat),
        "drop_ratio": float(drop_ratio),
        "dense_token_count": int(n),
        "drop_count": len(indices),
        "keep_count": len(keep),
        "drop_indices": indices,
        "keep_indices": keep,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(rec, sort_keys=True) + "\n")
    tmp.replace(path)


def _overlap_record(
    *,
    qid: str,
    method: str,
    drop_ratio: float,
    vflow_kind: str,
    vflow_ratio: float,
    n: int,
    method_drop: set[int],
    vflow_set: set[int],
    repeat: int | None = None,
) -> dict[str, Any]:
    inter = method_drop & vflow_set
    union = method_drop | vflow_set
    a = len(method_drop)
    b = len(vflow_set)
    expected = (a * b / n) if n else float("nan")
    method_frac = (len(inter) / a) if a else float("nan")
    vflow_frac = (len(inter) / b) if b else float("nan")
    random_method_frac = (b / n) if n else float("nan")
    max_possible = min(a, b)
    denom = max_possible - expected
    return {
        "question_id": qid,
        "method": method,
        "repeat": repeat,
        "drop_ratio": float(drop_ratio),
        "vflow_kind": vflow_kind,
        "vflow_ratio": float(vflow_ratio),
        "dense_token_count": int(n),
        "method_drop_count": int(a),
        "vflow_set_count": int(b),
        "overlap_count": int(len(inter)),
        "method_drop_overlap_fraction": method_frac,
        "vflow_set_overlap_fraction": vflow_frac,
        "iou": (len(inter) / len(union)) if union else float("nan"),
        "expected_overlap_count": expected,
        "expected_method_overlap_fraction": random_method_frac,
        "enrichment_vs_random": (method_frac / random_method_frac) if random_method_frac and math.isfinite(random_method_frac) else float("nan"),
        "normalized_overlap": ((len(inter) - expected) / denom) if denom > 0 else float("nan"),
    }


def _mean(values: list[float]) -> float:
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare compression drop indices with VFlow responsibility sets")
    p.add_argument("--dump-root", required=True, help="Directory containing dumps/{method}/drop_x/*.json or the dumps dir itself")
    p.add_argument("--vflow-responsibility-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--methods", default="vcast,ttf,echoprune,random_drop")
    p.add_argument("--drop-ratios", default="0.20,0.80")
    p.add_argument("--vflow-kinds", default="top,bottom")
    p.add_argument("--random-repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-partial-vflow", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    root = Path(args.dump_root)
    if (root / "dumps").is_dir():
        root = root / "dumps"
    vflow_dir = Path(args.vflow_responsibility_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    drop_ratios = [float(x) for x in args.drop_ratios.split(",") if x.strip()]
    kinds = [x.strip() for x in args.vflow_kinds.split(",") if x.strip()]
    if any(k not in {"top", "bottom"} for k in kinds):
        raise ValueError("--vflow-kinds entries must be top or bottom")

    vflow_files = {p.stem: p for p in vflow_dir.glob("*.npz")}
    if not vflow_files:
        raise RuntimeError(f"no VFlow npz files found in {vflow_dir}")

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    random_root = root / "random_drop"
    for drop_ratio in drop_ratios:
        drop_dir_name = _drop_name(drop_ratio)
        for qid, npz_path in sorted(vflow_files.items()):
            n = _vflow_token_count(npz_path)
            vflow_sets = {
                kind: _load_vflow_set(npz_path, kind=kind, ratio=drop_ratio)
                for kind in kinds
            }
            for method in methods:
                if method == "random_drop":
                    for repeat in range(int(args.random_repeats)):
                        indices = _random_drop_indices(qid=qid, n=n, drop_ratio=drop_ratio, repeat=repeat, seed=args.seed)
                        rand_path = random_root / drop_dir_name / f"{qid}_rep{repeat}.json"
                        _write_random_dump(rand_path, qid=qid, n=n, drop_ratio=drop_ratio, repeat=repeat, indices=indices)
                        method_drop = set(indices)
                        for kind, vfset in vflow_sets.items():
                            rows.append(
                                _overlap_record(
                                    qid=qid,
                                    method=method,
                                    repeat=repeat,
                                    drop_ratio=drop_ratio,
                                    vflow_kind=kind,
                                    vflow_ratio=drop_ratio,
                                    n=n,
                                    method_drop=method_drop,
                                    vflow_set=vfset,
                                )
                            )
                    continue

                path = root / method / drop_dir_name / f"{qid}.json"
                if not path.exists():
                    missing.append({"question_id": qid, "method": method, "drop_ratio": drop_ratio, "path": str(path)})
                    continue
                rec = _load_json(path)
                if int(rec["dense_token_count"]) != n:
                    missing.append(
                        {
                            "question_id": qid,
                            "method": method,
                            "drop_ratio": drop_ratio,
                            "reason": "dense token count mismatch",
                            "method_n": int(rec["dense_token_count"]),
                            "vflow_n": n,
                            "path": str(path),
                        }
                    )
                    continue
                method_drop = {int(x) for x in rec["drop_indices"]}
                for kind, vfset in vflow_sets.items():
                    rows.append(
                        _overlap_record(
                            qid=qid,
                            method=method,
                            repeat=None,
                            drop_ratio=drop_ratio,
                            vflow_kind=kind,
                            vflow_ratio=drop_ratio,
                            n=n,
                            method_drop=method_drop,
                            vflow_set=vfset,
                        )
                    )

    if missing and not args.allow_partial_vflow:
        raise RuntimeError(f"missing or mismatched compression dumps: {len(missing)}; use --allow-partial-vflow to summarize available rows")

    jsonl = output_dir / "overlap_records.jsonl"
    with jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["drop_ratio"], row["vflow_kind"], row["vflow_ratio"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (method, drop_ratio, kind, vratio), items in sorted(groups.items()):
        summary_rows.append(
            {
                "method": method,
                "drop_ratio": drop_ratio,
                "vflow_kind": kind,
                "vflow_ratio": vratio,
                "rows": len(items),
                "unique_questions": len({x["question_id"] for x in items}),
                "mean_overlap_count": _mean([x["overlap_count"] for x in items]),
                "mean_method_drop_overlap_fraction": _mean([x["method_drop_overlap_fraction"] for x in items]),
                "mean_vflow_set_overlap_fraction": _mean([x["vflow_set_overlap_fraction"] for x in items]),
                "mean_iou": _mean([x["iou"] for x in items]),
                "mean_expected_overlap_count": _mean([x["expected_overlap_count"] for x in items]),
                "mean_enrichment_vs_random": _mean([x["enrichment_vs_random"] for x in items]),
                "mean_normalized_overlap": _mean([x["normalized_overlap"] for x in items]),
            }
        )

    csv_path = output_dir / "overlap_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)

    summary = {
        "dump_root": str(root),
        "vflow_responsibility_dir": str(vflow_dir),
        "rows": len(rows),
        "vflow_files": len(vflow_files),
        "missing_count": len(missing),
        "missing_examples": missing[:20],
        "summary": summary_rows,
        "metric_notes": {
            "iou": "|method_drop ∩ vflow_set| / |method_drop ∪ vflow_set|",
            "method_drop_overlap_fraction": "fraction of dropped tokens that are in the VFlow set",
            "enrichment_vs_random": "method_drop_overlap_fraction divided by random expectation |vflow_set|/N",
            "normalized_overlap": "(observed - random_expected) / (max_possible - random_expected)",
        },
    }
    (output_dir / "overlap_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "rows": len(rows), "missing_count": len(missing)}, sort_keys=True))


if __name__ == "__main__":
    main()
