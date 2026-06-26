from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from visual_flow_probe.io_utils import atomic_write_json, save_npz_atomic  # type: ignore
else:
    from .io_utils import atomic_write_json, save_npz_atomic


def _read_records(manifest_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with manifest_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("skip"):
                records.append(rec)
    return records


def _safe_token_count_name(value: int) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(int(value)))


def aggregate_layer_matrix_dump(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    records = _read_records(manifest_path)
    if not records:
        raise RuntimeError(f"no valid records in {manifest_path}")

    shapes = [tuple(int(x) for x in rec["responsibility_shape"]) for rec in records]
    layer_counts = {shape[0] for shape in shapes}
    if len(layer_counts) != 1:
        raise RuntimeError(f"layer count mismatch in {manifest_path}: {sorted(layer_counts)}")
    num_layers = next(iter(layer_counts))
    max_visual_tokens = max(shape[1] for shape in shapes)

    responsibility = np.full(
        (len(records), num_layers, max_visual_tokens),
        np.nan,
        dtype=np.float32,
    )
    valid_mask = np.zeros((len(records), max_visual_tokens), dtype=np.bool_)
    question_ids: list[str] = []
    prompt_lengths: list[int] = []
    sequence_lengths: list[int] = []
    visual_token_counts: list[int] = []
    baseline_choices: list[str] = []
    ground_truths: list[str] = []
    response_shapes = np.asarray(shapes, dtype=np.int64)
    layers: np.ndarray | None = None
    visual_local_indices = np.arange(max_visual_tokens, dtype=np.int64)

    by_count: Counter[int] = Counter()
    matrices_by_count: dict[int, list[np.ndarray]] = {}
    record_indices_by_count: dict[int, list[int]] = {}

    for row_idx, rec in enumerate(records):
        npz_path = Path(rec["npz"])
        if not npz_path.is_absolute():
            npz_path = Path.cwd() / npz_path
        with np.load(npz_path) as arr:
            matrix = np.asarray(arr["responsibility"], dtype=np.float32)
            if layers is None:
                layers = np.asarray(arr["layers"], dtype=np.int64)
        if matrix.shape[0] != num_layers:
            raise RuntimeError(f"layer mismatch in {npz_path}: {matrix.shape}")
        n_tokens = int(matrix.shape[1])
        responsibility[row_idx, :, :n_tokens] = matrix
        valid_mask[row_idx, :n_tokens] = True
        by_count[n_tokens] += 1
        matrices_by_count.setdefault(n_tokens, []).append(matrix)
        record_indices_by_count.setdefault(n_tokens, []).append(row_idx)

        question_ids.append(str(rec.get("question_id", "")))
        prompt_lengths.append(int(rec.get("prompt_length", -1)))
        sequence_lengths.append(int(rec.get("sequence_length", -1)))
        visual_token_counts.append(int(rec.get("visual_token_count", n_tokens)))
        baseline_choices.append("" if rec.get("baseline_choice") is None else str(rec.get("baseline_choice")))
        ground_truths.append("" if rec.get("ground_truth") is None else str(rec.get("ground_truth")))

    if layers is None:
        layers = np.arange(num_layers, dtype=np.int64)

    all_padded_path = output_dir / "all_layer_responsibility_padded.npz"
    save_npz_atomic(
        all_padded_path,
        responsibility=responsibility,
        valid_visual_token_mask=valid_mask,
        question_ids=np.asarray(question_ids),
        layers=layers,
        visual_local_indices=visual_local_indices,
        visual_token_counts=np.asarray(visual_token_counts, dtype=np.int64),
        prompt_lengths=np.asarray(prompt_lengths, dtype=np.int64),
        sequence_lengths=np.asarray(sequence_lengths, dtype=np.int64),
        baseline_choices=np.asarray(baseline_choices),
        ground_truths=np.asarray(ground_truths),
        responsibility_shapes=response_shapes,
    )

    common_count, common_n = by_count.most_common(1)[0]
    common_path = output_dir / f"all_layer_responsibility_{_safe_token_count_name(common_count)}only.npz"
    common_rows = record_indices_by_count[common_count]
    save_npz_atomic(
        common_path,
        responsibility=np.stack(matrices_by_count[common_count], axis=0).astype(np.float32, copy=False),
        question_ids=np.asarray([question_ids[i] for i in common_rows]),
        layers=layers,
        visual_local_indices=np.arange(common_count, dtype=np.int64),
        visual_token_counts=np.asarray([visual_token_counts[i] for i in common_rows], dtype=np.int64),
        prompt_lengths=np.asarray([prompt_lengths[i] for i in common_rows], dtype=np.int64),
        sequence_lengths=np.asarray([sequence_lengths[i] for i in common_rows], dtype=np.int64),
        baseline_choices=np.asarray([baseline_choices[i] for i in common_rows]),
        ground_truths=np.asarray([ground_truths[i] for i in common_rows]),
    )

    summary = {
        "num_records": len(records),
        "all_padded_npz": str(all_padded_path),
        "all_padded_shape": list(responsibility.shape),
        "valid_visual_token_mask_shape": list(valid_mask.shape),
        "padding_value": "NaN beyond valid_visual_token_mask",
        "common_npz": str(common_path),
        "common_shape": [common_n, num_layers, common_count],
        "common_visual_token_count": int(common_count),
        "num_common": int(common_n),
        "visual_token_count_distribution": {str(k): int(v) for k, v in sorted(by_count.items())},
    }
    atomic_write_json(output_dir / "posthoc_aggregate_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Posthoc aggregate per-sample VFlow layer responsibility matrices")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = aggregate_layer_matrix_dump(Path(args.output_dir))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
