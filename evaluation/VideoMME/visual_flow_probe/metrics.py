from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def parse_choice(text: str, valid_labels: Sequence[str]) -> tuple[str | None, str]:
    labels = [str(x).upper() for x in valid_labels]
    if not labels:
        labels = list("ABCDE")
    label_re = "|".join(re.escape(x) for x in labels)
    s = (text or "").strip()
    standalone_all = [m.upper() for m in re.findall(rf"(?<![A-Za-z])({label_re})(?![A-Za-z])", s)]
    if len(set(standalone_all)) > 1 and re.search(r"(?i)\bor\b|/", s):
        return None, f"ambiguous:{sorted(set(standalone_all))}"
    patterns = [
        rf"(?i)(?:final\s+answer|answer|the\s+answer\s+is)\s*[:：]?\s*\(?\s*({label_re})\s*\)?\b",
        rf"^\(?\s*({label_re})\s*\)?[\.。\s]*$",
        rf"(?i)\boption\s*\(?\s*({label_re})\s*\)?\b",
    ]
    matches: list[str] = []
    for pat in patterns:
        found = [m.upper() for m in re.findall(pat, s)]
        if found:
            matches.extend(found)
            break
    if not matches:
        # Last resort: standalone label near the end, not embedded in words.
        found = standalone_all
        if found:
            matches = found[-1:]
    if not matches:
        return None, "no_valid_choice"
    unique = sorted(set(matches))
    if len(unique) > 1:
        return None, f"ambiguous:{unique}"
    return unique[0], "ok"


def paired_bootstrap_ci(
    diffs: Sequence[float],
    *,
    num_resamples: int = 2000,
    seed: int = 3407,
    alpha: float = 0.05,
) -> dict[str, float]:
    arr = np.asarray([x for x in diffs if np.isfinite(x)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.empty(num_resamples, dtype=np.float64)
    for i in range(num_resamples):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = float(sample.mean())
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "ci_low": float(np.percentile(means, 100 * alpha / 2)),
        "ci_high": float(np.percentile(means, 100 * (1 - alpha / 2))),
    }


def summarize_results(
    records: Iterable[dict[str, Any]],
    *,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 3407,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Aggregate intervention rows, averaging random reps per sample first."""
    condition_rows: list[dict[str, Any]] = []
    sample_condition: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec.get("skip"):
            continue
        qid = rec.get("question_id")
        for row in rec.get("interventions", []):
            key = (
                qid,
                row.get("target_mode"),
                row.get("score_type"),
                float(row.get("ratio")),
                row.get("condition"),
                rec.get("baseline_correct"),
                rec.get("duration"),
                rec.get("task_category"),
            )
            sample_condition[key].append(row)

    averaged: list[dict[str, Any]] = []
    for key, rows in sample_condition.items():
        drops = [float(r.get("logprob_drop", float("nan"))) for r in rows]
        changes = [1.0 if r.get("answer_changed") else 0.0 for r in rows]
        c2w = [1.0 if r.get("correct_to_wrong") else 0.0 for r in rows]
        qid, target_mode, score_type, ratio, condition, baseline_correct, duration, task_category = key
        averaged.append(
            {
                "question_id": qid,
                "target_mode": target_mode,
                "score_type": score_type,
                "ratio": ratio,
                "condition": condition,
                "baseline_correct": baseline_correct,
                "duration": duration,
                "task_category": task_category,
                "logprob_drop": float(np.nanmean(drops)),
                "answer_changed": float(np.nanmean(changes)),
                "correct_to_wrong": float(np.nanmean(c2w)),
            }
        )

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in averaged:
        grouped[(row["target_mode"], row["score_type"], row["ratio"], row["condition"])].append(row)

    csv_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"groups": [], "primary_comparisons": {}}
    for key, rows in sorted(grouped.items(), key=str):
        target_mode, score_type, ratio, condition = key
        drops = np.asarray([r["logprob_drop"] for r in rows], dtype=np.float64)
        changes = np.asarray([r["answer_changed"] for r in rows], dtype=np.float64)
        out = {
            "target_mode": target_mode,
            "score_type": score_type,
            "ratio": ratio,
            "condition": condition,
            "n": int(len(rows)),
            "mean_logprob_drop": float(np.nanmean(drops)) if drops.size else float("nan"),
            "mean_answer_changed": float(np.nanmean(changes)) if changes.size else float("nan"),
        }
        csv_rows.append(out)
        summary["groups"].append(out)

    def _paired_diff(cond_a: str, cond_b: str, metric: str, *, target_mode: str, score_type: str, ratio: float) -> dict[str, float]:
        by_q: dict[Any, dict[str, float]] = defaultdict(dict)
        for row in averaged:
            if row["target_mode"] == target_mode and row["score_type"] == score_type and row["ratio"] == ratio:
                by_q[row["question_id"]][row["condition"]] = float(row[metric])
        diffs = []
        for vals in by_q.values():
            if cond_a in vals and cond_b in vals:
                diffs.append(vals[cond_a] - vals[cond_b])
        return paired_bootstrap_ci(diffs, num_resamples=bootstrap_resamples, seed=bootstrap_seed)

    comparison_keys = sorted(
        {(row["target_mode"], row["score_type"], row["ratio"]) for row in averaged},
        key=str,
    )
    summary["paired_comparisons_by_group"] = []
    for target_mode, score_type, ratio in comparison_keys:
        summary["paired_comparisons_by_group"].append(
            {
                "target_mode": target_mode,
                "score_type": score_type,
                "ratio": ratio,
                "top_minus_random_temporal_matched_logprob_drop": _paired_diff(
                    "top",
                    "random_temporal_matched",
                    "logprob_drop",
                    target_mode=target_mode,
                    score_type=score_type,
                    ratio=ratio,
                ),
                "top_minus_bottom_logprob_drop": _paired_diff(
                    "top",
                    "bottom",
                    "logprob_drop",
                    target_mode=target_mode,
                    score_type=score_type,
                    ratio=ratio,
                ),
                "top_minus_random_temporal_matched_answer_change": _paired_diff(
                    "top",
                    "random_temporal_matched",
                    "answer_changed",
                    target_mode=target_mode,
                    score_type=score_type,
                    ratio=ratio,
                ),
                "top_minus_bottom_answer_change": _paired_diff(
                    "top",
                    "bottom",
                    "answer_changed",
                    target_mode=target_mode,
                    score_type=score_type,
                    ratio=ratio,
                ),
            }
        )

    primary_ratio = 0.20
    primary_note = "pre_registered_0.20"
    available_primary_ratios = sorted(
        {float(row["ratio"]) for row in averaged if row["target_mode"] == "decision" and row["score_type"] == "responsibility"}
    )
    if primary_ratio not in available_primary_ratios and available_primary_ratios:
        primary_ratio = available_primary_ratios[0]
        primary_note = "fallback_first_available_ratio"
    summary["primary_comparisons"] = {
        "ratio": primary_ratio,
        "ratio_note": primary_note,
        "top_minus_random_temporal_matched_logprob_drop": _paired_diff(
            "top",
            "random_temporal_matched",
            "logprob_drop",
            target_mode="decision",
            score_type="responsibility",
            ratio=primary_ratio,
        ),
        "top_minus_bottom_logprob_drop": _paired_diff(
            "top",
            "bottom",
            "logprob_drop",
            target_mode="decision",
            score_type="responsibility",
            ratio=primary_ratio,
        ),
        "top_minus_random_temporal_matched_answer_change": _paired_diff(
            "top",
            "random_temporal_matched",
            "answer_changed",
            target_mode="decision",
            score_type="responsibility",
            ratio=primary_ratio,
        ),
        "top_minus_bottom_answer_change": _paired_diff(
            "top",
            "bottom",
            "answer_changed",
            target_mode="decision",
            score_type="responsibility",
            ratio=primary_ratio,
        ),
    }
    return summary, csv_rows


def write_summary_files(records: list[dict[str, Any]], output_dir: Path, *, resamples: int, seed: int) -> None:
    summary, rows = summarize_results(records, bootstrap_resamples=resamples, bootstrap_seed=seed)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    csv_path = output_dir / "summary.csv"
    fieldnames = [
        "target_mode",
        "score_type",
        "ratio",
        "condition",
        "n",
        "mean_logprob_drop",
        "mean_answer_changed",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
