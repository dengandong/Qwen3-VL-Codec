from __future__ import annotations

from evaluation.VideoMME.visual_flow_probe.metrics import summarize_results


def test_bootstrap_averages_random_repetitions_per_sample() -> None:
    records = [
        {
            "question_id": "q1",
            "baseline_correct": True,
            "duration": "short",
            "task_category": "x",
            "interventions": [
                {"target_mode": "decision", "score_type": "responsibility", "ratio": 0.20, "condition": "top", "logprob_drop": 3.0, "answer_changed": True, "correct_to_wrong": True},
                {"target_mode": "decision", "score_type": "responsibility", "ratio": 0.20, "condition": "random_temporal_matched", "logprob_drop": 1.0, "answer_changed": False, "correct_to_wrong": False},
                {"target_mode": "decision", "score_type": "responsibility", "ratio": 0.20, "condition": "random_temporal_matched", "logprob_drop": 2.0, "answer_changed": False, "correct_to_wrong": False},
            ],
        }
    ]
    summary, rows = summarize_results(records, bootstrap_resamples=20, bootstrap_seed=0)
    random_row = [r for r in rows if r["condition"] == "random_temporal_matched"][0]
    assert random_row["mean_logprob_drop"] == 1.5
    comp = summary["primary_comparisons"]["top_minus_random_temporal_matched_logprob_drop"]
    assert comp["mean"] == 1.5
