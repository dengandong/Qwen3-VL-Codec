from __future__ import annotations

from evaluation.VideoMME.visual_flow_probe.metrics import parse_choice


def test_answer_parser_normal_forms() -> None:
    labels = list("ABCD")
    assert parse_choice("A", labels)[0] == "A"
    assert parse_choice("(B)", labels)[0] == "B"
    assert parse_choice("Answer: C", labels)[0] == "C"
    assert parse_choice("The final answer is D.", labels)[0] == "D"


def test_answer_parser_invalid_and_ambiguous() -> None:
    labels = list("ABCD")
    assert parse_choice("I cannot decide", labels)[0] is None
    choice, status = parse_choice("Answer: A or B", labels)
    assert choice is None
    assert "ambiguous" in status


def test_answer_parser_avoids_letters_inside_words() -> None:
    assert parse_choice("Because candles appear often", list("ABCD"))[0] is None
