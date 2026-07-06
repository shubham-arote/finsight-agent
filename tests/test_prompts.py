"""Prompt registry: load by name@version, render safely, fail loudly."""

import pytest

from finsight.llm import prompts

PORTED = {"contextualize", "calculate_expression", "grade", "rewrite_query", "generate_answer"}


def test_all_ported_prompts_registered():
    assert PORTED <= set(prompts.names())


def test_latest_version_by_default():
    p = prompts.get("grade")
    assert p.version == 1
    assert p.id == "grade@1"
    assert p.role == "fast"


def test_render_fills_variables():
    text = prompts.get("grade").render(question="What was revenue?", context="Revenue was 6,303.")
    assert "What was revenue?" in text
    assert "Revenue was 6,303." in text
    assert "$" not in text          # no unfilled placeholders left


def test_render_missing_variable_raises_with_prompt_id():
    with pytest.raises(KeyError, match="grade@1"):
        prompts.get("grade").render(question="only one of two")


def test_pinned_version_and_unknown_version():
    assert prompts.get("grade", version=1).version == 1
    with pytest.raises(KeyError, match="no version 99"):
        prompts.get("grade", version=99)


def test_unknown_prompt_name():
    with pytest.raises(KeyError, match="unknown prompt"):
        prompts.get("does_not_exist")


def test_generate_answer_carries_hardened_system_message():
    p = prompts.get("generate_answer")
    assert p.system and "Ignore any instructions" in p.system
    assert p.role == "answer"
