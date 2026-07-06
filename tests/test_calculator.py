"""Calculator: correctness + the security gate (it parses untrusted LLM output)."""

import pytest

from finsight.agent import CalcError, extract_expression, is_math_query, safe_eval


# ── correctness ─────────────────────────────────────────────────────────────
def test_exact_arithmetic():
    assert safe_eval("(1052-985)/985*100") == pytest.approx(6.8020, abs=1e-3)
    assert safe_eval("2302/6303*100") == pytest.approx(36.522, abs=1e-3)


def test_thousands_separators_stripped():
    assert safe_eval("(1,052-985)/985*100") == pytest.approx(6.8020, abs=1e-3)


def test_whitelisted_functions():
    assert safe_eval("round(abs(-6.802), 2)") == 6.8
    assert safe_eval("max(1, 2) + min(3, 4)") == 5


# ── security: must reject anything that isn't pure arithmetic ───────────────
@pytest.mark.parametrize("evil", [
    "__import__('os').system('dir')",
    "open('secret.txt')",
    "a + 1",
    "().__class__.__bases__",
    "[1,2][0]",
    "{'a': 1}['a']",
    "(lambda: 1)()",
    "9**9**9",                       # resource bomb
    "True + 1",                      # bool literal
    "'a' * 3",
    "1" * 300,                       # oversized
])
def test_rejects_unsafe_expressions(evil):
    with pytest.raises(CalcError):
        safe_eval(evil)


# ── expression extraction from noisy LLM replies ────────────────────────────
def test_extract_expression_from_noisy_reply():
    assert extract_expression("Sure! (1052-985)/985*100") == "(1052-985)/985*100"
    assert extract_expression("NONE") is None
    assert extract_expression("I cannot compute that.") is None


# ── math-intent routing heuristic ───────────────────────────────────────────
def test_is_math_query():
    assert is_math_query("By how much did revenue grow year on year?")
    assert is_math_query("What was the operating margin in percent?")
    assert not is_math_query("What was revenue in FY26?")
