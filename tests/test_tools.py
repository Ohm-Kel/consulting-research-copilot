import pytest

from copilot.tools import ToolError, calculate


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", 14),
        ("(3219 - 5700) / 5700 * 100", -43.526),
        ("pct_change(5700, 3219)", -43.526),
        ("pct_change(1550190, 1814616)", 17.058),
        ("share(2233090, 4985612)", 44.790),
        ("round(cagr(100, 121, 2), 2)", 10.0),
        ("$4,689 - 4689", 0),  # $ and thousands separators are stripped
        ("pct_change(1,550,190, 1,814,616)", 17.058),  # ...but argument commas are kept
        ("pct_change(5700,3219)", -43.526),
        ("2 ** 10", 1024),
    ],
)
def test_calculate(expression: str, expected: float) -> None:
    assert calculate(expression) == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",
        "open('x')",
        "a + 1",
        "(1).real",
        "1 / 0",
        "2 ** 1000",
        "((10 ** 100) ** 100) ** 100",  # would build a gigantic integer without float arithmetic
        "1e308 * 10",
        "sqrt(-1)",
        "abs((-8) ** 0.5)",
        "round(1, 2, 3)",
        "min()",
        "True + 1",
        "pct_change(0, 5)",
        "1 +",
        "[1, 2]",
        "1" + "+1" * 200,
        "(" * 500 + "1" + ")" * 500,
    ],
)
def test_calculate_rejects_unsafe_or_invalid_input(expression: str) -> None:
    with pytest.raises(ToolError):
        calculate(expression)
