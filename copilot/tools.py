"""Tools the agent can call: document retrieval and a safe calculator."""

import ast
import math
import operator
import re
from typing import Any, Callable

from copilot.retrieval import Hit, Retriever


class ToolError(ValueError):
    """Raised for bad tool input; the message is returned to the model."""


# ---------------------------------------------------------------- calculator

def pct_change(old: float, new: float) -> float:
    """Percentage change from old to new, e.g. pct_change(5700, 3219) = -43.53."""
    if old == 0:
        raise ToolError("pct_change: old value is zero")
    return (new - old) / abs(old) * 100


def share(part: float, whole: float) -> float:
    """part as a percentage of whole, e.g. margin = share(operating_income, revenue)."""
    if whole == 0:
        raise ToolError("share: whole is zero")
    return part / whole * 100


def cagr(start: float, end: float, years: float) -> float:
    """Compound annual growth rate in percent."""
    if start <= 0 or years <= 0:
        raise ToolError("cagr: start and years must be positive")
    return ((end / start) ** (1 / years) - 1) * 100


_BINARY: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow,
}
_UNARY: dict[type, Callable[[Any], Any]] = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCTIONS: dict[str, Callable[..., float]] = {
    "pct_change": pct_change, "share": share, "cagr": cagr,
    "round": round, "abs": abs, "min": min, "max": max, "sqrt": math.sqrt,
}


MAX_EXPRESSION_LENGTH = 300


def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression without `eval`: only numbers, + - * / **,
    parentheses and the whitelisted functions above are allowed. All numbers are
    floats, so huge powers overflow immediately instead of building giant integers."""
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ToolError(f"expression longer than {MAX_EXPRESSION_LENGTH} characters")
    try:
        # Drop "$" and thousands separators ("4,689" -> "4689") but keep argument commas.
        cleaned = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", expression.replace("$", ""))
        tree = ast.parse(cleaned, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ToolError(f"cannot parse expression: {expression!r}") from exc

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ToolError("exponent too large")
            if isinstance(node.op, ast.Div) and right == 0:
                raise ToolError("division by zero")
            result = _BINARY[type(node.op)](left, right)
            if isinstance(result, complex):  # e.g. (-8) ** 0.5
                raise ToolError("result is not a real number")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](walk(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS and not node.keywords:
            args = [walk(arg) for arg in node.args]
            if node.func.id == "round" and len(args) == 2:
                args[1] = int(args[1])  # round(x, 2.0) -> round(x, 2)
            return _FUNCTIONS[node.func.id](*args)
        raise ToolError(f"unsupported element in expression: {ast.dump(node)[:60]}")

    try:
        value = float(walk(tree))
    except ToolError:
        raise
    except (ArithmeticError, ValueError, TypeError, RecursionError) as exc:  # overflow, sqrt(-1), bad arg count...
        raise ToolError(f"cannot evaluate {expression!r}: {exc}") from exc
    if not math.isfinite(value):
        raise ToolError(f"result is not a finite number: {expression!r}")
    return value


# ----------------------------------------------------------------- retrieval

def retrieve_documents(retriever: Retriever, query: str, k: int = 5) -> list[Hit]:
    if not query.strip():
        raise ToolError("query must not be empty")
    return retriever.search(query, k=k)


# -------------------------------------------------------- schemas for the LLM

def _function(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    """An OpenAI Responses API function tool with strict argument checking."""
    return {
        "type": "function",
        "name": name,
        "description": description,
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


TOOL_SCHEMAS: list[dict[str, Any]] = [
    _function(
        "retrieve_documents",
        "Search the annual reports of Nike (FY2025), Lululemon (FY2024), Under Armour (FY2025), "
        "Columbia Sportswear (FY2024) and Deckers Brands (FY2025). Returns numbered passages with "
        "source IDs like S3 and a relevance score. Include the company name and specific terms "
        "in the query. If results look weak or off-topic, search again with different wording.",
        {"query": {"type": "string", "description": "Search query, e.g. 'Nike gross margin fiscal 2025 drivers'"}},
    ),
    _function(
        "calculate",
        "Evaluate an arithmetic expression exactly. Use it for every computed figure instead of "
        "mental math. Supports + - * / ** and parentheses, plus pct_change(old, new), "
        "share(part, whole), cagr(start, end, years), round(x, n), abs, min, max. "
        "Write numbers without thousands separators. Example: pct_change(5700, 3219).",
        {"expression": {"type": "string"}},
    ),
]
