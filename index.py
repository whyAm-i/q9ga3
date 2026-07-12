"""
Arithmetic word-problem solver microservice (Vercel entrypoint: api/index.py).

POST /solve
Body: {"problem_id": "p0", "problem": "..."}
Returns EXACTLY:
{
  "reasoning": "<string, >= 80 chars>",
  "answer": <integer>
}

DESIGN (v2): the LLM is no longer trusted to do arithmetic.
-------------------------------------------------------------
The earlier version asked the model to compute the final numeric answer
itself inside its own chain-of-thought. That failed in practice: a model
can write "2244 * 1.08 = 2425.02" when the true product is 2423.52, and
nothing checked that the model's own claimed step was correct. LLMs are
good at figuring out *which* numbers matter and *what operations* to
apply to them; they are not reliable calculators.

So now:
  1. The model identifies the relevant numbers (ignoring distractors) and
     emits a small ordered list of arithmetic STEPS as structured JSON
     (an expression string per step, e.g. "base * 0.85"), plus a rounding
     rule for the final step.
  2. We NEVER eval() the model's expressions with floats. We parse each
     expression into a restricted AST (only +, -, *, /, parentheses,
     numbers, and references to previously computed step variables) and
     evaluate it ourselves using Python's Fraction type, so every
     intermediate value is mathematically exact -- no float drift at all.
  3. We apply the model-specified rounding rule (or default) ONLY at the
     very end, server-side, using an unambiguous rounding function.
  4. We generate the final `reasoning` string ourselves from the executed
     steps (their descriptions + our exact computed values), so it is
     guaranteed to be internally consistent with `answer`. We do not
     trust prose the model wrote about arithmetic it didn't actually run.

This closes the exact bug class we found (a model claiming an arithmetic
result that doesn't match the true product/sum), because the number that
ends up in `answer` is never something the model computed -- it's always
something we computed, from operations the model only had to choose
correctly.
"""

import ast
import json
import os
import re
from fractions import Fraction
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Arithmetic Word Problem Solver")

LLM_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
LLM_BASE_URL = os.environ.get("AIPIPE_BASE_URL", "https://aipipe.org/openrouter/v1")
MODEL = os.environ.get("AIPIPE_MODEL", "openai/gpt-4.1-nano")
MAX_ATTEMPTS = 4
REQUEST_TIMEOUT = 60

# ---------------------------------------------------------------------------
# Prompt: the model now emits STRUCTURE, not a final numeric answer.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You analyze multi-step arithmetic word problems and break them
into a precise sequence of arithmetic operations. You do NOT need to compute
the final numeric result yourself -- a separate exact-arithmetic engine will
execute your steps. Your job is only to get the STRUCTURE right: which
numbers matter, in what order, with what operations, and what rounding rule
applies at the end.

Respond with ONLY a single JSON object, no markdown fences, no prose outside
the JSON, with EXACTLY these keys:

  "values": an object mapping short variable names to the relevant numbers
            pulled from the problem (as plain JSON numbers). Only include
            numbers that are actually needed. Do not include distractor
            numbers here.
  "steps": an ordered array of objects, each with:
      "var": a new variable name for this step's result (e.g. "base",
             "discounted", "taxed")
      "expr": a plain arithmetic expression string using ONLY: numbers,
              +, -, *, /, parentheses, and variable names already defined
              (either in "values" or an earlier step's "var"). No function
              calls, no percent signs -- express e.g. "15% discount" as
              multiplying by 0.85, and "8% tax" as multiplying by 1.08.
      "description": short human-readable label for this step
                      (e.g. "apply 15% bulk discount")
  "final_var": the variable name (from "steps") holding the final result
               before rounding.
  "rounding": one of "nearest", "floor", "ceil", or "exact"
              ("exact" means the final value should already be a whole
              number; use this for pure counting problems).
  "distractors_ignored": a short array of strings naming which numbers or
              details in the problem were irrelevant and why.

Example output for: "A workshop orders 150 tiles at 8 dollars each. Any
order of more than 50 units earns a 25% bulk discount. After the discount,
a 5% tax is applied. The truck holds 900 km worth of fuel and there are 3
product lines. What is the final total cost?"

{"values": {"qty": 150, "price": 8}, "steps": [{"var": "base", "expr": "qty * price", "description": "base cost before discount"}, {"var": "discounted", "expr": "base * 0.75", "description": "apply 25% bulk discount (order > 50 units)"}, {"var": "taxed", "expr": "discounted * 1.05", "description": "apply 5% tax"}], "final_var": "taxed", "rounding": "nearest", "distractors_ignored": ["truck fuel range (900 km) is irrelevant to cost", "number of product lines (3) is irrelevant to cost"]}

Do not include any other keys. Do not compute or state the final numeric
answer anywhere -- only provide the expressions and let the engine compute it.
"""


class SolveRequest(BaseModel):
    problem_id: str
    problem: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Safe, exact expression evaluator (no eval(), no floats).
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}


def _safe_eval_expr(expr: str, env: dict[str, Fraction]) -> Fraction:
    """
    Evaluate `expr` using only +, -, *, /, parentheses, numeric literals,
    and names already present in `env`. Everything is computed as an exact
    Fraction -- never a float -- so there is no possibility of the kind of
    silent rounding error an LLM's own arithmetic can introduce.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"could not parse expression {expr!r}: {e}") from e

    def _eval(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError(f"non-numeric literal in expression: {node.value!r}")
            # Route through str() for floats so "0.85" becomes an exact
            # Fraction(85, 100) rather than a binary-float approximation.
            return Fraction(str(node.value))
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise ValueError(f"unknown variable {node.id!r} in expression {expr!r}")
            return env[node.id]
        if isinstance(node, ast.BinOp):
            op_fn = _ALLOWED_BINOPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"operator {type(node.op).__name__} is not allowed")
            left = _eval(node.left)
            right = _eval(node.right)
            if op_fn is _ALLOWED_BINOPS[ast.Div] and right == 0:
                raise ValueError("division by zero in expression")
            return op_fn(left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            val = _eval(node.operand)
            return val if isinstance(node.op, ast.UAdd) else -val
        raise ValueError(f"disallowed syntax in expression {expr!r}: {ast.dump(node)}")

    return _eval(tree)


def _round_fraction(value: Fraction, mode: str) -> int:
    if mode == "floor":
        return value.numerator // value.denominator
    if mode == "ceil":
        return -(-value.numerator // value.denominator)
    if mode == "exact":
        if value.denominator != 1:
            raise ValueError(
                f"rounding mode 'exact' requires a whole number, got {float(value)}"
            )
        return value.numerator
    if mode == "nearest":
        # Round-half-up on the exact rational value (avoids banker's
        # rounding surprises and any float comparison issues).
        floor_val = value.numerator // value.denominator
        remainder = value - floor_val
        return floor_val + 1 if remainder >= Fraction(1, 2) else floor_val
    raise ValueError(f"unknown rounding mode {mode!r}")


def _execute_plan(plan: dict[str, Any]) -> tuple[int, str]:
    """
    Execute a validated structured plan and return (answer, reasoning),
    where `reasoning` is built entirely from our own computed values --
    never from model-authored arithmetic prose.
    """
    values = plan["values"]
    steps = plan["steps"]
    final_var = plan["final_var"]
    rounding = plan["rounding"]

    env: dict[str, Fraction] = {}
    for name, num in values.items():
        if isinstance(num, bool) or not isinstance(num, (int, float)):
            raise ValueError(f"value {name!r} is not numeric")
        env[name] = Fraction(str(num))

    reasoning_parts = []
    if values:
        values_str = ", ".join(f"{k}={v}" for k, v in values.items())
        reasoning_parts.append(f"Relevant values: {values_str}.")

    for step in steps:
        var = step["var"]
        expr = step["expr"]
        desc = step.get("description", "")
        result = _safe_eval_expr(expr, env)
        env[var] = result
        display_val = (
            str(result.numerator)
            if result.denominator == 1
            else f"{float(result):.6f}".rstrip("0").rstrip(".")
        )
        label = f" ({desc})" if desc else ""
        reasoning_parts.append(f"{var} = {expr} = {display_val}{label}.")

    if final_var not in env:
        raise ValueError(f"final_var {final_var!r} was never computed")

    final_value = env[final_var]
    answer = _round_fraction(final_value, rounding)

    if rounding != "exact" and final_value.denominator != 1:
        reasoning_parts.append(
            f"Rounding {float(final_value):.6f} to nearest integer ({rounding}): {answer}."
        )
    else:
        reasoning_parts.append(f"Final answer: {answer}.")

    distractors = plan.get("distractors_ignored") or []
    if distractors:
        reasoning_parts.append("Ignored as irrelevant: " + "; ".join(distractors) + ".")

    reasoning = " ".join(reasoning_parts)

    # Guarantee the >= 80 char contract even for very short problems,
    # without inventing false content -- pad with a factual restatement.
    if len(reasoning) < 80:
        reasoning += (
            f" (Computed via {len(steps)} exact arithmetic step"
            f"{'s' if len(steps) != 1 else ''} with no floating-point rounding.)"
        )

    return answer, reasoning


# ---------------------------------------------------------------------------
# Plan validation (shape of the model's structured output, not the math).
# ---------------------------------------------------------------------------

_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_plan_shape(obj: dict[str, Any]) -> list[str]:
    problems = []
    required_keys = {"values", "steps", "final_var", "rounding"}
    keys = set(obj.keys())
    missing = required_keys - keys
    extra = keys - (required_keys | {"distractors_ignored"})
    if missing:
        problems.append(f"missing keys: {sorted(missing)}")
    if extra:
        problems.append(f"unexpected extra keys: {sorted(extra)}")
    if missing or extra:
        return problems

    values = obj["values"]
    if not isinstance(values, dict) or not values:
        problems.append("`values` must be a non-empty object of name -> number")
    else:
        for k, v in values.items():
            if not _VAR_NAME_RE.match(k):
                problems.append(f"`values` key {k!r} is not a valid variable name")
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                problems.append(f"`values.{k}` must be a plain number")

    steps = obj["steps"]
    if not isinstance(steps, list) or not steps:
        problems.append("`steps` must be a non-empty array")
    else:
        for i, step in enumerate(steps):
            if not isinstance(step, dict) or "var" not in step or "expr" not in step:
                problems.append(f"steps[{i}] must be an object with 'var' and 'expr'")
                continue
            if not _VAR_NAME_RE.match(str(step["var"])):
                problems.append(f"steps[{i}].var is not a valid variable name")
            if not isinstance(step["expr"], str) or not step["expr"].strip():
                problems.append(f"steps[{i}].expr must be a non-empty string")

    if not isinstance(obj.get("final_var"), str):
        problems.append("`final_var` must be a string")

    if obj.get("rounding") not in {"nearest", "floor", "ceil", "exact"}:
        problems.append("`rounding` must be one of: nearest, floor, ceil, exact")

    return problems


# ---------------------------------------------------------------------------
# LLM call + JSON extraction (unchanged strategy, new target schema).
# ---------------------------------------------------------------------------


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group(0))


def _call_model(problem: str, correction: str | None = None) -> dict[str, Any]:
    user_content = f"Problem: {problem}"
    if correction:
        user_content += (
            f"\n\nYour previous response was invalid: {correction}\n"
            "Return ONLY the corrected JSON object, following the schema exactly."
        )

    headers = {"Content-Type": "application/json"}
    if LLM_TOKEN:
        headers["Authorization"] = f"Bearer {LLM_TOKEN}"

    resp = requests.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers=headers,
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
        },
        timeout=REQUEST_TIMEOUT,
    )

    if resp.status_code != 200:
        raise ValueError(f"LLM request failed ({resp.status_code}): {resp.text[:300]}")

    data = resp.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected LLM response shape: {data}") from e

    return _extract_json_object(text)


# ---------------------------------------------------------------------------
# Top-level solve: model produces a plan, we execute it, we validate the
# FINAL response contract (exactly {reasoning, answer}) before returning.
# ---------------------------------------------------------------------------


def _validate_final_shape(obj: dict[str, Any]) -> list[str]:
    problems = []
    if set(obj.keys()) != {"reasoning", "answer"}:
        problems.append(f"final object must have exactly keys reasoning, answer; got {sorted(obj.keys())}")
        return problems
    if not isinstance(obj["reasoning"], str) or len(obj["reasoning"]) < 80:
        problems.append("`reasoning` must be a string >= 80 characters")
    if isinstance(obj["answer"], bool) or not isinstance(obj["answer"], int):
        problems.append("`answer` must be a JSON integer")
    return problems


def solve_problem(problem: str) -> dict[str, Any]:
    correction = None
    last_error = None

    for _ in range(MAX_ATTEMPTS):
        try:
            plan = _call_model(problem, correction)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = str(e)
            correction = f"Could not parse a JSON object from your response ({e})."
            continue

        shape_problems = _validate_plan_shape(plan)
        if shape_problems:
            last_error = "; ".join(shape_problems)
            correction = last_error
            continue

        # Execute the plan with exact arithmetic. Any failure here (unknown
        # variable, bad expression, division by zero, non-integer under
        # 'exact' rounding, etc.) is fed back to the model as a correction,
        # exactly like a shape error -- the model gets to retry the
        # STRUCTURE, but never gets to supply the number itself.
        try:
            answer, reasoning = _execute_plan(plan)
        except (ValueError, KeyError, ZeroDivisionError) as e:
            last_error = f"plan execution failed: {e}"
            correction = last_error
            continue

        final_obj = {"reasoning": reasoning, "answer": answer}
        final_problems = _validate_final_shape(final_obj)
        if not final_problems:
            return final_obj

        # Should be unreachable given how we build final_obj, but keep the
        # safety net consistent with the retry loop.
        last_error = "; ".join(final_problems)
        correction = last_error

    raise HTTPException(
        status_code=502,
        detail=f"Model failed to produce a valid, executable plan after {MAX_ATTEMPTS} attempts: {last_error}",
    )


@app.post("/solve")
def solve(req: SolveRequest):
    result = solve_problem(req.problem)
    return result


@app.get("/")
def root():
    return {"service": "arithmetic-word-problem-solver", "status": "ok", "endpoints": ["/solve", "/health"]}


@app.get("/health")
def health():
    return {"status": "ok"}