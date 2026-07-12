"""
Arithmetic word-problem solver microservice (Vercel entrypoint: api/index.py).

POST /solve
Body: {"problem_id": "p0", "problem": "..."}
Returns EXACTLY:
{
  "reasoning": "<string, >= 80 chars>",
  "answer": <integer>
}

Design goals (per the grading contract):
  - Exactly two keys, no extras.
  - `answer` must be a genuine JSON integer (not "945", not 945.0).
  - `reasoning` must be a string of at least 80 characters showing steps.
  - No markdown, no currency symbols in the answer field.
  - Robust to LLM slip-ups: we validate the model's output ourselves and
    retry with corrective feedback rather than trusting it blindly.
"""

import json
import os
import re
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Arithmetic Word Problem Solver")

# AI Pipe: a proxy in front of OpenRouter/OpenAI/Gemini that swaps your JWT
# for the real provider key server-side. Docs: https://aipipe.org/
# We use its OpenRouter-compatible chat completions endpoint.
AIPIPE_TOKEN = os.environ["AIPIPE_TOKEN"]
AIPIPE_BASE_URL = os.environ.get("AIPIPE_BASE_URL", "https://aipipe.org/openrouter/v1")
MODEL = os.environ.get("AIPIPE_MODEL", "openai/gpt-4.1-mini")
MAX_ATTEMPTS = 4
REQUEST_TIMEOUT = 60

SYSTEM_PROMPT = """You solve multi-step arithmetic word problems.

Rules you MUST follow:
- The problem may contain irrelevant distractor numbers. Identify which numbers
  actually matter and ignore the rest, but you may mention why you ignored them.
- Work the problem step by step internally, then produce a reasoning string
  that shows those steps concisely (numbers and operations), at least 80
  characters long.
- The final answer must be a single integer. If the natural result is a
  decimal, round/truncate/compute exactly as the problem implies (e.g. money
  problems usually resolve to whole cents/dollars if the inputs do; if the
  problem doesn't specify, round to the nearest integer).
- Respond with ONLY a single JSON object, no markdown fences, no prose outside
  the JSON, with EXACTLY two keys:
    "reasoning": string (>= 80 characters, shows your work)
    "answer": integer (a bare JSON number, no quotes, no currency symbols,
              no decimal point)
- Do not include any other keys.

Example output:
{"reasoning": "Base = 150 * 8 = 1200. Order > 50 so apply 25% discount: 1200 * 0.75 = 900. Add 5% tax: 900 * 1.05 = 945. The km and product-line counts are irrelevant.", "answer": 945}
"""


class SolveRequest(BaseModel):
    problem_id: str
    problem: str = Field(..., min_length=1)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response, tolerating stray fences/text."""
    text = text.strip()
    # Strip markdown fences if the model added them despite instructions.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # If there's leading/trailing prose, grab the first {...} block.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group(0))


def _validate_shape(obj: dict[str, Any]) -> list[str]:
    """Return a list of problems with the shape; empty list means valid."""
    problems = []

    keys = set(obj.keys())
    if keys != {"reasoning", "answer"}:
        extra = keys - {"reasoning", "answer"}
        missing = {"reasoning", "answer"} - keys
        if extra:
            problems.append(f"unexpected extra keys: {sorted(extra)}")
        if missing:
            problems.append(f"missing keys: {sorted(missing)}")
        return problems  # no point checking further if keys are wrong

    reasoning = obj["reasoning"]
    answer = obj["answer"]

    if not isinstance(reasoning, str):
        problems.append("`reasoning` must be a string")
    elif len(reasoning) < 80:
        problems.append(
            f"`reasoning` must be at least 80 characters (got {len(reasoning)})"
        )

    # bool is a subclass of int in Python -- explicitly reject it.
    if isinstance(answer, bool) or not isinstance(answer, int):
        problems.append("`answer` must be a JSON integer, not a string/float/bool")

    if isinstance(answer, str):
        problems.append("`answer` must not be a quoted string")

    return problems


def _call_model(problem: str, correction: str | None = None) -> dict[str, Any]:
    user_content = f"Problem: {problem}"
    if correction:
        user_content += (
            f"\n\nYour previous response was invalid: {correction}\n"
            "Return ONLY the corrected JSON object, following the rules exactly."
        )

    resp = requests.post(
        f"{AIPIPE_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {AIPIPE_TOKEN}",
            "Content-Type": "application/json",
        },
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
        raise ValueError(f"AI Pipe request failed ({resp.status_code}): {resp.text[:300]}")

    data = resp.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected AI Pipe response shape: {data}") from e

    return _extract_json_object(text)


def solve_problem(problem: str) -> dict[str, Any]:
    correction = None
    last_error = None

    for _ in range(MAX_ATTEMPTS):
        try:
            obj = _call_model(problem, correction)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = str(e)
            correction = f"Could not parse a JSON object from your response ({e})."
            continue

        problems = _validate_shape(obj)
        if not problems:
            # Normalize: ensure answer is plain int, reasoning is plain str.
            return {"reasoning": str(obj["reasoning"]), "answer": int(obj["answer"])}

        last_error = "; ".join(problems)
        correction = last_error

    raise HTTPException(
        status_code=502,
        detail=f"Model failed to produce valid output after {MAX_ATTEMPTS} attempts: {last_error}",
    )


@app.post("/solve")
def solve(req: SolveRequest):
    result = solve_problem(req.problem)
    return result


@app.get("/health")
def health():
    return {"status": "ok"}
