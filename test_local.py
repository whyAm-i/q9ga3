"""
Quick smoke test against a running instance (default: http://localhost:8000).

Usage:
    python test_local.py
    python test_local.py https://your-deployed-url.com
"""

import sys
import json
import urllib.request

base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

problems = [
    {
        "problem_id": "p0",
        "problem": (
            "A workshop orders 150 tiles at 8 dollars each. Any order of more "
            "than 50 units earns a 25% bulk discount. After the discount, a 5% "
            "sales tax is applied. The warehouse is 12 km away and stocks 4 "
            "product lines. What is the final cost in dollars?"
        ),
        "expected": 945,
    },
    {
        "problem_id": "p1",
        "problem": (
            "A bus has 3 rows of 4 seats and 2 rows of 5 seats. It makes 6 "
            "stops on its route, and the driver has 14 years of experience. "
            "How many total seats does the bus have?"
        ),
        "expected": 22,
    },
]

for p in problems:
    payload = json.dumps({"problem_id": p["problem_id"], "problem": p["problem"]}).encode()
    req = urllib.request.Request(
        f"{base_url}/solve",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())

    keys_ok = set(result.keys()) == {"reasoning", "answer"}
    type_ok = isinstance(result.get("answer"), int) and not isinstance(result.get("answer"), bool)
    len_ok = isinstance(result.get("reasoning"), str) and len(result["reasoning"]) >= 80
    correct = result.get("answer") == p["expected"]

    print(f"[{p['problem_id']}] keys_ok={keys_ok} type_ok={type_ok} len_ok={len_ok} "
          f"correct={correct} (got {result.get('answer')}, expected {p['expected']})")
    print(f"        reasoning: {result.get('reasoning')}")
