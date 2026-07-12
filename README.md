# Arithmetic Word Problem Solver

A FastAPI microservice with one endpoint, `POST /solve`, that answers
multi-step arithmetic word problems in a strict JSON contract.

## Contract

Request:
```json
{"problem_id": "p0", "problem": "..."}
```

Response (always exactly these two keys):
```json
{"reasoning": "at least 80 characters of step-by-step work...", "answer": 945}
```

`answer` is always a bare JSON integer — never a string, float, or value with
a currency symbol.

## How it stays reliable

1. **System prompt** instructs the model to ignore distractor numbers, show
   its steps, and emit only a two-key JSON object.
2. **Server-side validation** (`_validate_shape`) independently checks:
   - exactly the keys `reasoning` and `answer`, nothing else
   - `reasoning` is a string ≥ 80 characters
   - `answer` is a real `int` (booleans and strings are explicitly rejected)
3. **Self-correcting retries**: if validation fails or the JSON can't be
   parsed (e.g. the model added a stray code fence), the service sends the
   error back to the model and asks it to correct just that, up to 4
   attempts, before returning it to the caller.
4. Only after passing validation does the service return the JSON — so a
   malformed model response never reaches the grader.

## Auth: AI Pipe

This service calls the model through [AI Pipe](https://aipipe.org/), which
proxies OpenRouter (and OpenAI/Gemini) using a JWT instead of a raw provider
API key. You get your token by logging into aipipe.org — it looks like a
normal JWT (`eyJhbGciOi...`).

Set it as `AIPIPE_TOKEN`. By default the service calls
`https://aipipe.org/openrouter/v1/chat/completions` with model
`openai/gpt-4.1-nano` — the exact model shown in AI Pipe's own docs/examples.
Other OpenRouter models (e.g. `anthropic/claude-sonnet-4`,
`google/gemini-2.5-flash`, `openai/gpt-4.1`) can be set via `AIPIPE_MODEL`,
but some return `402 Insufficient credits` even when your AI Pipe balance
looks fine — that error comes from OpenRouter's side of the proxy, not your
AI Pipe account, so not every model is necessarily covered. Stick with
`openai/gpt-4.1-nano` if you hit that.

Note: AI Pipe's free tier is $0.10/week — fine for testing and light grading
runs, but check your usage at `https://aipipe.org` if you're running a large
problem set.

## Project layout

```
solver-service/
├── api/
│   └── index.py     # FastAPI app — Vercel auto-detects this as the entrypoint
├── requirements.txt
├── vercel.json       # sets function timeout; entrypoint detection is automatic
├── test_local.py
└── Dockerfile        # optional, only needed if you deploy elsewhere instead
```

## Running locally

```bash
export AIPIPE_TOKEN=eyJhbGciOi...       # your AI Pipe JWT
pip install -r requirements.txt
uvicorn api.index:app --host 0.0.0.0 --port 8000
```

Then in another terminal:
```bash
python test_local.py
```

## Deploying to Vercel

Vercel auto-detects Python apps: it looks for an entrypoint at `api/index.py`
(among a few other recognized names), finds `fastapi` in `requirements.txt`,
and builds the whole app into a single Vercel Function — no manual
`builds`/`routes` config needed. `vercel.json` here only sets the function
timeout.

### Via the dashboard
1. Push this folder to a GitHub repo.
2. [vercel.com/new](https://vercel.com/new) → import the repo.
3. In Project Settings → Environment Variables, add `AIPIPE_TOKEN` (and
   optionally `AIPIPE_MODEL`, `AIPIPE_BASE_URL`).
4. Deploy. Your endpoint will be `https://<project>.vercel.app/solve`.

### Via the CLI
```bash
npm i -g vercel
vercel login
vercel env add AIPIPE_TOKEN production      # paste your JWT when prompted
vercel --prod
```

Give the grader:
```
https://<project>.vercel.app/solve
```

### Timeout heads-up
Vercel's **Hobby plan caps function duration at 60 seconds** (Pro allows
more). Each `/solve` call may make up to `MAX_ATTEMPTS` (4) sequential model
calls if the model keeps producing invalid JSON — on a slow model that could
approach the limit. If you see `504 FUNCTION_INVOCATION_TIMEOUT`:
- lower `MAX_ATTEMPTS` in `api/index.py`, and/or
- use a faster model via `AIPIPE_MODEL` if `openai/gpt-4.1-nano` still feels slow, and/or
- raise `maxDuration` in `vercel.json` if you're on a plan that allows it.

### If you deploy elsewhere instead
A `Dockerfile` is included for Render/Fly.io/Railway/any container host, in
case Vercel's function limits don't fit your grading run — same
`AIPIPE_TOKEN` env var, endpoint is `/solve` either way.

## Notes / things you may want to tune

- `MAX_ATTEMPTS` in `main.py` controls how many self-correction retries are
  allowed before returning a 502. Raise it if you see occasional grading
  failures on hard problems.
- `AIPIPE_MODEL` defaults to `openai/gpt-4.1-nano` (the model AI Pipe's docs
  use). Swap only if you've confirmed the alternative model doesn't hit
  `402 Insufficient credits` on AI Pipe's OpenRouter proxy — that error can
  happen per-model regardless of your own AI Pipe balance.
- If the grader ever sends malformed input (missing `problem` field),
  FastAPI/pydantic will return a 422 automatically — you may want to wrap
  this in a custom handler if the grader expects the solver's own JSON shape
  even on bad input.
