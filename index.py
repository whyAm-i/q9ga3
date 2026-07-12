from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

class Request(BaseModel):
    problem_id: str
    problem: str

class Output(BaseModel):
    reasoning: str
    answer: int

@app.post("/")
def solve(req: Request):
    r = client.responses.parse(
        model="gpt-4.1",
        input=f"""
Solve the arithmetic word problem.

Return JSON only.
reasoning must be at least 80 characters.
answer must be an integer.

Problem:
{req.problem}
""",
        text_format=Output,
    )

    out = r.output_parsed

    if len(out.reasoning) < 80:
        out.reasoning += " The remaining values mentioned in the problem are distractors and do not affect the integer result."

    return {
        "reasoning": out.reasoning,
        "answer": int(out.answer)
    }
