import os
import json
from anthropic import Anthropic

client = Anthropic()

def prosecutor_argue(case_facts: dict, precedents: str) -> str:
    prompt = f"""You are a prosecutor at the European Court of Human Rights.
Your role is to argue that the state HAS violated the applicant's human rights.

CASE FACTS:
{json.dumps(case_facts, indent=2)}

RELEVANT PRECEDENTS:
{precedents}

Build the strongest possible argument that human rights were violated.
Structure your argument as:
1. Main violation alleged
2. Key facts supporting the violation
3. Relevant precedents supporting your position
4. Conclusion"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def prosecutor_rebut(case_facts: dict, precedents: str, defender_arguments: str) -> str:
    prompt = f"""You are a prosecutor at the European Court of Human Rights.
You have heard the defender's arguments. Now provide a rebuttal.

CASE FACTS:
{json.dumps(case_facts, indent=2)}

RELEVANT PRECEDENTS:
{precedents}

DEFENDER'S ARGUMENTS:
{defender_arguments}

Rebut the defender's arguments point by point.
Show why their arguments are weak or incorrect.
Strengthen your position that human rights WERE violated."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text