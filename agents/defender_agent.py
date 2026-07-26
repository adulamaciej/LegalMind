import json
from config import call_claude, MODEL


def defender_argue(case_facts: dict, precedents: str) -> str:
    prompt = f"""You are a defense lawyer at the European Court of Human Rights.
Your role is to argue that the state has NOT violated the applicant's human rights.

CASE FACTS:
{json.dumps(case_facts, indent=2)}

RELEVANT PRECEDENTS:
{precedents}

Build the strongest possible argument that human rights were NOT violated.
Structure your argument as:
1. Main defense position
2. Key facts supporting no violation
3. Relevant precedents supporting your position
4. Conclusion"""

    try:
        return call_claude(prompt, model=MODEL, max_tokens=800)
    except Exception as e:
        print(f"Defender argument failed: {e}")
        return "The defense was unable to present arguments due to a technical error."


def defender_respond(case_facts: dict, precedents: str, prosecutor_rebuttal: str) -> str:
    prompt = f"""You are a defense lawyer at the European Court of Human Rights.
You have heard the prosecutor's rebuttal. Provide your final response.

CASE FACTS:
{json.dumps(case_facts, indent=2)}

RELEVANT PRECEDENTS:
{precedents}

PROSECUTOR'S REBUTTAL:
{prosecutor_rebuttal}

Respond to the prosecutor's rebuttal.
Maintain your position that human rights were NOT violated.
This is your final statement — make it count."""

    try:
        return call_claude(prompt, model=MODEL, max_tokens=800)
    except Exception as e:
        print(f"Defender response failed: {e}")
        return "The defense was unable to respond due to a technical error."