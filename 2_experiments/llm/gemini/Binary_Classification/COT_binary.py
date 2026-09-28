import sys
import os
import re
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Connect_Gemini import call_gemini

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
DATASET  = " "
CSV_NAME = " "

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

INPUT_DIR  = os.path.join(BASE_DIR, "1_data", "raw", "unlabeled")
OUTPUT_DIR = os.path.join(BASE_DIR, "2_experiments", "llm", "gemini", "Gemini_results", "Binary_Classification")

INPUT_CSV  = os.path.join(INPUT_DIR, DATASET, CSV_NAME)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, CSV_NAME.replace(".csv", "_cot.csv"))

TEMPERATURE = 0.0
BATCH_SIZE = 25


# chain of thought prompt on ISO 29148
def build_batch_prompt(items):
    lines = "\n".join([f"{i}: {text}" for i, text in items])

    return f"""
You are classifying software requirements using ISO/IEC/IEEE 29148 concepts.

Labels:
- FR = Functional Requirement: specifies system behavior / functions / services / interactions (what the system does).
- NFR = Non-Functional Requirement: specifies quality attributes, constraints, or conditions (how well / under what constraints),
        e.g., performance, security, usability, reliability, maintainability, portability, compatibility, availability,
        compliance/standards, resource limits, environment/operational constraints.

Reasoning protocol:
- For each requirement, think step-by-step and decide:
  1) Is it describing a system function/behavior/service? -> likely FR
  2) Or is it a quality/constraint/condition on behavior (speed, security, uptime, standards, UI look, etc.)? -> likely NFR
- If it contains both, choose the PRIMARY intent:
  - If the core is a function ("shall do X"), label FR.
  - If the core is a constraint/quality on a function ("shall do X within 2 seconds", "shall be secure", "shall be available"),
    label NFR.

Output rules (STRICT):
-Return ONLY csv in this format:
-<Requirement>:<Type>
- No explanations, no extra text, no bullet points.

Requirements:
{lines}
""".strip()


def parse_batch_output(raw):
    out = {}
    if not raw:
        return out
    for line in raw.strip().splitlines():
        line = line.strip().upper()
        m = re.match(r"^\s*(\d+)\s*:\s*(FR|NFR)\s*$", line)
        if m:
            out[int(m.group(1))] = m.group(2)
    return out


def main():
    df = pd.read_csv(INPUT_CSV)
    if "Requirement" not in df.columns:
        raise ValueError(f"CSV must contain 'Requirement'. Found: {list(df.columns)}")

    requirements = df["Requirement"].astype(str).tolist()
    predictions = ["UNKNOWN"] * len(requirements)

    for start in range(range(0, len(requirements), BATCH_SIZE), desc="Gemini CoT"):
        batch = [(i, requirements[i]) for i in range(start, min(start + BATCH_SIZE, len(requirements)))]
        prompt = build_batch_prompt(batch)
        raw = call_gemini(prompt, temperature=TEMPERATURE)

        mapping = parse_batch_output(raw)
        for i, _ in batch:
            if i in mapping:
                predictions[i] = mapping[i]

    # save
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df = pd.DataFrame({
        "Requirement": df["Requirement"],
        "Gemini_Prediction": predictions
    })
    out_df.to_csv(OUTPUT_CSV, index=False)

    print("Saved:", OUTPUT_CSV)
    print("UNKNOWN:", predictions.count("UNKNOWN"))


if __name__ == "__main__":
    main()