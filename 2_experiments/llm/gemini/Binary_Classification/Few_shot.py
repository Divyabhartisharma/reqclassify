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

# project folder (Automated_classification)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

INPUT_DIR  = os.path.join(BASE_DIR, "1_data", "raw", "unlabeled")
OUTPUT_DIR = os.path.join(BASE_DIR, "2_experiments", "llm", "gemini", "Gemini_results", "Binary_Classification")

INPUT_CSV  = os.path.join(INPUT_DIR, DATASET, CSV_NAME)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, CSV_NAME.replace(".csv", "_fewshot.csv"))

TEMPERATURE = 0.0
BATCH_SIZE = 25
# -----------------------------
# Few-shot prompt
# -----------------------------
def build_batch_prompt(items):

    examples = """
Examples:

Requirement: The product shall allow a player to end a game at any time during the game.
Answer: FR

Requirement: The product shall be able to process all transactions.The product shall process minimum of 1 million transactions per year.
Answer: NFR
"""

    lines = "\n".join([f"{i}: {text}" for i, text in items])

    return f"""
Classify each requirement as exactly one label:

FR = Functional Requirement  
NFR = Non-Functional Requirement  

{examples}

Return ONLY csv in this format:
<Requirement>:<Type>

Requirements:
{lines}
""".strip()

# -----------------------------
# Gemini output
# -----------------------------
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


# -----------------------------
# Main
# -----------------------------
def main():
    df = pd.read_csv(INPUT_CSV)

    if "Requirement" not in df.columns:
        raise ValueError(f"CSV must contain 'Requirement'. Found: {list(df.columns)}")

    requirements = df["Requirement"].astype(str).tolist()
    predictions = ["UNKNOWN"] * len(requirements)

    for start in range(range(0, len(requirements), BATCH_SIZE), desc="Gemini Few-shot"):
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