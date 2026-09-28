import sys, os
import re
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Connect_Gemini import call_gemini
from prompts import *

#dataset = "promise"   # promise or userstory
#technique = "zero"    # zero / iso / few / cot

base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))


def get_prompt(dataset, technique):
    if dataset == "promise":
        if technique == "zero":
            return promise_zero
        if technique == "iso":
            return promise_iso
        if technique == "few":
            return promise_few
        if technique == "cot":
            return promise_cot
    else:
        if technique == "zero":
            return userstory_zero
        if technique == "iso":
            return userstory_iso
        if technique == "few":
            return userstory_few
        if technique == "cot":
            return userstory_cot


def read_output(raw, labels, start, end, preds):
    for line in raw.strip().splitlines():
        m = re.match(r"^\s*(\d+)\s*:\s*(" + "|".join(labels) + r")\s*$", line.strip().upper())
        if m:
            idx = int(m.group(1))
            if start <= idx < end:
                preds[idx] = m.group(2)


if dataset == "promise":
   # input_csv = os.path.join(base, "1_data", "raw", "unlabeled", " ")
    labels = promise_labels
    examples = promise_examples
else:
    #input_csv = os.path.join(base, "1_data", "raw", "unlabeled", " ")
    labels = userstory_labels
    examples = userstory_examples

prompt_text = get_prompt(dataset, technique)

batch_size = 20
if technique == "zero":
    batch_size = 25

output_csv = os.path.join(base, "2_experiments", "llm", "gemini", "Gemini_results", "NFR_Subtype",
                          dataset + "_" + technique + ".csv")


df = pd.read_csv(input_csv)
reqs = df["Requirement"].astype(str).tolist()
preds = ["UNKNOWN"] * len(reqs)

print(dataset, technique, "-", len(reqs), "requirements")

for start in tqdm(range(0, len(reqs), batch_size)):
    end = min(start + batch_size, len(reqs))
    lines = ""
    for i in range(start, end):
        lines += str(i) + ": " + reqs[i] + "\n"

    prompt = prompt_text.format(lines=lines.strip(), examples=examples).strip()
    raw = call_gemini(prompt, temperature=0.0)
    # print(raw)

    if raw == "ERROR" or not raw:
        print("batch failed", start)
        continue

    read_output(raw, labels, start, end, preds)

out = pd.DataFrame()
out["ID"] = range(len(df))
out["Requirement"] = df["Requirement"]
out["Gemini_Prediction"] = preds
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
out.to_csv(output_csv, index=False)

print("saved", output_csv)
print("unknown:", preds.count("UNKNOWN"))