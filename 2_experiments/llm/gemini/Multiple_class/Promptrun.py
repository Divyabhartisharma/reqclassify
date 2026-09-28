import sys,os
import re
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Connect_Gemini import call_gemini
from prompts_multiclass import *

# change here
#dataset = " "  #promise (12 class) / userstory (8 class)
#technique = " "   #zero, iso, few, cot

base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

if dataset=="promise":
    # input_csv = os.path.join(base,"1_data","raw","unlabeled"," ")
    labels = promise_labels
    examples = promise_examples
    if technique=="zero": prompt_text = promise_zero
    if technique=="iso": prompt_text = promise_iso
    if technique=="few": prompt_text = promise_few
    if technique=="cot": prompt_text = promise_cot
else:
    #input_csv = os.path.join(base,"1_data","raw","unlabeled"," ")
    labels = userstory_labels
    examples = userstory_examples
    if technique=="zero": prompt_text = userstory_zero
    if technique=="iso": prompt_text = userstory_iso
    if technique=="few": prompt_text = userstory_few
    if technique=="cot": prompt_text = userstory_cot

#batch size same as before
bs = 20
if technique=="zero":
    bs = 25

output_csv = os.path.join(base,"2_experiments","llm","gemini","Gemini_results","Multiple_Class",dataset+"_"+technique+".csv")

df = pd.read_csv(input_csv)
reqs = df["Requirement"].astype(str).tolist()
preds = ["UNKNOWN"]*len(reqs)
print(dataset,technique,len(reqs))

for start in tqdm(range(0,len(reqs),bs)):
    end = min(start+bs,len(reqs))
    lines = ""
    for i in range(start,end):
        lines += str(i)+": "+reqs[i]+"\n"

    prompt = prompt_text.format(lines=lines.strip(),examples=examples).strip()
    raw = call_gemini(prompt,temperature=0.0)
    #print(raw)

    if raw=="ERROR" or not raw:
        print("failed",start)
        continue

    for line in raw.strip().splitlines():
        m = re.match(r"^\s*(\d+)\s*:\s*("+"|".join(labels)+r")\s*$", line.strip().upper())
        if m:
            idx = int(m.group(1))
            if idx>=start and idx<end:
                preds[idx] = m.group(2)

out = pd.DataFrame()
out["ID"] = range(len(df))
out["Requirement"] = df["Requirement"]
out["Gemini_Prediction"] = preds
os.makedirs(os.path.dirname(output_csv),exist_ok=True)
out.to_csv(output_csv,index=False)

print("saved",output_csv)
print("unknown",preds.count("UNKNOWN"))