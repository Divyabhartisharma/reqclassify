
import os
import time
import pandas as pd

from hybrid_classifier import (
    classify_stage1_type,
    classify_stage2_quality,
    get_table_name,
    should_run_stage2_for_dataset,
)

# =========================================================
# CONFIG
# =========================================================
INPUT_CSV = r"D:\Divya\Master_Thesis_NFR_Classification\1_data\raw\unlabeled\Userstory\UserStory.csv"
OUTPUT_CSV = r"D:\Divya\Master_Thesis_NFR_Classification\4_results\ReqClassify\New_results\Reqclassify_Userstory_8class_test23csv"

# Force evaluation to User Story only.
DATASET_NAME = "Promise"
ACTIVE_TABLE = get_table_name(DATASET_NAME)
ISO_TOP_K = 3
EX_TOP_K = 3
SLEEP_SECONDS = 2.0
SAVE_EVERY_N_ROWS = 5

# =========================================================
# TARGET CLASSES
# =========================================================
# Only these labels will be written to the output CSV.
CUSTOM_CLASSES = ["FR", "Q_SC", "Q_PE", "Q_MN", "Q_PO", "Q_A", "Q_SE", "Q_US","LF","L","O","Q_FT"]

# If the classifier predicts something outside the 8-class user-story label space,
# this fallback is used so the output remains valid for 8-class evaluation.
FALLBACK_CLASS = "FR"

# =========================================================
# LABEL MAPPINGS
# =========================================================
TYPE_MAP = {
    "FUNCTIONAL": "FR",
    "QUALITY": "Q",
    "CONSTRAINT": "C",
    "REGULATORY": "R",
    "INTERFACE": "I",
    "BUSINESS": "B",
    "Unknown": "O",
    "UNKNOWN": "O",
}

SUBTYPE_TO_CUSTOM = {
    "Security": "Q_SE",
    "PerformanceEfficiency": "Q_PE",
    "Usability": "Q_US",
    "Maintainability": "Q_MN",
    "Compatibility": "Q_CM",
    "Portability": "Q_PO",
    "Reliability": "Q_RL",

    # For your 8-class setup, these are merged into closest available classes.
    "FaultTolerance": "Q_RL",
    "Availability": "Q_RL",
    "Scalability": "Q_CM",

    # FunctionalSuitability is treated as functional in your custom classes.
    "FunctionalSuitability": "FR",
}

# Used for RAG/human-feedback labels that may already come in compact format.
_SYSMAP = {
    "FR": "FR",
    "FUNCTIONAL": "FR",
    "Normal Operation": "FR",
    "FR_OP": "FR",
    "Configuration": "FR",
    "FR_CFG": "FR",
    "Initialization": "FR",
    "FR_INIT": "FR",
    "Shutdown": "FR",
    "FR_SHUT": "FR",
    "Fault Handling": "FR",
    "FR_FAULT": "FR",

    "PerformanceEfficiency": "Q_PE",
    "PE": "Q_PE",
    "Q_PE": "Q_PE",
    "Security": "Q_SE",
    "SE": "Q_SE",
    "Q_SE": "Q_SE",
    "Usability": "Q_US",
    "US": "Q_US",
    "Q_US": "Q_US",
    "Maintainability": "Q_MN",
    "MN": "Q_MN",
    "Q_MN": "Q_MN",
    "Compatibility": "Q_CM",
    "CM": "Q_CM",
    "Q_CM": "Q_CM",
    "Portability": "Q_PO",
    "PO": "Q_PO",
    "Q_PO": "Q_PO",
    "Reliability": "Q_RL",
    "RL": "Q_RL",
    "Q_RL": "Q_RL",
    "Availability": "Q_RL",
    "AV": "Q_RL",
    "Q_AV": "Q_RL",
    "FaultTolerance": "Q_RL",
    "FT": "Q_RL",
    "Q_FT": "Q_RL",
    "Scalability": "Q_CM",
    "SC": "Q_CM",
    "Q_SC": "Q_CM",
    "FunctionalSuitability": "FR",
    "FS": "FR",
    "Q_FS": "FR",
}


def map_to_8class(main_type: str, raw_subtype: str = "") -> str:
    """Return exactly one of the 8 target labels."""

    # 1. Functional user stories remain FR.
    if main_type == "FUNCTIONAL":
        return "FR"

    # 2. Quality user stories use subtype mapping.
    if main_type == "QUALITY":
        if raw_subtype in SUBTYPE_TO_CUSTOM:
            return SUBTYPE_TO_CUSTOM[raw_subtype]

        if raw_subtype in _SYSMAP:
            mapped = _SYSMAP[raw_subtype]
            return mapped if mapped in CUSTOM_CLASSES else FALLBACK_CLASS

        return FALLBACK_CLASS

    # 3. If RAG override or unexpected label already gives a target class, keep it.
    for value in [raw_subtype, main_type]:
        mapped = _SYSMAP.get(value, value)
        if mapped in CUSTOM_CLASSES:
            return mapped

    # 4. For 8-class evaluation, all unsupported types are forced to fallback.
    return FALLBACK_CLASS


# =========================================================
# MAIN LOOP
# =========================================================
def run_full_evaluation():
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {INPUT_CSV}")
        return

    if "Requirement" not in df.columns:
        print("ERROR: CSV must have a 'Requirement' column exactly with this spelling.")
        print(f"Available columns: {list(df.columns)}")
        return

    df = df.dropna(subset=["Requirement"]).reset_index(drop=True)
    total_rows = len(df)

    print("=" * 70)
    print("ReqClassify User Story 8-Class Evaluation")
    print("=" * 70)
    print(f"Dataset name     : {DATASET_NAME}")
    print(f"RAG example table: {ACTIVE_TABLE}")
    print(f"Target classes   : {', '.join(CUSTOM_CLASSES)}")
    print(f"Input rows       : {total_rows}")
    print("Stage 2 policy   : only QUALITY gets ISO 25010 subtype")
    print("=" * 70)

    start_row = 0
    results = []

    if os.path.exists(OUTPUT_CSV):
        existing_df = pd.read_csv(OUTPUT_CSV)
        start_row = len(existing_df)
        results = existing_df.to_dict("records")
        print(f"Resuming from row {start_row + 1}...")
    else:
        print("Starting fresh classification...")

    if start_row >= total_rows:
        print("Already completed! Check your output file.")
        return

    remaining = total_rows - start_row
    est_min = (remaining * SLEEP_SECONDS) / 60
    print(f"Remaining: {remaining} rows | Minimum wait time: ~{est_min:.1f} min\n")

    for i in range(start_row, total_rows):
        req_text = str(df.loc[i, "Requirement"]).strip()
        print(f"\n[{i + 1}/{total_rows}] Processing: {req_text[:90]}...")

        try:
            # Stage 1: classify main type using only User Story examples.
            stage1 = classify_stage1_type(
                req_text,
                iso_top_k=ISO_TOP_K,
                ex_top_k=EX_TOP_K,
                table_name=ACTIVE_TABLE,
            )
            main_label = stage1.get("type", "Unknown")
            raw_subtype = stage1.get("subtype", "") or ""

            # Stage 2: for User Story config, this should run only for QUALITY.
            if should_run_stage2_for_dataset(
                main_type=main_label,
                dataset_name=DATASET_NAME,
                table_name=ACTIVE_TABLE,
            ):
                stage2 = classify_stage2_quality(
                    req_text,
                    iso_top_k=ISO_TOP_K,
                    ex_top_k=EX_TOP_K,
                    main_type=main_label,
                    table_name=ACTIVE_TABLE,
                )
                raw_subtype = stage2.get("subtype", "") or ""

            predicted_class = map_to_8class(main_label, raw_subtype)

            print(f"  Stage 1: {main_label}")
            print(f"  Stage 2: {raw_subtype if raw_subtype else '-'}")
            print(f"  Final  : {predicted_class}")

            results.append({
                "Requirement": req_text,
                "Predicted_Class": predicted_class,
                "Stage1_Type": main_label,
                "Stage2_Subtype": raw_subtype,
                "RAG_Table": ACTIVE_TABLE,
            })

        except Exception as e:
            print(f"  API/Error at row {i + 1}: {str(e)[:120]}")
            results.append({
                "Requirement": req_text,
                "Predicted_Class": "ERROR",
                "Stage1_Type": "ERROR",
                "Stage2_Subtype": "",
                "RAG_Table": ACTIVE_TABLE,
            })
            time.sleep(15)

        if (i + 1) % SAVE_EVERY_N_ROWS == 0 or (i + 1) == total_rows:
            os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
            print(f"  [Saved at row {i + 1}] -> {OUTPUT_CSV}")

        time.sleep(SLEEP_SECONDS)

    print(f"\nFINISHED! Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    run_full_evaluation()
