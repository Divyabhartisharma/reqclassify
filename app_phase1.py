import streamlit as st
import json
import time
import re as _re_custom
import pandas as pd

from rag_layer import store_human_feedback
from hybrid_classifier import (
    DATASET_CONFIG,
    classify_stage1_type,
    classify_stage2_quality,
    classify_batch_single_shot,
    get_dataset_config,
    get_dataset_valid_classes,
    stage2_for_dataset,
)


def _parse_custom_classes(text):
    # pull out class names from "classify into: FR, Q_PE, ..." style instructions
    parts = text.split(":", 1)[1] if ":" in text else text
    STOP = {"The","You","In","If","For","These","Only","Into","And","Or",
            "Classify","Requirement","Categories","Category","Following",
            "Requirements","Classes","Specific","Into","These","All"}
    found = _re_custom.findall(r"([A-Z][A-Za-z0-9_]*)", parts)
    seen = set()
    result = []
    for c in found:
        if c not in STOP and c.upper() not in seen:
            seen.add(c.upper())
            result.append(c)
    return result


# all known subtype codes — used to decide whether to show subtype badge
_SUBTYPE_CODES = {
    "Q_SE","Q_PE","Q_US","Q_MN","Q_CM","Q_PO","Q_RL","Q_FT","Q_SC","Q_FS","Q_RL",
    "US","PE","MN","SE","CM","PO","RL","FT","SC","FS","A",
    "FR_OP","FR_CFG","FR_INIT","FR_SHUT","FR_FAULT",
    "C_O","C_LF","C_D","O","LF",
    "R_L",
}

_TYPE_ONLY = {
    "FR","Q","C","R","B","PE","SE","US","MN","CM","PO","RL","AV","FT","SC",
    "D","OTHER","O","LF","A","L"
}

# maps various label variants to canonical short codes
_SYSMAP = {
    "FR":"FR","FUNCTIONAL":"FR",
    "Normal Operation":"FR_OP","FR_OP":"FR_OP",
    "Configuration":"FR_CFG","FR_CFG":"FR_CFG",
    "Initialization":"FR_INIT","FR_INIT":"FR_INIT",
    "Shutdown":"FR_SHUT","FR_SHUT":"FR_SHUT",
    "Fault Handling":"FR_FAULT","FR_FAULT":"FR_FAULT",
    "PerformanceEfficiency":"Q_PE","PE":"Q_PE","Q_PE":"Q_PE",
    "Security":"Q_SE","SE":"Q_SE","Q_SE":"Q_SE",
    "Usability":"Q_US","US":"Q_US","Q_US":"Q_US",
    "Maintainability":"Q_MN","MN":"Q_MN","Q_MN":"Q_MN",
    "Compatibility":"Q_CM","CM":"Q_CM","Q_CM":"Q_CM",
    "Portability":"Q_PO","PO":"Q_PO","Q_PO":"Q_PO",
    "Reliability":"Q_RL","RL":"Q_RL","Q_RL":"Q_RL",
    "Availability":"Q_RL","AV":"Q_RL",
    "FaultTolerance":"Q_FT","FT":"Q_FT","Q_FT":"Q_FT",
    "Scalability":"Q_SC","SC":"Q_SC","Q_SC":"Q_SC",
    "Operational":"C_O","C_O":"C_O","CONSTRAINT":"C_O",
    "Look_and_Feel":"C_LF","C_LF":"C_LF",
    "Design":"C_D","C_D":"C_D",
    "Legal":"R_L","R_L":"R_L","REGULATORY":"R",
    "QUALITY":"Q","BUSINESS":"B","INTERFACE":"I",
}


def _apply_custom(main_type, short_type, raw_sub, short_sub, classes):
    if not classes:
        return short_type, True

    user_up = {c.upper(): c for c in classes}

    candidates = []
    for val in [raw_sub, short_sub, short_type, main_type]:
        if not val:
            continue
        mapped = _SYSMAP.get(val)
        if mapped:
            candidates.append(mapped.upper())
        candidates.append(val.upper().replace(" ", "_"))

    matched = None
    for code in candidates:
        if code in user_up:
            matched = user_up[code]
            break

    if not matched:
        for ov in ["OTHER", "O", "OTHERS", "?"]:
            if ov in user_up:
                matched = user_up[ov]
                break
    if not matched:
        matched = "?"

    show_sub = all(c.upper() in _SUBTYPE_CODES for c in classes)
    return matched, show_sub


st.set_page_config(
    page_title="ReqClassify",
    page_icon="📋",
    layout="wide"
)

st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-size: 20px !important;
    }
    .stTextArea textarea {
        font-size: 20px !important;
    }
    .stButton button {
        font-size: 20px !important;
        padding: 10px 24px !important;
    }
    .stSelectbox div {
        font-size: 20px !important;
    }
    span[style*="border-radius:14px"] {
        font-size: 19px !important;
        padding: 8px 22px !important;
    }
    code {
        font-size: 19px !important;
    }
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stStatusWidget"] {display: none;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("ReqClassify")
st.caption("Automated Software Requirement Classification  .  ISO 29148 + Legacy Subtypes")
st.divider()

# short codes for display badges
TYPE_MAP = {
    "FUNCTIONAL": "FR", "QUALITY": "Q", "CONSTRAINT": "C",
    "REGULATORY": "R", "INTERFACE": "I", "BUSINESS": "B"
}

SUBTYPE_MAP = {
    "Security": "SE", "PerformanceEfficiency": "PE",
    "Reliability": "RL", "Availability": "RL",
    "Usability": "US", "Maintainability": "MN", "Compatibility": "CM",
    "Portability": "PO", "FunctionalSuitability": "FS",
    "Scalability": "SC", "FaultTolerance": "FT",
    "Operational": "C_O", "OPERATIONAL": "C_O",
    "Look_and_Feel": "C_LF", "LOOK_AND_FEEL": "C_LF",
    "Legal": "R_L", "LEGAL": "R_L",
    "Design": "C_D",
    "Configuration": "FR_CFG",
    "Initialization": "FR_INIT",
    "Normal Operation": "FR_OP",
    "Shutdown": "FR_SHUT",
    "Fault Handling": "FR_FAULT",
}

# maps combined codes back to (full_type, short_type, subtype_name, subtype_code)
COMBINED_SPLIT = {
    "FR_CFG":   ("FUNCTIONAL", "FR",  "Configuration",        "FR_CFG"),
    "FR_OP":    ("FUNCTIONAL", "FR",  "Normal Operation",     "FR_OP"),
    "FR_INIT":  ("FUNCTIONAL", "FR",  "Initialization",       "FR_INIT"),
    "FR_SHUT":  ("FUNCTIONAL", "FR",  "Shutdown",             "FR_SHUT"),
    "FR_FAULT": ("FUNCTIONAL", "FR",  "Fault Handling",       "FR_FAULT"),
    "Q_PE":     ("QUALITY",    "Q",   "PerformanceEfficiency","Q_PE"),
    "Q_US":     ("QUALITY",    "Q",   "Usability",            "Q_US"),
    "Q_CM":     ("QUALITY",    "Q",   "Compatibility",        "Q_CM"),
    "Q_MN":     ("QUALITY",    "Q",   "Maintainability",      "Q_MN"),
    "Q_PO":     ("QUALITY",    "Q",   "Portability",          "Q_PO"),
    "Q_RL":     ("QUALITY",    "Q",   "Reliability",          "Q_RL"),
    "Q_SE":     ("QUALITY",    "Q",   "Security",             "Q_SE"),
    "Q_FT":     ("QUALITY",    "Q",   "FaultTolerance",       "Q_FT"),
    "Q_SC":     ("QUALITY",    "Q",   "Scalability",          "Q_SC"),
    "C_O":      ("CONSTRAINT", "C",   "Operational",          "C_O"),
    "C_LF":     ("CONSTRAINT", "C",   "Look and Feel",        "C_LF"),
    "C_D":      ("CONSTRAINT", "C",   "Design",               "C_D"),
    "R_L":      ("REGULATORY", "R",   "Legal",                "R_L"),
    "FR":       ("FUNCTIONAL", "FR",  None,                   None),
    "NFR":      ("QUALITY",    "Q",   None,                   None),
}

TYPE_COLORS = {
    "FUNCTIONAL": "#2E86AB",
    "QUALITY": "#4A90D9",
    "CONSTRAINT": "#5C6BC0",
    "REGULATORY": "#7B68EE",
    "INTERFACE": "#26A69A",
    "BUSINESS": "#66BB6A",
    "OTHER": "#78909C",
    "Unknown": "#78909C",
}

SUBTYPE_COLORS = {
    "Security":              "#5C6BC0",
    "PerformanceEfficiency": "#2E86AB",
    "Reliability":           "#26A69A",
    "Availability":          "#26A69A",
    "Usability":             "#4A90D9",
    "Maintainability":       "#7B68EE",
    "Compatibility":         "#66BB6A",
    "Portability":           "#78909C",
    "Scalability":           "#FFA726",
    "FaultTolerance":        "#EF5350",
    "FunctionalSuitability": "#42A5F5",
    "Configuration":         "#1565C0",
    "Initialization":        "#0277BD",
    "Normal Operation":      "#00838F",
    "Shutdown":              "#558B2F",
    "Fault Handling":        "#E65100",
    "Operational":           "#8D6E63",
    "Look and Feel":         "#EC407A",
    "Look_and_Feel":         "#EC407A",
    "Design":                "#AB47BC",
    "Legal":                 "#6A1B9A",
    "Unknown":               "#78909C",
}


def badge(label: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:white;padding:6px 18px;'
        f'border-radius:14px;font-weight:bold;font-size:14px;">{label}</span>'
    )


def safe_parse_json(raw_text: str):
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
    if "[" in raw_text and "]" in raw_text:
        raw_text = raw_text[raw_text.index("["):raw_text.rindex("]") + 1]
    return json.loads(raw_text)


st.sidebar.header("Settings")

selected_dataset = st.sidebar.selectbox(
    "Example Dataset for RAG",
    list(DATASET_CONFIG.keys()),
    index=0
)
dataset_cfg = get_dataset_config(selected_dataset)
active_table = dataset_cfg["table_name"]
st.sidebar.caption(dataset_cfg.get("description", ""))

iso_top_k = st.sidebar.slider("ISO chunks (top-k)", 1, 6, 3)
ex_top_k = st.sidebar.slider("Example chunks (top-k)", 1, 6, 3)
batch_chunk = st.sidebar.number_input("Batch chunk size", min_value=5, max_value=100, value=20, step=5)
sleep_between = st.sidebar.number_input("Delay between calls (sec)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

tab1, tab2 = st.tabs(["📋 Classify", "📂 Batch CSV"])

with tab1:
    req_input = st.text_area(
        "Enter requirements",
        height=180,
        placeholder="",
        key="req_input_box"
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        classify_btn = st.button("🔍 Classify", type="primary", use_container_width=True)
    with col_btn2:
        clear_btn = st.button("🗑️ Clear History", use_container_width=False)

    if clear_btn:
        st.session_state["chat_history"] = []
        st.rerun()

    if classify_btn:
        all_lines = [r.strip() for r in req_input.strip().splitlines() if r.strip()]
        if not all_lines:
            st.warning("Please enter at least one requirement.")
        else:
            custom_classes = []
            requirements = all_lines
            if len(all_lines) > 1 and "classify" in all_lines[0].lower() and "into" in all_lines[0].lower():
                custom_classes = _parse_custom_classes(all_lines[0])
                requirements = all_lines[1:]
                if custom_classes:
                    st.info(f"Classifying into: {', '.join(custom_classes)}")

            for single_req in requirements:
                with st.spinner("Classifying..."):
                    try:
                        stage1 = classify_stage1_type(
                            single_req,
                            iso_top_k=iso_top_k,
                            ex_top_k=ex_top_k,
                            table_name=active_table,
                            custom_classes=custom_classes or None,
                            dataset_name=selected_dataset
                        )
                        main_type = stage1["type"]
                        confidence = stage1["confidence"]
                        ambiguous = stage1["ambiguous"]
                        amb_note = stage1["note"]
                        short_type = TYPE_MAP.get(main_type, main_type)

                        quality_subtype = stage1.get("subtype")
                        short_subtype = SUBTYPE_MAP.get(quality_subtype, quality_subtype) if quality_subtype else None

                        if quality_subtype is None and stage2_for_dataset(
                            main_type=main_type,
                            dataset_name=selected_dataset,
                            table_name=active_table,
                        ):
                            stage2 = classify_stage2_quality(
                                single_req,
                                iso_top_k=iso_top_k,
                                ex_top_k=ex_top_k,
                                main_type=main_type,
                                table_name=active_table,
                                custom_classes=custom_classes or None,
                                dataset_name=selected_dataset
                            )
                            quality_subtype = stage2["subtype"]
                            short_subtype = SUBTYPE_MAP.get(quality_subtype, quality_subtype)

                        display_classes = custom_classes or get_dataset_valid_classes(selected_dataset, active_table)

                        if display_classes:
                            if main_type in display_classes:
                                short_type = main_type
                                quality_subtype = None
                                short_subtype = None
                            else:
                                short_type, show_sub = _apply_custom(
                                    main_type, short_type,
                                    quality_subtype or "", short_subtype or "",
                                    display_classes
                                )
                                main_type = short_type
                                if not show_sub or short_type.upper() in _SUBTYPE_CODES:
                                    quality_subtype = None
                                    short_subtype = None

                        st.session_state["chat_history"].append({
                            "requirement": single_req,
                            "type": main_type,
                            "short_type": short_type,
                            "confidence": confidence,
                            "ambiguous": ambiguous,
                            "amb_note": amb_note,
                            "quality_subtype": quality_subtype,
                            "short_subtype": short_subtype,
                            "feedback_saved": False,
                            "feedback_label": None,
                        })
                    except Exception as e:
                        st.error(f"Classification failed: {str(e)}")

    st.divider()

    if not st.session_state["chat_history"]:
        st.info("No classifications yet. Enter a requirement above and click Classify.")
    else:
        st.markdown(f"**{len(st.session_state['chat_history'])} requirement(s) classified**")
        st.divider()

        for idx, item in enumerate(reversed(st.session_state["chat_history"])):
            real_idx = len(st.session_state["chat_history"]) - 1 - idx
            raw_label = item["type"]

            if item.get("ambiguous"):
                stage1_type, stage1_short = "OTHER", "O"
                stage2_name, stage2_short = None, None
            elif raw_label in COMBINED_SPLIT:
                stage1_type, stage1_short, stage2_name, stage2_short = COMBINED_SPLIT[raw_label]
            elif item.get("quality_subtype"):
                stage1_type  = item["type"]
                stage1_short = item["short_type"]
                stage2_name  = item["quality_subtype"]
                stage2_short = item["short_subtype"]
                combined_key = f"{stage1_short}_{stage2_short}" if stage2_short else None
                if combined_key and combined_key in COMBINED_SPLIT:
                    stage1_type, stage1_short, stage2_name, stage2_short = COMBINED_SPLIT[combined_key]
            else:
                stage1_type  = item["type"]
                stage1_short = item["short_type"]
                stage2_name, stage2_short = None, None

            with st.container():
                st.markdown(f"**#{real_idx + 1}** `{item['requirement']}`")

                col1, col2 = st.columns([3, 3])
                with col1:
                    color = TYPE_COLORS.get(stage1_type, "#78909C")
                    badge_text = f"{stage1_type}  ({stage1_short})" if stage1_type != stage1_short else stage1_type
                    st.markdown(badge(badge_text, color), unsafe_allow_html=True)
                with col2:
                    if stage2_name and not item.get("ambiguous"):
                        sub_color = SUBTYPE_COLORS.get(stage2_name, "#78909C")
                        sub_text = f"{stage2_name}  ({stage2_short})" if stage2_short else stage2_name
                        st.markdown(badge(sub_text, sub_color), unsafe_allow_html=True)

                st.markdown("")

                if item["feedback_saved"]:
                    st.success(f"Saved: **{item['feedback_label']}**")
                else:
                    with st.expander(f"Correct this classification (#{real_idx + 1})"):
                        fb_col1, fb_col2, fb_col3 = st.columns([2, 2, 1])

                        with fb_col1:
                            correct_type = st.selectbox(
                                "Correct Type",
                                ["FR", "Q", "C", "R", "B"],
                                index=["FR", "Q", "C", "R", "B"].index(
                                    item["short_type"]
                                ) if item["short_type"] in ["FR", "Q", "C", "R", "B"] else 0,
                                key=f"fb_type_{real_idx}"
                            )

                        with fb_col2:
                            if correct_type == "FR":
                                sub_options = ["", "CFG", "INIT", "OP", "SHUT", "FAULT"]
                            elif correct_type == "Q":
                                sub_options = ["", "SE", "PE", "A", "FT", "US", "MN", "CM", "PO", "FS", "SC"]
                            elif correct_type == "C":
                                sub_options = ["", "O", "LF"]
                            elif correct_type == "R":
                                sub_options = ["", "L"]
                            else:
                                sub_options = [""]

                            correct_subtype = st.selectbox(
                                f"Subtype (for {correct_type})",
                                sub_options,
                                key=f"fb_sub_{real_idx}"
                            )

                        with fb_col3:
                            st.markdown("<br>", unsafe_allow_html=True)
                            if st.button("Save", key=f"fb_save_{real_idx}", use_container_width=True):
                                combined = f"{correct_type}_{correct_subtype}" if correct_subtype else correct_type
                                try:
                                    store_human_feedback(item["requirement"], combined, table_name=active_table)
                                    st.session_state["chat_history"][real_idx]["feedback_saved"] = True
                                    st.session_state["chat_history"][real_idx]["feedback_label"] = combined
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Save failed: {str(e)}")

                st.divider()

with tab2:
    st.info("Upload a CSV with requirements. First column is used automatically — any header name works.")

    batch_cls_input = st.text_input(
        "Classify into specific classes (Optional):",
        placeholder="e.g. FR, Q_PE, Q_SE   or   FR_OP, FR_CFG, Q_PO"
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    rows = []
    b_custom = []

    if uploaded:
        df_upload = pd.read_csv(uploaded)
        first_col = df_upload.iloc[:, 0].dropna().str.strip()
        rows = [r for r in first_col.tolist() if len(r) > 10]

        if rows and "classify" in rows[0].lower() and "into" in rows[0].lower():
            b_custom = _parse_custom_classes(rows[0])
            if b_custom:
                rows = rows[1:]

        st.success(f"{len(rows)} requirements loaded")
        st.dataframe({"requirement": rows[:5]}, use_container_width=True)

    if st.button("Classify All", type="primary", use_container_width=True) and rows:
        if batch_cls_input.strip():
            b_custom = _parse_custom_classes("classify into: " + batch_cls_input.strip())

        if b_custom:
            st.info(f"Classifying into: {', '.join(b_custom)}")

        results = []
        progress = st.progress(0)
        status = st.empty()
        total = len(rows)
        error_count = 0

        for i in range(0, total, batch_chunk):
            chunk = rows[i:i + batch_chunk]
            status.text(f"Processing {min(i + batch_chunk, total)}/{total}... (errors: {error_count})")

            try:
                data = classify_batch_single_shot(
                    chunk,
                    iso_top_k=iso_top_k,
                    ex_top_k=ex_top_k,
                    table_name=active_table,
                    dataset_name=selected_dataset,
                    custom_classes=b_custom or None,
                )

                for item in data:
                    idx = item.get("id", 1) - 1
                    req_text = chunk[idx] if 0 <= idx < len(chunk) else item.get("requirement", "")
                    main_type = item.get("type", "Unknown")
                    raw_subtype = item.get("subtype", "") or ""

                    combined_label = None
                    short_type_raw = TYPE_MAP.get(main_type, main_type)
                    short_sub_raw  = SUBTYPE_MAP.get(raw_subtype, raw_subtype) if raw_subtype else ""

                    if main_type in COMBINED_SPLIT:
                        combined_label = main_type
                    elif short_sub_raw:
                        candidate = f"{short_type_raw}_{short_sub_raw}"
                        if candidate in COMBINED_SPLIT:
                            combined_label = candidate

                    if combined_label and combined_label in COMBINED_SPLIT:
                        _, csv_type, csv_subtype, _ = COMBINED_SPLIT[combined_label]
                    else:
                        csv_type    = short_type_raw
                        csv_subtype = raw_subtype

                    results.append((req_text, csv_type, csv_type, csv_subtype, csv_subtype))

            except Exception as e:
                error_count += len(chunk)
                for req_text in chunk:
                    results.append((req_text, "ERROR", "ERR", str(e)[:80], ""))

            progress.progress(min((i + batch_chunk) / total, 1.0))
            time.sleep(sleep_between)

        status.text(f"Done! {total - error_count}/{total} classified successfully.")

        display_results = []
        for r in results:
            display_results.append({
                "Requirement": r[0],
                "Type": r[2],
                "Subtype": r[4],
            })

        result_df = pd.DataFrame(display_results)
        st.divider()
        st.subheader("Results")
        st.dataframe(result_df, use_container_width=True)
        st.divider()
        st.subheader("Classification Distribution")

        col_dist1, col_dist2 = st.columns(2)

        with col_dist1:
            st.markdown("**By Type**")
            type_counts = result_df["Type"].value_counts()
            st.bar_chart(type_counts)

        with col_dist2:
            st.markdown("**By Subtype**")
            sub_df = result_df[result_df["Subtype"] != ""]
            if not sub_df.empty:
                subtype_counts = sub_df["Subtype"].value_counts()
                st.bar_chart(subtype_counts)
            else:
                st.info("No subtypes found.")

        st.divider()
        st.markdown("**Summary**")
        summary_cols = st.columns(len(type_counts))
        for i, (label, count) in enumerate(type_counts.items()):
            with summary_cols[i]:
                st.metric(label=label, value=count)

        st.download_button(
            "Download Results CSV",
            result_df.to_csv(index=False, encoding="utf-8"),
            file_name="classification_results.csv",
            mime="text/csv",
            use_container_width=True
        )

st.divider()
st.markdown(
    "<center><small>ReqClassify · ISO Aligned Requirement Classification Framework</small></center>",
    unsafe_allow_html=True
)