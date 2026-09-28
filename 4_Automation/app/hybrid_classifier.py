# hybrid_classifier.py

# in this file main classification of the requirements done with hybrid mechanism.
#   1) RAG  - first similar labeled examples from supabase(table) are taken
#   2) LLM  - if RAG is not sure,  then gemini used to classify
#   Stage 1 (classify_stage1_type) -> find the main type
#       - search the most similar labeled examples of the selected dataset
#       - if the top example score is >= 0.80, then that label is taken
#       - else get document context + examples and ask gemini
#         (FUNCTIONAL / QUALITY / CONSTRAINT / REGULATORY or the dataset classes)
#   Stage 2 (classify_stage2_quality) -> find the subtype, only if needed
#       - FUNCTIONAL -> Configuration / Initialization / Normal Operation /
#                       Shutdown / Fault Handling (only for datasets like autosar)
#       - QUALITY    -> Security, PerformanceEfficiency, Usability ... (uses ISO 25010)
#       - REGULATORY -> always Legal
#       - CONSTRAINT -> no subtype
# DATASET_CONFIG    = settings per dataset (which table, which classes, when stage 2 runs)
# DATASET_LABEL_MAP = maps dataset labels (like PE, Q_SE, FR_CFG) to (type, subtype)
# thresholds        = 0.80 ver smilar example available

import re
import time
from gemini_client import get_client
from rag_layer import (
    embed_text,
    retrieve_iso_context_by_vec,
    retrieve_example_context_by_vec,
)

LLM_MODEL = "models/gemini-2.5-flash-lite"

VALID_TYPES = {
    "FUNCTIONAL", "QUALITY", "CONSTRAINT",
    "REGULATORY", "INTERFACE", "BUSINESS"
}

VALID_SUBTYPES = {
    "Security", "PerformanceEfficiency", "FaultTolerance",
    "Usability", "Maintainability", "Compatibility",
    "Portability", "FunctionalSuitability", "Reliability",
    "Scalability", "Operational", "Look_and_Feel", "Legal"
}

FUNCTIONAL_SUBTYPES = {
    "Configuration",
    "Initialization",
    "Normal Operation",
    "Shutdown",
    "Fault Handling"
}

RAG_EXACT_MATCH_THRESHOLD = 0.80
RAG_SIMILAR_MATCH_THRESHOLD = 0.0

DATASET_THRESHOLDS = {
    "examples_userstory":   (0.80, 0.0),
    "examples_promise":     (0.80, 0.0),
    "examples_aerobert":    (0.80, 0.0),
    "examples_autosar":     (0.80, 0.0),
    "requirement_examples": (0.80, 0.0),
}


def get_thresholds(table_name: str):
    return DATASET_THRESHOLDS.get(
        table_name,
        (RAG_EXACT_MATCH_THRESHOLD, RAG_SIMILAR_MATCH_THRESHOLD)
    )

#checking each and every dataset examples for different correction
DATASET_CONFIG = {
    "Default (All)": {
        "table_name": "requirement_examples",
        "stage2_types": {"FUNCTIONAL", "QUALITY", "REGULATORY"},
        "functional_subtypes_enabled": True,
        "valid_classes": [
            "FR", "Q_SE", "Q_PE", "Q_US", "Q_MN", "Q_CM",
            "Q_PO", "Q_RL", "Q_FT", "Q_SC", "Q_LF", "Q_O", "C", "R"
        ],
    },

    "User Story": {
        "table_name": "examples_userstory",
        "stage2_types": {"QUALITY"},
        "functional_subtypes_enabled": False,
        "valid_classes": [
            "FR", "Q_US", "Q_CM", "Q_MN", "Q_PE",
            "Q_PO", "Q_RL", "Q_SE"
        ],
    },

    "Autosar CAN": {
        "table_name": "examples_autosar",
        "stage2_types": {"FUNCTIONAL", "QUALITY"},
        "functional_subtypes_enabled": True,
        "valid_classes": [
            "FR", "FR_CFG", "FR_OP", "FR_INIT", "FR_SHUT", "FR_FAULT",
            "Q_PE", "Q_PO", "Q_US", "Q_RL", "Q_SC", "NFR"
        ],
    },

    "PROMISE": {
        "table_name": "examples_promise",
        "description": " ",
        "stage2_types": {"QUALITY", "REGULATORY"},
        "functional_subtypes_enabled": False,
        "valid_classes": [
            "FR",
            "Q_PE", "Q_SE", "Q_US", "Q_MN", "Q_PO",
            "Q_SC", "Q_FT", "Q_RL",
            "C", "R_L"
        ],
    },

    "AeroBERT": {
        "table_name": "examples_aerobert",
        "description": " ",
        "stage2_types": {"FUNCTIONAL", "QUALITY"},
        "functional_subtypes_enabled": False,
        "valid_classes": ["FR", "Q_PE", "C"],
    },
}


DATASET_LABEL_MAP = {
    "examples_userstory": {
        "FR": ("FUNCTIONAL", None),
        "US": ("QUALITY", "Usability"),
        "CM": ("QUALITY", "Compatibility"),
        "MN": ("QUALITY", "Maintainability"),
        "PE": ("QUALITY", "PerformanceEfficiency"),
        "PO": ("QUALITY", "Portability"),
        "RL": ("QUALITY", "Reliability"),
        "SE": ("QUALITY", "Security"),
        "FT": ("QUALITY", "FaultTolerance"),
        "SC": ("QUALITY", "Scalability"),
        "Q_US": ("QUALITY", "Usability"),
        "Q_CM": ("QUALITY", "Compatibility"),
        "Q_MN": ("QUALITY", "Maintainability"),
        "Q_PE": ("QUALITY", "PerformanceEfficiency"),
        "Q_PO": ("QUALITY", "Portability"),
        "Q_RL": ("QUALITY", "Reliability"),
        "Q_SE": ("QUALITY", "Security"),
        "Q_FT": ("QUALITY", "FaultTolerance"),
        "Q_SC": ("QUALITY", "Scalability"),
    },

    "examples_autosar": {
        "FR":       ("FUNCTIONAL", None),
        "FR_CFG":   ("FUNCTIONAL", "Configuration"),
        "FR_OP":    ("FUNCTIONAL", "Normal Operation"),
        "FR_INIT":  ("FUNCTIONAL", "Initialization"),
        "FR_SHUT":  ("FUNCTIONAL", "Shutdown"),
        "FR_FAULT": ("FUNCTIONAL", "Fault Handling"),
        "NFR":      ("QUALITY", None),
        "Q_PE": ("QUALITY", "PerformanceEfficiency"),
        "Q_PO": ("QUALITY", "Portability"),
        "Q_US": ("QUALITY", "Usability"),
        "Q_RL": ("QUALITY", "Reliability"),
        "Q_SC": ("QUALITY", "Scalability"),
        "Configuration":    ("FUNCTIONAL", "Configuration"),
        "Initialization":   ("FUNCTIONAL", "Initialization"),
        "Normal Operation": ("FUNCTIONAL", "Normal Operation"),
        "Shutdown":         ("FUNCTIONAL", "Shutdown"),
        "Fault Handling":   ("FUNCTIONAL", "Fault Handling"),
        "Performance": ("QUALITY", "PerformanceEfficiency"),
        "Portability": ("QUALITY", "Portability"),
        "Reliability": ("QUALITY", "Reliability"),
        "Scalability": ("QUALITY", "Scalability"),
        "Usability":   ("QUALITY", "Usability"),
    },

    "examples_promise": {
        "FR":   ("FUNCTIONAL", None),
        "Q_PE": ("QUALITY", "PerformanceEfficiency"),
        "Q_SE": ("QUALITY", "Security"),
        "Q_US": ("QUALITY", "Usability"),
        "Q_MN": ("QUALITY", "Maintainability"),
        "Q_PO": ("QUALITY", "Portability"),
        "Q_SC": ("QUALITY", "Scalability"),
        "Q_FT": ("QUALITY", "FaultTolerance"),
        "Q_RL": ("QUALITY", "Reliability"),
        "C_O":  ("CONSTRAINT", None),
        "C_LF": ("CONSTRAINT", None),
        "R_L":  ("REGULATORY", "Legal"),
        "PE": ("QUALITY", "PerformanceEfficiency"),
        "SE": ("QUALITY", "Security"),
        "US": ("QUALITY", "Usability"),
        "MN": ("QUALITY", "Maintainability"),
        "PO": ("QUALITY", "Portability"),
        "SC": ("QUALITY", "Scalability"),
        "FT": ("QUALITY", "FaultTolerance"),
        "A":  ("QUALITY", "Reliability"),
        "O":  ("QUALITY", "Operational"),
        "LF": ("QUALITY", "Look_and_Feel"),
        "L":  ("REGULATORY", "Legal"),
    },

    "examples_aerobert": {
        "FR":   ("FUNCTIONAL", None),
        "Q_PE": ("QUALITY", "PerformanceEfficiency"),
        "C":  ("CONSTRAINT", None),
        "PE":   ("QUALITY", "PerformanceEfficiency"),
        "D":    ("CONSTRAINT", "Operational"),
    },

    "requirement_examples": {
        "FR":       ("FUNCTIONAL", None),
        "FR_OP":    ("FUNCTIONAL", "Normal Operation"),
        "FR_CFG":   ("FUNCTIONAL", "Configuration"),
        "FR_INIT":  ("FUNCTIONAL", "Initialization"),
        "FR_SHUT":  ("FUNCTIONAL", "Shutdown"),
        "FR_FAULT": ("FUNCTIONAL", "Fault Handling"),
        "Q_SE": ("QUALITY", "Security"),
        "Q_PE": ("QUALITY", "PerformanceEfficiency"),
        "Q_US": ("QUALITY", "Usability"),
        "Q_MN": ("QUALITY", "Maintainability"),
        "Q_CM": ("QUALITY", "Compatibility"),
        "Q_PO": ("QUALITY", "Portability"),
        "Q_RL": ("QUALITY", "Reliability"),
        "Q_FT": ("QUALITY", "FaultTolerance"),
        "Q_SC": ("QUALITY", "Scalability"),
        "SE": ("QUALITY", "Security"),
        "PE": ("QUALITY", "PerformanceEfficiency"),
        "US": ("QUALITY", "Usability"),
        "MN": ("QUALITY", "Maintainability"),
        "CM": ("QUALITY", "Compatibility"),
        "PO": ("QUALITY", "Portability"),
        "RL": ("QUALITY", "Reliability"),
        "FT": ("QUALITY", "FaultTolerance"),
        "SC": ("QUALITY", "Scalability"),
        "C_O":  ("CONSTRAINT", None),
        "C_LF": ("CONSTRAINT", None),
        "R": ("REGULATORY", "Legal"),
        "L": ("REGULATORY", "Legal"),
    },
}


def get_dataset_config(dataset_name: str = "Default (All)") -> dict:
    if dataset_name in DATASET_CONFIG:
        return DATASET_CONFIG[dataset_name]
    for cfg in DATASET_CONFIG.values():
        if dataset_name == cfg.get("table_name"):
            return cfg
    return DATASET_CONFIG["Default (All)"]


def get_table_name(dataset_name: str = "Default (All)") -> str:
    return get_dataset_config(dataset_name)["table_name"]


def get_dataset_valid_classes(dataset_name: str = None, table_name: str = None) -> list:
    cfg = get_dataset_config(table_name or dataset_name or "Default (All)")
    return cfg.get("valid_classes", [])


def stage2_for_dataset(
        main_type: str,
        dataset_name: str = "Default (All)",
        table_name: str | None = None
) -> bool:
    cfg = get_dataset_config(table_name or dataset_name or "Default (All)")
    return main_type in cfg.get("stage2_types", set())


def should_classify_functional_subtypes(
        dataset_name: str = "Default (All)",
        table_name: str = None
) -> bool:
    cfg = get_dataset_config(table_name or dataset_name or "Default (All)")
    return cfg.get("functional_subtypes_enabled", False)


def _extract_top_rag_label(example_context: str):
    if not example_context:
        return None, None

    score_match = re.search(
        r"\[Example\s+1\s*\|\s*score\s*=\s*([0-9.]+)\]",
        example_context,
        re.IGNORECASE
    )
    if not score_match:
        score_match = re.search(
            r"\[Example\s+\d+\s*\|\s*score\s*=\s*([0-9.]+)\]",
            example_context,
            re.IGNORECASE
        )

    label_match = re.search(
        r"^\s*Label\s*:\s*([A-Za-z0-9_/ -]+)\s*$",
        example_context,
        re.IGNORECASE | re.MULTILINE
    )

    if not score_match or not label_match:
        print("=== RAG PARSE FAILED — raw context snippet ===")
        print(repr(example_context[:400]))
        return None, None

    try:
        score = float(score_match.group(1))
    except ValueError:
        return None, None

    return label_match.group(1).strip(), score


def _normalize_label(label: str) -> str:
    return label.strip().upper().replace("-", "_").replace(" ", "_")


def _rag_label_to_result(label: str, table_name: str = None):
    if not label:
        return None

    normalized = _normalize_label(label)

    if table_name:
        ds_map = DATASET_LABEL_MAP.get(table_name, {})
        if normalized in ds_map:
            return ds_map[normalized]
        if label.strip() in ds_map:
            return ds_map[label.strip()]
        if label.strip().upper() in ds_map:
            return ds_map[label.strip().upper()]

    functional_aliases = {
        "FR_OP":               "Normal Operation",
        "FR_NORMAL_OPERATION": "Normal Operation",
        "NORMAL_OPERATION":    "Normal Operation",
        "FR_INIT":             "Initialization",
        "FR_INITIALIZATION":   "Initialization",
        "INITIALIZATION":      "Initialization",
        "FR_CFG":              "Configuration",
        "FR_CONFIGURATION":    "Configuration",
        "CONFIGURATION":       "Configuration",
        "FR_SHUT":             "Shutdown",
        "SHUTDOWN":            "Shutdown",
        "FR_FAULT":            "Fault Handling",
        "FR_FAULT_HANDLING":   "Fault Handling",
        "FAULT_HANDLING":      "Fault Handling",
    }
    if normalized in functional_aliases:
        return "FUNCTIONAL", functional_aliases[normalized]

    for valid_type in VALID_TYPES:
        if normalized == _normalize_label(valid_type):
            return valid_type, None

    for subtype in FUNCTIONAL_SUBTYPES:
        if normalized == _normalize_label(subtype):
            return "FUNCTIONAL", subtype

    for subtype in VALID_SUBTYPES:
        if normalized == _normalize_label(subtype):
            if subtype in {"Operational", "Look_and_Feel"}:
                return "QUALITY", subtype
            if subtype == "Legal":
                return "REGULATORY", subtype
            return "QUALITY", subtype

    return None


def _build_rag_examples_block(ex_context: str) -> str:
    if not ex_context or not ex_context.strip():
        return ""
    return f"""
REAL LABELED EXAMPLES FROM KNOWLEDGE BASE:
The following examples are verified human-labeled requirements similar to the current requirement.
Use these examples as reference, but rely primarily on the ISO definitions in the reference context for your final decision.

{ex_context}
"""


_CLASS_MEANINGS = {
    "FR":    "FR (Functional — an action, function or service the system performs, like process, send, display, calculate. NOT physical design, installation, placement or hardware constraints)",
    "NFR":   "NFR (Non-Functional — quality-related behaviour)",
    "FR_CFG":   "FR_CFG (Functional/Configuration — setup or configurable behaviour)",
    "FR_OP":    "FR_OP (Functional/Normal Operation — regular runtime behaviour)",
    "FR_INIT":  "FR_INIT (Functional/Initialization — startup or preparation before use)",
    "FR_SHUT":  "FR_SHUT (Functional/Shutdown — stopping, disabling, or de-initialization behaviour)",
    "FR_FAULT": "FR_FAULT (Functional/Fault Handling — behaviour during error, failure, or abnormal condition)",
    "Configuration":    "Configuration (Functional subtype — setup or configurable behaviour)",
    "Initialization":   "Initialization (Functional subtype — startup or preparation before use)",
    "Normal Operation": "Normal Operation (Functional subtype — regular runtime behaviour)",
    "Shutdown":         "Shutdown (Functional subtype — stopping, disabling, or de-initialization behaviour)",
    "Fault Handling":   "Fault Handling (Functional subtype — behaviour during error, failure, or abnormal condition)",
    "US": "US (Usability — ease of use and user interaction quality)",
    "CM": "CM (Compatibility — integration or coexistence with other systems)",
    "MN": "MN (Maintainability — easy to modify, test, or maintain)",
    "PE": "PE (PerformanceEfficiency — time, speed, response, throughput, latency)",
    "PO": "PO (Portability — operation across platforms, devices, or environments)",
    "RL": "RL (Reliability — stable and dependable operation)",
    "SE": "SE (Security — protection, access control, privacy, or encryption)",
    "SC": "SC (Scalability — ability to handle growing users, data, or load)",
    "FT": "FT (FaultTolerance — recovery, failover, resilience, or backup)",
    "A":  "A (Availability — system access or uptime)",
    "O":  "O (Operational Quality — environment, deployment, or operating condition as a quality concern)",
    "LF": "LF (Look and Feel — visual appearance, layout, or branding as a quality concern)",
    "L":  "L (Legal/Regulatory — compliance with laws, rules, or standards)",
    "D":  "D (Design Constraint — physical or structural design condition)",
    "Q_US": "Q_US (Quality/Usability — ease of use and user interaction quality)",
    "Q_CM": "Q_CM (Quality/Compatibility — integration or coexistence with other systems)",
    "Q_MN": "Q_MN (Quality/Maintainability — easy to modify, test, or maintain)",
    "Q_PE": "Q_PE (Quality/PerformanceEfficiency — time, speed, response, throughput, latency)",
    "Q_PO": "Q_PO (Quality/Portability — operation across platforms, devices, or environments)",
    "Q_RL": "Q_RL (Quality/Reliability — stable and dependable operation)",
    "Q_SE": "Q_SE (Quality/Security — protection, access control, privacy, or encryption)",
    "Q_FT": "Q_FT (Quality/FaultTolerance — recovery, failover, resilience, or backup)",
    "Q_SC": "Q_SC (Quality/Scalability — growing users, data, or load)",
    "C":    "C (Constraint — restriction on design, platform, environment, or implementation)",
    "R_L":  "R_L (Regulatory/Legal — compliance with laws, rules, or standards)",
    "OTHER": "OTHER (none of the given classes fits, e.g. a design / installation / physical constraint when no class for that is given)",
}


def _build_custom_classes_block(custom_classes: list) -> str:
    if not custom_classes:
        return ""
    classes_with_meaning = []
    for c in custom_classes:
        meaning = _CLASS_MEANINGS.get(c, _CLASS_MEANINGS.get(c.upper(), c))
        classes_with_meaning.append(f"  - {meaning}")
    classes_str = ", ".join(custom_classes)
    meanings_block = "\n".join(classes_with_meaning)
    return f"""
USER-DEFINED CLASSIFICATION TASK:
Classify this requirement into exactly one of these classes:

{meanings_block}

Rules:
- Final answer must be exactly one of these codes: {classes_str}
- Do not output any other label.
- Decide based on the meaning and purpose of the requirement, not only single keywords.
- First decide what kind of requirement it is: function, quality, design/constraint or regulatory.
  If that kind is not in the list above, answer OTHER. Do not force a wrong class.
"""


# which ISO type a class belongs to, None if we dont know (then no check)
def _class_type(label):
    if not label:
        return None
    l = label.strip().upper().replace(" ", "_")
    if l in ["FR", "FUNCTIONAL"] or l.startswith("FR_"):
        return "FUNCTIONAL"
    if l in ["Q", "NFR", "QUALITY", "PE", "SE", "US", "MN", "CM", "PO", "RL", "FT", "SC", "A", "PERFORMANCE"] or l.startswith("Q_"):
        return "QUALITY"
    if l in ["C", "D", "CONSTRAINT", "DESIGN"] or l.startswith("C_"):
        return "CONSTRAINT"
    if l in ["R", "L", "REGULATORY", "LEGAL"] or l.startswith("R_"):
        return "REGULATORY"
    if l in ["I", "INTERFACE"]:
        return "INTERFACE"
    if l in ["B", "BUSINESS"]:
        return "BUSINESS"
    return None


_gemini_client = None


def _get_cached_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = get_client()
    return _gemini_client


def _call_gemini(prompt: str, max_retries: int = 8) -> str:
    client = _get_cached_client()
    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "thinking_config": {"thinking_budget": 0}
                }
            )
            return (resp.text or "").strip()
        except Exception as e:
            err_str = str(e).lower()
            is_overload = any(
                x in err_str
                for x in ["503", "overload", "unavailable", "429", "exhausted"]
            )
            if attempt == max_retries - 1:
                raise
            wait = delay * 3 if is_overload else delay
            time.sleep(min(wait, 60))
            delay = min(delay * 2, 30)


def _extract_label(raw: str, valid_set: set, custom_classes: list = None) -> str:
    search_sets = []
    if custom_classes:
        search_sets.append(custom_classes)
    search_sets.append(valid_set)

    raw_stripped = raw.strip()
    match = re.search(
        r"FINAL ANSWER\s*:\s*([A-Za-z0-9_ ]+)",
        raw,
        re.IGNORECASE
    )
    if match:
        candidate = match.group(1).strip()
        for s in search_sets:
            for label in s:
                if label.upper() == candidate.upper():
                    return label

    for s in search_sets:
        for label in s:
            if label.upper() == raw_stripped.upper():
                return label

    for s in search_sets:
        for label in s:
            if re.search(r"\b" + re.escape(label) + r"\b", raw, re.IGNORECASE):
                return label

    return "Unknown"


def _extract_confidence(raw: str) -> str:
    match = re.search(r"CONFIDENCE\s*:\s*(HIGH|MEDIUM|LOW)", raw, re.IGNORECASE)
    return match.group(1).upper() if match else "MEDIUM"


def _extract_ambiguity(raw: str) -> tuple:
    amb_match = re.search(r"AMBIGUOUS\s*:\s*(YES|NO)", raw, re.IGNORECASE)
    note_match = re.search(r"AMBIGUITY NOTE\s*:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE)
    is_ambiguous = amb_match.group(1).upper() == "YES" if amb_match else False
    note = note_match.group(1).strip() if note_match else ""
    return is_ambiguous, note


_FUNCTIONAL_SUBTYPE_DEFINITIONS = """
## SUBTYPE DEFINITIONS

### CONFIGURATION
PURPOSE:
The requirement exists to give engineers or integrators control over HOW the module
behaves across different projects or deployments. The module itself does not act —
it exposes a settable property. The value is determined outside the module at
build time or integration time, not during runtime execution.

SELF-TEST: Ask "Is this requirement saying that something about the module
CAN BE SET by a project engineer or integrator at Pre-Compile, Link-Time,
or Post-Build time?"
If YES → Configuration.

Correct examples:
- Bit timing and baud rate shall be configurable (engineer sets the value)
- Hardware reception filter shall be statically configurable
- Timeout values shall be configurable per connection
- Polling or interrupt mode shall be Pre-Compile-Time configurable per event
- The number of TX hardware objects shall be configurable

DO NOT classify as Configuration:
- "The driver shall initialize the baud rate registers at startup"
  → Initialization (module acts at startup, engineer is not setting a value)
- "The module shall transmit using the configured baud rate"
  → Normal Operation (runtime behaviour using an already-set value)
- "The module shall re-initialize the baud rate after BusOff"
  → Fault Handling (triggered by an error condition)

---

### INITIALIZATION
PURPOSE:
The requirement describes actions the MODULE must take ONCE when it transitions
from unpowered or reset state to ready state. The module is the actor — it sets
up its own registers, variables, and hardware objects before communication starts.

SELF-TEST: Ask "Is this a one-time startup sequence where the MODULE ITSELF
prepares its internal state before normal use begins?"
If YES → Initialization.

Correct examples:
- The CAN Driver shall initialize all registers of the CAN Hardware Unit at startup
- The initialization function shall only be called once during startup
- All module global variables shall be initialized before first use
- The driver shall be initialized during the power-up reset sequence of the ECU

DO NOT classify as Initialization:
- "Baud rate shall be configurable"
  → Configuration (engineer sets value, module does not act)
- "The module shall re-initialize after BusOff"
  → Fault Handling (triggered by error, not regular startup)
- "The module shall initialize the hardware to ignore remote frames"
  → If it is a one-time startup action → Initialization
  → If it is a configurable parameter an engineer turns on/off → Configuration

---

### NORMAL OPERATION
PURPOSE:
The requirement describes what the module does continuously and repeatedly during
its active running state — processing data, sending messages, receiving frames,
notifying upper layers, filtering traffic. This is the module's regular ongoing
job after it has started and before it is stopped.

SELF-TEST: Ask "Is this the module's ongoing behaviour while communication is
actively running — something it does repeatedly during operation?"
If YES → Normal Operation.

Correct examples:
- The CAN Driver shall provide a service to enable and disable interrupts
- The driver shall notify the CAN Interface about successful reception
- The driver shall provide a dynamic transmission request service
- The module shall forward L-PDUs to upper layers
- The driver shall guarantee data consistency of received L-PDUs

DO NOT classify as Normal Operation:
- "Response time shall be under 5ms"
  → Quality requirement (PerformanceEfficiency), not a functional subtype
- "Baud rate shall be configurable"
  → Configuration
- "The driver shall notify the CAN Interface if the controller enters bus-off state"
  → Fault Handling (triggered by error, not regular flow)

---

### SHUTDOWN
PURPOSE:
The requirement describes what the module must do when being stopped, disabled,
or de-initialized in a controlled and intentional way. This is the reverse of
Initialization — the module tears down its state cleanly.

SELF-TEST: Ask "Is this describing controlled teardown or intentional stopping
of the module by design, not because of an error?"
If YES → Shutdown.

Correct examples:
- The CAN Driver shall implement an interface for de-initialization
- The module shall put itself in a state that accepts a subsequent initialization call
- The transceiver shall be set to standby then sleep during system shutdown
- The CAN Stack shall stop communication before shutdown

DO NOT classify as Shutdown:
- "The module shall recover from bus-off"
  → Fault Handling (error driven, not intentional shutdown)
- "The module shall be re-initialized"
  → If after an error → Fault Handling
  → If part of a planned reconfiguration → Initialization

---

### FAULT HANDLING
PURPOSE:
The requirement describes what the module must do when something goes WRONG —
an error condition, failure, or abnormal state occurs. The trigger is always an
unplanned problem: a hardware fault, communication error, or detected failure.
Focus is on detection, reporting, or recovery from problems.

SELF-TEST: Ask "Is this behaviour triggered by an error, fault, or abnormal
condition rather than by normal system flow?"
If YES → Fault Handling.

Correct examples:
- The CAN driver shall notify the CAN Interface if the controller enters bus-off state
- The CAN driver shall not recover from bus-off automatically
- The State Manager shall control the BusOff recovery algorithm
- The bus transceiver driver shall check the transceiver for correctness and report faults
- The CAN-TP shall abort the segmentation session on unexpected N-PDU

DO NOT classify as Fault Handling:
- "The module shall be initialized"
  → Initialization (planned startup)
- "The module shall de-initialize"
  → Shutdown (planned teardown)
- "The module shall handle wakeup during sleep transition"
  → Normal Operation or Initialization depending on context
"""


def classify_stage1_type(
        requirement: str,
        iso_top_k: int = 3,
        ex_top_k: int = 3,
        table_name: str = "requirement_examples",
        custom_classes: list = None,
        dataset_name: str = None,
) -> dict:

    if not custom_classes:
        custom_classes = get_dataset_valid_classes(dataset_name, table_name) or None

    # add OTHER so gemini is not forced to pick a wrong class
    if custom_classes and "OTHER" not in [c.upper() for c in custom_classes]:
        custom_classes = list(custom_classes) + ["OTHER"]

    qvec = embed_text(requirement)
    exact_thresh, similar_thresh = get_thresholds(table_name)

    ex = retrieve_example_context_by_vec(
        qvec,
        top_k=ex_top_k,
        query_text=requirement,
        table_name=table_name
    )

    print("=== Examples Retrieved ===")
    print(ex)

    rag_label, rag_score = _extract_top_rag_label(ex)
    rag_result = _rag_label_to_result(rag_label, table_name=table_name)

    print("=== RAG Tier Check ===")
    print(f"RAG LABEL : {rag_label}")
    print(f"RAG SCORE : {rag_score}")
    print(f"RAG RESULT: {rag_result}")

    # if RAG confidence is high enough, skip Gemini entirely
    if rag_result is not None and rag_score is not None and rag_score >= exact_thresh:
        main_type, subtype = rag_result

        result = {
            "type": main_type,
            "confidence": "HIGH",
            "ambiguous": False,
            "note": f"Strong match in knowledge base (score={rag_score}).",
            "raw": f"RAG override: {rag_label} score={rag_score}",
            "iso_context": "",
            "ex_context": ex,
            "rag_label": rag_label,
            "rag_score": rag_score,
            "source": "RAG (exact match)",
            "_qvec": qvec,
        }

        if subtype is not None:
            result["subtype"] = subtype

        return result

    # no exact match -> gemini classifies, but only with the iso definitions + examples from rag

    iso = retrieve_iso_context_by_vec(
        qvec,
        "iso29148",
        top_k=iso_top_k,
        query_text=requirement
    )

    print("=== ISO 29148 Context (Stage 1) ===")
    print(iso)

    rag_examples_block = _build_rag_examples_block(ex)
    custom_classes_block = _build_custom_classes_block(custom_classes)

    prompt = f"""
You are a requirements classification assistant.

{custom_classes_block}

Classify the given requirement.

General taxonomy:
- FUNCTIONAL: the requirement describes what the system, module, component, or service must do.
- QUALITY: the requirement describes how well the system performs or behaves.
- CONSTRAINT: the requirement describes a restriction on design, platform, environment, appearance, or implementation.
- REGULATORY: the requirement describes compliance with laws, standards, rules, or certifications.

Important rule:
Do not classify only by matching single words.
The same technical word may appear in different contexts.
Classify based on the main purpose of the requirement.

Verified labelled examples:
{rag_examples_block}

Reference context:
{iso}

Requirement:
"{requirement}"

How to decide:
1. Use ONLY the ISO 29148 definitions in the reference context to find the TYPE of the requirement.
   Do not use your own knowledge. Compare the requirement with each definition and its characteristics and examples.
2. Then choose the class that has the same TYPE.
   If user-defined classes are shown and none of them has this TYPE, the answer is OTHER.
3. Use the labelled examples only when they are really similar to the requirement.

Output format:
DEFINITION USED: <which ISO definition fits and why, one sentence>
TYPE: <FUNCTIONAL|QUALITY|CONSTRAINT|REGULATORY|INTERFACE|BUSINESS>
FINAL ANSWER: <label>
CONFIDENCE: <HIGH|MEDIUM|LOW>
AMBIGUOUS: <YES|NO>
AMBIGUITY NOTE: <short reason if YES, else None>
"""

    raw = _call_gemini(prompt)
    print("=== Gemini Answer (Stage 1) ===")
    print(raw)
    label = _extract_label(raw, VALID_TYPES, custom_classes=custom_classes)

    # check: the class must be of the same type gemini found, if not -> OTHER
    type_match = re.search(r"TYPE\s*:\s*([A-Z]+)", raw, re.IGNORECASE)
    found_type = type_match.group(1).upper() if type_match else None
    label_type = _class_type(label)
    if (custom_classes and "OTHER" in custom_classes and found_type and label_type
            and label_type != found_type):
        print(f"=== type mismatch: gemini TYPE={found_type} but class {label} is {label_type} -> OTHER ===")
        label = "OTHER"

    if custom_classes and label in custom_classes:
        return {
            "type": label,
            "confidence": "HIGH",
            "ambiguous": False,
            "note": f"Classified into custom class: {label}",
            "raw": raw,
            "iso_context": iso,
            "ex_context": ex,
            "source": "Gemini (with RAG context)",
            "_qvec": qvec,
        }

    confidence = _extract_confidence(raw)
    is_ambiguous, note = _extract_ambiguity(raw)

    return {
        "type": label,
        "confidence": confidence,
        "ambiguous": is_ambiguous,
        "note": note,
        "raw": raw,
        "iso_context": iso,
        "ex_context": ex,
        "source": "Gemini (with RAG context)",
        "_qvec": qvec,
    }


def classify_stage2_quality(
        requirement: str,
        iso_top_k: int = 3,
        ex_top_k: int = 3,
        main_type: str = "QUALITY",
        table_name: str = "requirement_examples",
        custom_classes: list = None,
        dataset_name: str = None,
        precomputed_vec=None,
) -> dict:

    if not custom_classes:
        custom_classes = get_dataset_valid_classes(dataset_name, table_name) or None

    exact_thresh, similar_thresh = get_thresholds(table_name)
    qvec = precomputed_vec if precomputed_vec is not None else embed_text(requirement)

    # functional subtype path
    if main_type == "FUNCTIONAL":

        run_functional_subtypes = should_classify_functional_subtypes(
            dataset_name=dataset_name,
            table_name=table_name
        )

        if not run_functional_subtypes:
            print(f"=== Functional subtypes DISABLED for dataset: {dataset_name or table_name} ===")
            return {
                "subtype": None,
                "confidence": "HIGH",
                "ambiguous": False,
                "note": "Functional subtype classification not enabled for this dataset. Plain FR kept.",
                "raw": "Dataset config: functional_subtypes_enabled=False"
            }

        ex = retrieve_example_context_by_vec(
            qvec,
            top_k=ex_top_k,
            query_text=requirement,
            table_name=table_name
        )

        rag_label, rag_score = _extract_top_rag_label(ex)
        rag_result = _rag_label_to_result(rag_label, table_name=table_name)

        print("=== Stage 2 FUNCTIONAL RAG Check ===")
        print(f"RAG LABEL={rag_label} SCORE={rag_score} RESULT={rag_result}")

        if rag_result is not None and rag_score is not None and rag_score >= exact_thresh:
            mapped_type, mapped_subtype = rag_result
            if mapped_type == "FUNCTIONAL" and mapped_subtype is not None:
                return {
                    "subtype": mapped_subtype,
                    "confidence": "HIGH",
                    "ambiguous": False,
                    "note": f"Strong match in knowledge base (score={rag_score}).",
                    "raw": f"RAG override: {rag_label} score={rag_score}"
                }

        iso = retrieve_iso_context_by_vec(
            qvec,
            "iso29148",
            top_k=iso_top_k,
            query_text=requirement
        )

        print("=== ISO 29148 Context (Stage 2 FUNCTIONAL) ===")
        print(iso)

        rag_examples_block = _build_rag_examples_block(ex)

        prompt = f"""
You are a requirements classification assistant for automotive embedded software.

The requirement has already been classified as FUNCTIONAL.
Now assign exactly one lifecycle phase subtype.

{_FUNCTIONAL_SUBTYPE_DEFINITIONS}

---

## YOUR TASK

Step 1: Read the requirement carefully.
Step 2: Identify the MAIN PURPOSE — what is this requirement trying to achieve?
Step 3: Apply the SELF-TEST question for each candidate subtype.
Step 4: Pick ONE subtype based on the purpose, not on single keywords.

Verified labelled examples from knowledge base (strongest signal — use first):
{rag_examples_block}

Additional reference context:
{iso}

Requirement to classify:
"{requirement}"

---

Output format (all five lines required, in this exact order):
REASONING: <one sentence about the main purpose of this requirement>
FINAL ANSWER: <Configuration | Initialization | Normal Operation | Shutdown | Fault Handling>
CONFIDENCE: <HIGH | MEDIUM | LOW>
AMBIGUOUS: <YES | NO>
AMBIGUITY NOTE: <short reason if YES, else None>
"""

        valid_functional_set = {
            "Configuration",
            "Initialization",
            "Normal Operation",
            "Shutdown",
            "Fault Handling"
        }

        raw = _call_gemini(prompt)
        label = _extract_label(raw, valid_functional_set, custom_classes=None)
        confidence = _extract_confidence(raw)
        is_ambiguous, note = _extract_ambiguity(raw)

        return {
            "subtype": label,
            "confidence": confidence,
            "ambiguous": is_ambiguous,
            "note": note,
            "raw": raw,
        }

    # regulatory always maps to Legal
    if main_type == "REGULATORY":
        return {
            "subtype": "Legal",
            "confidence": "HIGH",
            "ambiguous": False,
            "note": "",
            "raw": "REGULATORY → Legal"
        }

    # CONSTRAINT: no subtype classification
    if main_type == "CONSTRAINT":
        return {
            "subtype": None,
            "confidence": "HIGH",
            "ambiguous": False,
            "note": "Constraint requirements have no subtype.",
            "raw": "CONSTRAINT → no subtype"
        }

    # quality subtype path
    ex = retrieve_example_context_by_vec(
        qvec,
        top_k=ex_top_k,
        query_text=requirement,
        table_name=table_name
    )

    rag_label, rag_score = _extract_top_rag_label(ex)
    rag_result = _rag_label_to_result(rag_label, table_name=table_name)

    print("=== Stage 2 QUALITY RAG Check ===")
    print(f"RAG LABEL={rag_label} SCORE={rag_score} RESULT={rag_result}")

    # for the default mixed table, skip RAG subtype override to avoid cross-domain noise
    # domain-specific tables still use RAG override as normal
    if table_name != "requirement_examples":
        if rag_result is not None and rag_score is not None and rag_score >= exact_thresh:
            mapped_type, mapped_subtype = rag_result
            if mapped_type == "QUALITY" and mapped_subtype is not None:
                return {
                    "subtype": mapped_subtype,
                    "confidence": "HIGH",
                    "ambiguous": False,
                    "note": f"Strong match in knowledge base (score={rag_score}).",
                    "raw": f"RAG override: {rag_label} score={rag_score}"
                }

    iso = retrieve_iso_context_by_vec(
        qvec,
        "iso25010",
        top_k=iso_top_k,
        query_text=requirement
    )

    print("=== ISO 25010 Context (Stage 2 QUALITY) ===")
    print(iso)

    rag_examples_block = _build_rag_examples_block(ex)
    custom_classes_block = _build_custom_classes_block(custom_classes)

    prompt = f"""
You are a requirements classification assistant.

{custom_classes_block}

The requirement is already classified as QUALITY.
Classify it into the correct quality subtype.

Quality subtype meanings:

Security:
The requirement describes protection of data, users, access, authentication, authorization, privacy, or encryption.

PerformanceEfficiency:
The requirement describes time, speed, response time, throughput, latency, or resource efficiency.

Reliability:
The requirement describes stable service, availability, uptime, dependable operation, or continuous operation.

FaultTolerance:
The requirement describes recovery, backup, failover, resilience, or behaviour after failure.

Scalability:
The requirement describes handling more users, more data, more load, or growing system capacity.

Usability:
The requirement describes ease of use, learnability, user understanding, clarity, or user interaction quality.

Maintainability:
The requirement describes ease of modification, testing, reuse, modularity, or future changes.

Compatibility:
The requirement describes coexistence, interoperability, or integration with another system.

Portability:
The requirement describes use across platforms, operating systems, devices, or environments.

Look_and_Feel:
The requirement describes visual appearance, UI layout, styling, branding, or presentation of the system.

Operational:
The requirement describes runtime operating conditions, deployment environment, operating modes, or environmental constraints on quality.

FunctionalSuitability:
Use only when the requirement is quality-related but does not clearly fit the other quality subtypes.

Important rule:
Do not classify only by matching single words.
Classify based on the main purpose of the requirement.

Reference examples (use as supporting context only):
{rag_examples_block}

ISO 25010 reference context:
{iso}

Requirement:
"{requirement}"

Output format:
FINAL ANSWER: <Security | PerformanceEfficiency | Reliability | FaultTolerance | Usability | Scalability | Maintainability | Compatibility | Portability | Look_and_Feel | Operational | FunctionalSuitability>
CONFIDENCE: <HIGH|MEDIUM|LOW>
AMBIGUOUS: <YES|NO>
AMBIGUITY NOTE: <short reason if YES, else None>
"""

    valid_quality_set = {
        "Security", "PerformanceEfficiency", "Reliability",
        "FaultTolerance", "Usability", "Scalability",
        "Maintainability", "Compatibility", "Portability",
        "FunctionalSuitability", "Look_and_Feel", "Operational"
    }

    raw = _call_gemini(prompt)
    label = _extract_label(raw, valid_quality_set, custom_classes=custom_classes)
    confidence = _extract_confidence(raw)
    is_ambiguous, note = _extract_ambiguity(raw)

    return {
        "subtype": label,
        "confidence": confidence,
        "ambiguous": is_ambiguous,
        "note": note,
        "raw": raw,
    }


def classify_batch_single_shot(
        requirements: list,
        iso_top_k: int = 2,
        ex_top_k: int = 2,
        table_name: str = "requirement_examples",
        dataset_name: str = "Default (All)",
        custom_classes: list = None,
):
    results = []

    for i, requirement in enumerate(requirements, start=1):
        stage1 = classify_stage1_type(
            requirement,
            iso_top_k=iso_top_k,
            ex_top_k=ex_top_k,
            table_name=table_name,
            custom_classes=custom_classes,
            dataset_name=dataset_name
        )

        main_type   = stage1.get("type", "Unknown")
        subtype     = stage1.get("subtype")
        stage1_qvec = stage1.get("_qvec")

        run_stage2 = stage2_for_dataset(
            main_type=main_type,
            dataset_name=dataset_name,
            table_name=table_name,
        )

        if subtype is None and run_stage2:
            stage2 = classify_stage2_quality(
                requirement,
                iso_top_k=iso_top_k,
                ex_top_k=ex_top_k,
                main_type=main_type,
                table_name=table_name,
                custom_classes=custom_classes,
                dataset_name=dataset_name,
                precomputed_vec=stage1_qvec,
            )
            subtype = stage2.get("subtype")

        item = {
            "id": i,
            "requirement": requirement,
            "type": main_type
        }

        if subtype is not None and run_stage2:
            item["subtype"] = subtype

        results.append(item)

    return results