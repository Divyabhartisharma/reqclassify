# TASKS:
#   userstory_nfr   -> 7 NFR subtypes      (User-story/1600 samples)
#   promise_nfr     -> 11 NFR subtypes     (PROMISE/969 samples)

# =========================================================
# TASK SETTINGS
# input path
# output folder
# =========================================================
TASKS = {
    "userstory_nfr": {
        "input": "10006_dataset_Multi_NFRclass.csv",
        "output_folder": "NFR_Subtype",
        "labels": ["CM", "MN", "PE", "PO", "RL", "SE", "US"],
        "batch_size": {"zero": 25, "iso": 20, "few": 20, "cot": 20},
        "delay": {"zero": 0, "iso": 0, "few": 0, "cot": 0},
    },
    "promise_nfr": {
        "input": "PROMISE_exp_NFRclass.csv",
        "output_folder": "NFR_Subtype",
        "labels": ["A", "FT", "L", "O", "SE", "US", "PE", "MN", "PO", "SC", "OP"],
        "batch_size": {"zero": 25, "iso": 20, "few": 20, "cot": 20},
        "delay": {"zero": 0, "iso": 0, "few": 0, "cot": 0},
    },
}

# =========================================================
# USER STORY  -  7 NFR subtypes
# =========================================================
def userstory_zero(lines):
    return f"""
Each of the following is a NON-FUNCTIONAL requirement.

Classify each into exactly ONE of these labels:
CM, MN, PE, PO, RL, SE, US

Return ONLY lines in this format:
<Requirement>:<Type>

Requirements:
{lines}
""".strip()


def userstory_iso(lines):
    return f"""
Each Requirement text below is a NON-FUNCTIONAL requirement.

Classify using ISO/IEC 25010 quality characteristics guidance:

- Performance efficiency -> PE
- Security -> SE
- Usability -> US
- Reliability -> RL
- Maintainability -> MN
- Portability -> PO
- Compatibility -> CM

Output must be exactly ONE of:
CM, MN, PE, PO, RL, SE, US

Return ONLY csv in this format:
<Requirement>:<Type>

Requirements:
{lines}
""".strip()


def userstory_few(lines):
    examples = """
Examples:

Requirement: The system shall respond to each request within 2 seconds under normal load.
Answer: PE

Requirement: Passwords shall be encrypted and access shall require multi-factor authentication.
Answer: SE

Requirement: The interface shall be intuitive and easy to learn for first-time users.
Answer: US

Requirement: The system shall recover automatically after a crash without losing committed data.
Answer: RL

Requirement: The code shall be modular and changes shall require minimal modifications.
Answer: MN

Requirement: The application shall run on Windows and Linux without modification.
Answer: PO

Requirement: The system shall integrate with external services using standard APIs/protocols.
Answer: CM
"""
    return f"""
Each of the following is a NON-FUNCTIONAL requirement.

Classify each into exactly ONE of these labels:
CM, MN, PE, PO, RL, SE, US

Return ONLY csv in this format:
<Requirement>:<Type>

{examples}

Requirements:
{lines}
""".strip()


def userstory_cot(lines):
    return f"""
All statements below are NON-FUNCTIONAL requirements.

Classify each requirement into EXACTLY ONE of the following ISO/IEC 25010 quality subtypes:

PE  → Performance efficiency (response time, throughput, latency, scalability, resource usage)
SE  → Security (authentication, authorization, confidentiality, integrity, encryption, audit)
US  → Usability (ease of use, accessibility, UI clarity, user guidance)
RL  → Reliability (fault tolerance, recovery, robustness, stability)
MN  → Maintainability (modifiability, testability, logging, diagnosability)
PO  → Portability (platform independence, installability, migration)
CM  → Compatibility (interoperability, integration, standards compliance)

1. Identify the PRIMARY quality concern.
2. Distinguish carefully:
3. If multiple qualities appear, choose the dominant one.
4. Select the closest semantic match.

Output rules (STRICT):
Return ONLY csv in this format:
<Requirement>:<Type>
- No explanations, No extra text, No markdown, No bullet points
Requirements:
{lines}
""".strip()


# =========================================================
# PROMISE  -  11 NFR subtypes
# =========================================================
def promise_zero(lines):
    return f"""
Each of the following is a NON-FUNCTIONAL requirement.

Classify each into exactly ONE of these categories:

A, FT, L, O, SE, US, PE, MN, PO, SC, OP

Return ONLY csv in this format:
<Requirement>:<Type>
Requirements:
{lines}
""".strip()


def promise_iso(lines):
    return f"""
Each statement below is a NON-FUNCTIONAL requirement.

Classify using ISO/IEC 25010 quality characteristics guidance (and ISO/IEC/IEEE 29148 style of quality requirements).

Output must be exactly ONE of these dataset labels:
A, FT, L, O, SE, US, PE, MN, PO, SC, OP

Return ONLY csv in this format:
<Requirement>:<Type>

Requirements:
{lines}
""".strip()


def promise_few(lines):
    examples = """
Examples:

Requirement: The system shall respond to any request within 2 seconds.
Answer: PE

Requirement: The interface shall be intuitive for first-time users.
Answer: US

Requirement: The system shall encrypt all stored passwords.
Answer: SE

Requirement: The system shall run on Windows and Linux platforms.
Answer: PO

Requirement: The system shall provide a detailed online help manual.
Answer: L

Requirement: The system shall log all user activities for auditing.
Answer: SC

Requirement: The source code shall be modular and easy to modify.
Answer: MN

Requirement: The system shall allow configuration of business rules without redeployment.
Answer: FT

Requirement: The system shall integrate with external payment gateways.
Answer: A

Requirement: The system shall manage and terminate inactive user sessions.
Answer: O

Requirement: The system shall support CSV and JSON data export.
Answer: OP
"""
    return f"""
Each of the following is a NON-FUNCTIONAL requirement.

Classify each into exactly ONE of these labels:

A, FT, L, O, SE, US, PE, MN, PO, SC, OP

Return ONLY csv in this format:
<Requirement>:<Type>

{examples}

Requirements:
{lines}
""".strip()


def promise_cot(lines):
    return f"""
You are classifying software requirements into Non-Functional Requirement (NFR) subtypes
using ISO/IEC 25010 quality characteristics.

IMPORTANT:
All requirements provided are assumed to be NON-FUNCTIONAL.
Do NOT classify anything as FR.
Select the single best NFR subtype label.

Allowed labels (choose EXACTLY one):
A, FT, L, O, SE, US, PE, MN, PO, SC, OP

Internal reasoning protocol:
1) Identify the PRIMARY quality attribute or constraint expressed.
2) Ignore secondary wording and focus on dominant intent.
3) If multiple qualities appear, select the most explicitly stated or emphasized one.
4) Map to the closest ISO 25010-aligned concept:

Guidance:
- PE → performance efficiency (response time, throughput, latency, scalability, resource usage)
- SE → security (confidentiality, authentication, authorization, encryption, audit, integrity)
- US → usability (UI clarity, learnability, accessibility, error messages, help support)
- MN → maintainability (modifiability, analyzability, logging for debugging, testability)
- PO → portability (installability, OS/platform adaptation, migration)
- A → availability / uptime / service continuity
- OP / SC → interoperability / compatibility / integration constraints
- FT → fault tolerance / resilience / recovery
- L → legal, regulatory, or compliance constraints
- O → operational/environmental constraints (hardware limits, deployment conditions, physical constraints)

Output rules (STRICT):
-Return ONLY csv in this format:
<Requirement>:<Type>
- No explanations, No reasoning, No extra text, No bullet points
Requirements:
{lines}
""".strip()


# =========================================================
# =========================================================
PROMPTS = {
    "userstory_nfr": {"zero": userstory_zero, "iso": userstory_iso, "few": userstory_few, "cot": userstory_cot},
    "promise_nfr": {"zero": promise_zero, "iso": promise_iso, "few": promise_few, "cot": promise_cot},
}