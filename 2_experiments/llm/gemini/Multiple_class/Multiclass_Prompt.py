# prompts for multi class (promise 12 class, user story 8 class)

promise_labels = ["FR", "A", "FT", "L", "O", "SE", "US", "PE", "MN", "PO", "SC", "OP"]
userstory_labels = ["CM", "FR", "MN", "PE", "PO", "RL", "SE", "US"]


# user story 8 class

userstory_zero = """
Classify each software requirement into exactly ONE of these labels:

CM, FR, MN, PE, PO, RL, SE, US

- Output CSV ONLY in exactly this format:
  <Requirement>:<Type>

Requirements:
{lines}
"""

userstory_iso = """
Classify software requirements using ISO/IEC/IEEE 29148 and ISO/IEC 25010 guidance.

- FR: specifies system behavior/function/service.
- Otherwise choose the best fitting quality/constraint class guided by ISO 25010:
  - Performance efficiency -> PE
  - Security -> SE
  - Usability -> US
  - Reliability -> RL
  - Maintainability -> MN
  - Portability -> PO
  - Compatibility -> CM

Output must be exactly ONE of:
CM, FR, MN, PE, PO, RL, SE, US

- Output CSV ONLY in exactly this format:
  <Requirement>:<Type>

Requirements:
{lines}
"""

userstory_examples = """
Examples:

Requirement: The system shall allow users to create, edit, and delete tasks.
Answer: FR

Requirement: The system shall respond to search queries within 2 seconds under normal load.
Answer: PE

Requirement: The system shall encrypt stored passwords and require MFA for admin accounts.
Answer: SE

Requirement: The interface shall be easy to learn for first-time users with clear guidance.
Answer: US

Requirement: The system shall run on Windows and Linux without modification.
Answer: PO

Requirement: The system shall recover automatically after a crash and not lose committed data.
Answer: RL

Requirement: The code shall be modular and changes shall require minimal modifications.
Answer: MN

Requirement: The system shall integrate with external identity providers using standard protocols.
Answer: CM
"""

userstory_few = """
Classify each software requirement into exactly ONE label:

CM, FR, MN, PE, PO, RL, SE, US

- Output CSV ONLY in exactly this format:
  <Requirement>:<Type>

{examples}

Requirements:
{lines}
"""

userstory_cot = """
You are an expert in software requirements classification.

Classify each requirement into EXACTLY ONE of these 8 labels:

FR  → Functional Requirement (system behavior or service)
PE  → Performance (response time, throughput, resource usage, scalability)
US  → Usability (UI clarity, accessibility, learnability)
SE  → Security (authentication, authorization, confidentiality, integrity)
MN  → Maintainability (modifiability, testability, logging, analyzability)
PO  → Portability (platform independence, installability, migration)
RL  → Reliability (stability, failure handling, robustness)
CM  → Compatibility (interoperability, integration, standards compliance)

Internal reasoning steps (DO NOT output reasoning):
1. Determine whether the requirement describes system behavior (FR) 
   or a quality attribute (NFR).
2. If NFR, identify the dominant quality concern.
3. If multiple qualities appear, choose the most emphasized one.
4. Choose the closest semantic match from the 8 labels.

Output rules (STRICT):
- Output CSV ONLY in exactly this format:
  <Requirement>:<Type>
- No explanations, No extra text, No markdown, No bullet points

Requirements:
{lines}
"""


# promise 12 class
# TODO: OP here but promise has LF, check

promise_zero = """
Classify each software requirement into exactly ONE of the following labels:

FR, A, FT, L, O, SE, US, PE, MN, PO, SC, OP

- Output CSV ONLY in exactly this format:
  <Requirement>:<Type>

Requirements:
{lines}
"""

promise_iso = """
You are classifying software requirements using ISO/IEC/IEEE 29148 and ISO/IEC 25010.

Step 1 (ISO 29148 framing):
- If the statement specifies a system function/behavior/service → choose FR.
- Otherwise it is a quality/constraint requirement → choose the best fitting non-functional label.

Step 2 (ISO 25010 quality model guidance):
Use ISO 25010 quality characteristics to guide selection for NFR-like requirements:
- Performance Efficiency → often PE
- Security → SE
- Usability → US
- Reliability → often A (availability) or related label if your dataset uses A for availability
- Maintainability → MN
- Portability → PO
- Compatibility → often OP/SC depending on dataset usage

IMPORTANT:
Output must be exactly ONE of these dataset labels:
FR, A, FT, L, O, SE, US, PE, MN, PO, SC, OP

- Output CSV ONLY in exactly this format:
  <Requirement>:<Type>

Requirements:
{lines}
"""

promise_examples = """
Examples:

Requirement: The system shall allow users to upload files.
Answer: FR

Requirement: The system shall integrate with third-party payment APIs.
Answer: A

Requirement: The system shall allow administrators to configure business rules without redeployment.
Answer: FT

Requirement: The system shall provide an online help manual for all features.
Answer: L

Requirement: The system shall manage user sessions and terminate inactive sessions after 15 minutes.
Answer: O

Requirement: The system shall authenticate users and encrypt stored passwords.
Answer: SE

Requirement: The interface shall be easy to learn for first-time users.
Answer: US

Requirement: The system shall respond to search requests within 2 seconds.
Answer: PE

Requirement: The source code shall be modular and easy to maintain.
Answer: MN

Requirement: The system shall run on Windows, Linux, and macOS.
Answer: PO

Requirement: The system shall record all user actions in an audit log.
Answer: SC

Requirement: The system shall support import and export in CSV and JSON format.
Answer: OP
"""

promise_few = """
Classify each software requirement into exactly ONE label:

FR, A, FT, L, O, SE, US, PE, MN, PO, SC, OP

- Output CSV ONLY in exactly this format:
  <Requirement>:<Type>

{examples}

Requirements:
{lines}
"""

promise_cot = """
You are classifying software requirements using ISO/IEC/IEEE 29148 and ISO/IEC 25010.

Goal:
Assign exactly ONE label per requirement from this set:
FR, A, FT, L, O, SE, US, PE, MN, PO, SC, OP

Internal reasoning protocol (DO NOT reveal your reasoning):
1) Decide if the requirement is primarily:
   - Functional (what the system shall do) -> FR
   - Non-functional (quality attribute, constraint, condition, or limitation) -> choose the best fitting NFR label
2) If the requirement mixes function + constraint:
   - If the main intent is a function and constraint is secondary -> FR
   - If the main intent is a quality/constraint (time, security, compliance, availability, resource limits, etc.) -> NFR label
3) Choose the closest dataset label using ISO 25010 concepts as guidance:
   - PE: performance efficiency (latency, throughput, response time, resource usage, capacity)
   - SE: security (confidentiality, integrity, authentication, authorization, auditing, encryption)
   - US: usability (UI/UX, learnability, accessibility, error prevention, help/documentation for users)
   - MN: maintainability (modularity, analyzability, modifiability, testability, logging for debugging/maintenance)
   - PO: portability (installability, adaptability, platform/OS migration)
   - OP / SC: interoperability/compatibility style constraints (integration, interfaces, standards, co-existence)
   - A: availability / uptime / service continuity (if A is used for availability in this dataset)
   - FT, L, O: use ONLY if the requirement clearly matches these dataset categories by intent
     (If unsure between multiple labels, select the most dominant/explicitly stated intent.)

Output rules (STRICT):
- Output CSV ONLY in exactly this format:
  <Requirement>:<Type>
- No explanations, no extra text, no headings, no bullet points, no code fences.
- Labels must be exactly one of: FR, A, FT, L, O, SE, US, PE, MN, PO, SC, OP

Requirements:
{lines}
"""