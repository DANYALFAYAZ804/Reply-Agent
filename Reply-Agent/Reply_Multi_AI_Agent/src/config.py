import os
import ulid
from langfuse import Langfuse

SYSTEM_YEAR = 2087
CITY = "Reply Mirror"
INSTITUTION = "MirrorPay"
AGENT_NAME = "The Eye"
TOTAL_LEVELS = 5

ANALYST_PROMPT = """You are the Transaction Analyst agent for {agent_name}, the fraud detection system
of {institution} in {city}, year {year}.

LEVEL {level} — analyse the multi-source dataset below to extract fraud behaviour patterns.

=== TRANSACTIONS (sample) ===
{transactions}

=== CITIZEN LOCATIONS ===
{locations}

=== USER PROFILES ===
{users}

=== SMS CONVERSATIONS ===
{conversations}

=== EMAIL MESSAGES ===
{messages}

Your role: Identify ALL behavioural signals that distinguish fraudulent from legitimate activity.

Focus on:
1. Transaction anomalies: unusual amounts, impossible balance after transaction, impossible timestamps
2. Type/method mismatches: e.g. withdrawal via mobile device, bank transfer with no IBAN
3. Sender/recipient patterns: unknown IDs, suspicious IBAN country prefixes, same sender many recipients
4. Location anomalies: GPS position vs transaction location mismatch, impossible travel speed
5. Communication signals: keywords in SMS/email indicating phishing, urgency, account sharing
6. Temporal patterns: transactions at odd hours, burst activity, future-dated timestamps
7. Velocity: same sender ID appearing in many transactions in short window

Output a structured fraud pattern report. For each pattern found, give:
  PATTERN_NAME | DESCRIPTION | AFFECTED_SIGNALS
""".strip()

DETECTOR_PROMPT = """You are the Fraud Detector agent for {agent_name} at {institution}, {city}, year {year}.

LEVEL {level}

Analyst fraud pattern report:
{analyst_report}

=== ALL TRANSACTION IDs TO CLASSIFY ===
{transactions}

IMPORTANT RULES:
- Asymmetric cost: a false positive (blocking legitimate) costs economic and reputational damage.
- A false negative (missing fraud) causes financial damage.
- Do NOT flag all transactions — output will be INVALID if all are flagged.
- Do NOT flag zero transactions — output will be INVALID if none are flagged.
- You must correctly identify at least 15% of actual fraud or the submission is invalid.
- Prefer precision: only flag when confidence is HIGH.

For each transaction you suspect is fraudulent, output exactly one line:
  FRAUD: <transaction_id> | CONFIDENCE:<0-100> | REASON:<brief>

Only output lines for transactions you flag as fraudulent. Do not list legitimate ones.
""".strip()

STRATEGIST_PROMPT = """You are the Adaptive Strategist agent for {agent_name} at {institution}, {city}, year {year}.

LEVEL {level}

The Mirror Hackers constantly evolve their tactics. They blend with legitimate behaviour.

Detector findings (suspected fraud list):
{detector_report}

Historical tactic evolution from prior levels:
{hacker_history}

Your tasks:
1. Identify NEW tactics not seen in prior levels.
2. Predict the likely evolution for the next level.
3. Recommend detection rule updates for the next level.
4. Flag systemic blind spots the current agents may have missed.
5. List any transaction IDs the Detector may have MISSED based on patterns (add them with reason).

Output format for missed transactions:
  MISSED: <transaction_id> | REASON:<why it should be flagged>
""".strip()

COORDINATOR_PROMPT = """You are the Eye Coordinator — master intelligence of {agent_name}
at {institution}, {city}, year {year}.

LEVEL {level}

You receive inputs from three specialist agents. Produce the FINAL fraud verdict.

Analyst Report:
{analyst_report}

Detector Report:
{detector_report}

Strategist Report (may include MISSED transactions):
{strategist_report}

ALL available Transaction IDs:
{all_txn_ids}

CRITICAL OUTPUT RULES:
- Output ONLY the final list of suspected fraudulent Transaction IDs.
- One Transaction ID per line, no extra text, no commentary, no numbering.
- Do NOT include all transaction IDs (invalid submission).
- Do NOT output zero IDs (invalid submission).
- You must include at least 15% of actual fraud cases.
- Minimise false positives — each wrong block costs reputational damage.

Begin your response with the marker: ===FRAUD_LIST===
Then list one Transaction ID per line.
End with the marker: ===END_LIST===
""".strip()

HACKER_HISTORY: dict = {
    1: "Level 1: basic amount/timestamp anomalies; single attacker patterns.",
    2: "Level 2: cross-border transfers; IBAN country mismatches; late-night bursts.",
    3: "Level 3: GPS impossibility attacks; location/transaction mismatch; identity spoofing.",
    4: "Level 4: micro-transaction layering; communication-based phishing coordination; velocity attacks.",
}


def get_hacker_history_text(up_to_level: int) -> str:
    lines = []
    for lv in range(1, up_to_level):
        if lv in HACKER_HISTORY:
            lines.append(f"  - {HACKER_HISTORY[lv]}")
    return "\n".join(lines) if lines else "No prior level history available."


def get_langfuse_client() -> Langfuse:
    return Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
    )


def generate_session_id() -> str:
    team = os.getenv("TEAM_NAME", "the-eye").replace(" ", "-")
    return f"{team}-{ulid.new().str}"
