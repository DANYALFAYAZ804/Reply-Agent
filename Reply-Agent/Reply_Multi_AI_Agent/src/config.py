import os
import ulid
from langfuse import Langfuse

SYSTEM_YEAR = 2087
CITY = "Reply Mirror"
INSTITUTION = "MirrorPay"
AGENT_NAME = "The Eye"
TOTAL_LEVELS = 5

ANALYST_PROMPT = """You are the Transaction Analyst for {agent_name} ({institution}, {city}, year {year}).
LEVEL {level} — scan this batch and extract CONCRETE fraud signals only.

=== TRANSACTIONS ===
{transactions}

=== LOCATIONS ===
{locations}

=== USERS ===
{users}

=== SMS ===
{conversations}

=== EMAIL ===
{messages}

Output ONLY a compact list of fraud signals. For every suspicious transaction ID found, output:
  FLAG: <txn_id> | SIGNALS: <comma-separated anomalies>

Anomaly codes to use (pick all that apply):
  AMT_HIGH, AMT_LOW, BAL_IMPOSSIBLE, TS_FUTURE, TS_ODD_HOUR, BURST_SENDER,
  IBAN_MISMATCH, METHOD_MISMATCH, GPS_IMPOSSIBLE, IDENTITY_SPOOF, PHISHING_SMS,
  PHISHING_MAIL, VELOCITY_HIGH, UNKNOWN_RECIPIENT, SAME_SENDER_MULTI

Do NOT output explanations. Do NOT list legitimate transactions. Only FLAG lines.
""".strip()

DETECTOR_PROMPT = """You are the Fraud Detector for {agent_name} ({institution}, {city}, year {year}).
LEVEL {level}

Analyst flags (suspicious signals per txn_id):
{analyst_report}

=== TRANSACTIONS TO SCORE ===
{transactions}

DECISION RULES (apply strictly):
- FRAUD if 2+ anomaly signals match the analyst flags for that txn_id.
- FRAUD if amount is impossible given balance_after (balance_after < 0 or balance_after > pre-balance).
- FRAUD if timestamp is in the future or sender appears 5+ times in this batch.
- LEGITIMATE if 0-1 weak signals only.

Output ONLY fraud decisions. For each fraud:
  FRAUD: <txn_id>

No commentary. No legitimate listings. No explanations. Only FRAUD lines.
""".strip()

STRATEGIST_PROMPT = """You are the Adaptive Strategist for {agent_name} ({institution}, {city}, year {year}).
LEVEL {level}

Detector FRAUD list:
{detector_report}

Prior hacker tactics:
{hacker_history}

Tasks:
1. List any txn IDs the Detector likely MISSED (pattern-based).
2. Identify new tactics not in prior levels.

Output format — missed IDs only:
  MISSED: <txn_id>

No commentary. Only MISSED lines (or nothing if none).
""".strip()

COORDINATOR_PROMPT = """You are the Eye Coordinator for {agent_name} ({institution}, {city}, year {year}).
LEVEL {level}

Detector FRAUD lines:
{detector_report}

Strategist MISSED lines:
{strategist_report}

Available Transaction IDs (this batch):
{all_txn_ids}

RULES:
- Combine all FRAUD and MISSED IDs from the reports above.
- Only include IDs that appear in the Available Transaction IDs list.
- Output NOTHING except the markers and IDs below.
- Do NOT include every ID. Do NOT output zero IDs.

===FRAUD_LIST===
<one transaction_id per line — fraud and missed combined>
===END_LIST===
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
