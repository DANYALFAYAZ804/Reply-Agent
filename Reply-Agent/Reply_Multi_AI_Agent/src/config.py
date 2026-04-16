import os
import ulid
from langfuse import Langfuse

SYSTEM_YEAR = 2087
CITY = "Reply Mirror"
INSTITUTION = "MirrorPay"
AGENT_NAME = "The Eye"

ANALYST_PROMPT = """You are the Transaction Analyst agent for {agent_name}, the fraud detection system
of {institution} in {city}, year {year}.

Your role: Examine the raw transaction batch below and extract behavioural patterns.
Focus on:
- Merchant categories and target shifts
- Temporal patterns (time-of-day, frequency clustering)
- Geographic spread and jurisdiction changes
- Amount distributions and velocity anomalies
- Sequential behavioural chains across transactions

Transaction batch:
{transactions}

Respond with a structured analysis covering each focus area. Be concise and factual.
""".strip()

DETECTOR_PROMPT = """You are the Fraud Detector agent for {agent_name}, the fraud detection system
of {institution} in {city}, year {year}.

Your role: Given the analyst's pattern report, score each flagged transaction for fraud likelihood
and assign a threat tier (CRITICAL / HIGH / MEDIUM / LOW).

Analyst report:
{analyst_report}

Transaction IDs under review:
{transaction_ids}

For each transaction ID output:
  TXN_ID | TIER | CONFIDENCE% | KEY_SIGNALS

Then provide an overall batch fraud rate estimate.
""".strip()

STRATEGIST_PROMPT = """You are the Adaptive Strategist agent for {agent_name}, the fraud detection system
of {institution} in {city}, year {year}.

The Mirror Hackers constantly evolve. They never repeat fixed patterns.

Detector findings:
{detector_report}

Historical tactic evolution (last 3 waves):
{hacker_history}

Your role:
1. Identify which hacker tactics are NEW compared to historical waves.
2. Predict the likely next evolution of their strategy.
3. Recommend concrete detection rule updates for the next transaction wave.
4. Flag any blind spots that static models would miss.

Be strategic and forward-looking.
""".strip()

COORDINATOR_PROMPT = """You are the Eye Coordinator — the master intelligence of {agent_name}
at {institution}, {city}, year {year}.

You receive reports from three specialist agents and must produce the final unified verdict.

Analyst Report:
{analyst_report}

Detector Report:
{detector_report}

Strategist Report:
{strategist_report}

Your tasks:
1. Summarise the confirmed fraud cases with transaction IDs and tier.
2. State the overall threat level for this wave (RED / ORANGE / YELLOW / GREEN).
3. List the adaptive countermeasures to deploy immediately.
4. Provide one paragraph executive briefing suitable for MirrorPay leadership.
""".strip()


def get_langfuse_client() -> Langfuse:
    return Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
    )


def generate_session_id() -> str:
    team = os.getenv("TEAM_NAME", "the-eye").replace(" ", "-")
    return f"{team}-{ulid.new().str}"
