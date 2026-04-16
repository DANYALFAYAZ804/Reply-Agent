import re
import time
import logging
from dataclasses import dataclass
from typing import List, Set, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import observe
from langfuse.langchain import CallbackHandler

from src.config import (
    ANALYST_PROMPT,
    DETECTOR_PROMPT,
    STRATEGIST_PROMPT,
    COORDINATOR_PROMPT,
    SYSTEM_YEAR,
    CITY,
    INSTITUTION,
    AGENT_NAME,
    get_hacker_history_text,
)
from src.data import (
    LevelDataset,
    format_transactions_block,
    format_locations_block,
    format_users_block,
    format_conversations_block,
    format_messages_block,
    get_all_txn_ids,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 5


@dataclass
class LevelReport:
    level: int
    analyst_report: str
    detector_report: str
    strategist_report: str
    coordinator_report: str
    suspected_ids: List[str]


def _call_agent(model, prompt: str, session_id: str, agent_name: str = "agent") -> str:
    handler = CallbackHandler()
    messages = [HumanMessage(content=prompt)]
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = model.invoke(
                messages,
                config={
                    "callbacks": [handler],
                    "metadata": {"langfuse_session_id": session_id},
                },
            )
            return response.content
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "401" in err_str or "Authentication" in err_str or "Missing" in err_str:
                raise RuntimeError(
                    f"\n[AUTH ERROR] {agent_name} failed — your OPENROUTER_API_KEY is invalid or missing.\n"
                    f"Open Reply_Multi_AI_Agent/.env and set a real key from https://openrouter.ai\n"
                    f"Original error: {e}"
                ) from e
            if "context" in err_str.lower() or "token" in err_str.lower() or "length" in err_str.lower():
                raise RuntimeError(
                    f"\n[TOKEN LIMIT] {agent_name} prompt is too large. Reduce max_rows.\nOriginal: {e}"
                ) from e
            logger.warning(f"[{agent_name}] attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)

    raise RuntimeError(f"[{agent_name}] failed after {MAX_RETRIES} attempts. Last error: {last_error}")


def _parse_fraud_list(text: str, all_ids: Set[str]) -> List[str]:
    suspected = []

    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if "===FRAUD_LIST===" in stripped:
            in_block = True
            continue
        if "===END_LIST===" in stripped:
            in_block = False
            continue
        if in_block and stripped and stripped in all_ids:
            suspected.append(stripped)

    if not suspected:
        uuid_pattern = re.compile(
            r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
            re.IGNORECASE,
        )
        seen = set()
        for match in uuid_pattern.finditer(text):
            uid = match.group(1)
            if uid in all_ids and uid not in seen:
                suspected.append(uid)
                seen.add(uid)

    return suspected


def _chunk_transactions(dataset: LevelDataset, chunk_size: int = 60) -> List[str]:
    txns = dataset.transactions
    chunks = []
    for i in range(0, len(txns), chunk_size):
        chunks.append(format_transactions_block(txns[i:i + chunk_size], max_rows=chunk_size))
    return chunks


@observe()
def run_analyst(session_id: str, model, dataset: LevelDataset) -> str:
    txn_chunks = _chunk_transactions(dataset, chunk_size=60)
    partial_reports = []

    for idx, chunk_text in enumerate(txn_chunks, 1):
        chunk_label = f"(batch {idx}/{len(txn_chunks)})"
        loc_text = format_locations_block(dataset.locations, max_rows=20) if idx == 1 else "(see batch 1)"
        usr_text = format_users_block(dataset.users, max_rows=10) if idx == 1 else "(see batch 1)"
        conv_text = format_conversations_block(dataset.conversations, max_rows=8) if idx == 1 else "(see batch 1)"
        msg_text = format_messages_block(dataset.messages, max_rows=5) if idx == 1 else "(see batch 1)"

        prompt = ANALYST_PROMPT.format(
            agent_name=AGENT_NAME,
            institution=INSTITUTION,
            city=CITY,
            year=SYSTEM_YEAR,
            level=dataset.level,
            transactions=f"{chunk_label}\n{chunk_text}",
            locations=loc_text,
            users=usr_text,
            conversations=conv_text,
            messages=msg_text,
        )
        result = _call_agent(model, prompt, session_id, agent_name=f"Analyst-batch{idx}")
        partial_reports.append(f"--- Batch {idx} ---\n{result}")

    return "\n\n".join(partial_reports)


@observe()
def run_detector(session_id: str, model, analyst_report: str, dataset: LevelDataset) -> str:
    txn_chunks = _chunk_transactions(dataset, chunk_size=80)
    all_fraud_lines = []

    for idx, chunk_text in enumerate(txn_chunks, 1):
        analyst_summary = analyst_report[:2000] + ("..." if len(analyst_report) > 2000 else "")
        prompt = DETECTOR_PROMPT.format(
            agent_name=AGENT_NAME,
            institution=INSTITUTION,
            city=CITY,
            year=SYSTEM_YEAR,
            level=dataset.level,
            analyst_report=analyst_summary,
            transactions=f"(batch {idx}/{len(txn_chunks)})\n{chunk_text}",
        )
        result = _call_agent(model, prompt, session_id, agent_name=f"Detector-batch{idx}")
        all_fraud_lines.append(f"--- Batch {idx} ---\n{result}")

    return "\n\n".join(all_fraud_lines)


@observe()
def run_strategist(session_id: str, model, analyst_report: str, detector_report: str, dataset: LevelDataset) -> str:
    analyst_summary = analyst_report[:1500] + ("..." if len(analyst_report) > 1500 else "")
    detector_summary = detector_report[:2000] + ("..." if len(detector_report) > 2000 else "")

    prompt = STRATEGIST_PROMPT.format(
        agent_name=AGENT_NAME,
        institution=INSTITUTION,
        city=CITY,
        year=SYSTEM_YEAR,
        level=dataset.level,
        detector_report=detector_summary,
        hacker_history=get_hacker_history_text(dataset.level),
    )
    return _call_agent(model, prompt, session_id, agent_name="Strategist")


@observe()
def run_coordinator(
    session_id: str,
    model,
    analyst_report: str,
    detector_report: str,
    strategist_report: str,
    dataset: LevelDataset,
) -> str:
    all_ids = get_all_txn_ids(dataset.transactions)
    analyst_summary = analyst_report[:1200] + ("..." if len(analyst_report) > 1200 else "")
    detector_summary = detector_report[:2500] + ("..." if len(detector_report) > 2500 else "")
    strategist_summary = strategist_report[:1000] + ("..." if len(strategist_report) > 1000 else "")

    BATCH_SIZE = 200
    id_chunks = [all_ids[i:i + BATCH_SIZE] for i in range(0, len(all_ids), BATCH_SIZE)]
    all_suspected: List[str] = []
    all_ids_set = set(all_ids)

    for idx, id_chunk in enumerate(id_chunks, 1):
        prompt = COORDINATOR_PROMPT.format(
            agent_name=AGENT_NAME,
            institution=INSTITUTION,
            city=CITY,
            year=SYSTEM_YEAR,
            level=dataset.level,
            analyst_report=analyst_summary if idx == 1 else "(see batch 1)",
            detector_report=detector_summary if idx == 1 else "(see batch 1)",
            strategist_report=strategist_summary if idx == 1 else "(see batch 1)",
            all_txn_ids="\n".join(id_chunk),
        )
        result = _call_agent(model, prompt, session_id, agent_name=f"Coordinator-batch{idx}")
        batch_ids = _parse_fraud_list(result, all_ids_set)
        for bid in batch_ids:
            if bid not in all_suspected:
                all_suspected.append(bid)

    full_report = (
        f"Coordinator consolidated {len(all_suspected)} suspected fraud IDs "
        f"from {len(id_chunks)} batch(es).\n"
        f"===FRAUD_LIST===\n"
        + "\n".join(all_suspected)
        + "\n===END_LIST==="
    )
    return full_report


@observe()
def run_level_pipeline(
    session_id: str,
    analyst_model,
    detector_model,
    strategist_model,
    coordinator_model,
    dataset: LevelDataset,
) -> LevelReport:
    all_ids_set = set(get_all_txn_ids(dataset.transactions))
    total = len(all_ids_set)

    print(f"  [Analyst]     analysing {len(dataset.transactions)} transactions in batches...")
    analyst_report = run_analyst(session_id, analyst_model, dataset)

    print(f"  [Detector]    scoring transactions for fraud...")
    detector_report = run_detector(session_id, detector_model, analyst_report, dataset)

    print(f"  [Strategist]  adapting to new hacker patterns...")
    strategist_report = run_strategist(session_id, strategist_model, analyst_report, detector_report, dataset)

    print(f"  [Coordinator] consolidating final fraud verdict...")
    coordinator_report = run_coordinator(
        session_id, coordinator_model, analyst_report, detector_report, strategist_report, dataset
    )

    suspected_ids = _parse_fraud_list(coordinator_report, all_ids_set)

    if len(suspected_ids) == 0:
        logger.warning("Coordinator returned 0 fraud IDs — falling back to Detector output.")
        suspected_ids = _parse_fraud_list(detector_report, all_ids_set)

    if len(suspected_ids) == total:
        logger.warning("All transactions flagged — trimming to top 40% by detector order.")
        suspected_ids = suspected_ids[: max(1, int(total * 0.4))]

    print(f"  [Done]        {len(suspected_ids)} transactions flagged as suspected fraud.")

    return LevelReport(
        level=dataset.level,
        analyst_report=analyst_report,
        detector_report=detector_report,
        strategist_report=strategist_report,
        coordinator_report=coordinator_report,
        suspected_ids=suspected_ids,
    )
