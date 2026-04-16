import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Set, Optional, Tuple
from langchain_core.messages import HumanMessage

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
    format_transactions_compact,
    format_locations_block,
    format_users_block,
    format_conversations_block,
    format_messages_block,
    get_all_txn_ids,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_DELAY = 2
MAX_WORKERS = 8

_UUID = re.compile(
    r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
    re.IGNORECASE,
)


@dataclass
class LevelReport:
    level: int
    analyst_report: str
    detector_report: str
    strategist_report: str
    coordinator_report: str
    suspected_ids: List[str]


def _call_agent(model, prompt: str, session_id: str, agent_name: str = "agent") -> str:
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            last_error = e
            err_str = str(e)

            if "401" in err_str or "403" in err_str:
                raise RuntimeError(
                    f"\n[AUTH ERROR] {agent_name} — OPENROUTER_API_KEY is invalid.\n"
                    f"Edit Reply_Multi_AI_Agent/.env and set a real key from https://openrouter.ai\n"
                ) from e

            if "429" in err_str or "rate" in err_str.lower():
                wait = RETRY_DELAY * attempt * 3
                logger.warning(f"  [{agent_name}] rate limited — waiting {wait}s...")
                time.sleep(wait)
                continue

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)

    raise RuntimeError(f"[{agent_name}] failed after {MAX_RETRIES} attempts. Last: {last_error}")


def _extract_uuids_from_prefixed_lines(text: str, all_ids: Set[str], prefixes: tuple) -> List[str]:
    results: List[str] = []
    seen: Set[str] = set()
    for line in text.splitlines():
        stripped = line.strip().upper()
        if any(stripped.startswith(p) for p in prefixes):
            for match in _UUID.finditer(line):
                uid = match.group(1)
                if uid in all_ids and uid not in seen:
                    results.append(uid)
                    seen.add(uid)
    return results


def _parse_fraud_list(text: str, all_ids: Set[str]) -> List[str]:
    suspected: List[str] = []
    seen: Set[str] = set()
    in_block = False

    for line in text.splitlines():
        stripped = line.strip()
        if "===FRAUD_LIST===" in stripped:
            in_block = True
            continue
        if "===END_LIST===" in stripped:
            in_block = False
            continue
        if in_block and stripped and stripped in all_ids and stripped not in seen:
            suspected.append(stripped)
            seen.add(stripped)

    if not suspected:
        for match in _UUID.finditer(text):
            uid = match.group(1)
            if uid in all_ids and uid not in seen:
                suspected.append(uid)
                seen.add(uid)

    return suspected


def _run_parallel(model, prompts: List[Tuple[int, str]], session_id: str, role: str) -> List[Tuple[int, str]]:
    results: List[Tuple[int, str]] = []
    total = len(prompts)

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, total)) as executor:
        future_to_idx = {
            executor.submit(_call_agent, model, prompt, session_id, f"{role}-b{idx}"): idx
            for idx, prompt in prompts
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()
            results.append((idx, result))
            print(f"    [{role}] batch {idx}/{total} done", flush=True)

    results.sort(key=lambda x: x[0])
    return results


def run_analyst(session_id: str, model, dataset: LevelDataset) -> str:
    txns = dataset.transactions
    chunk_size = 120
    loc_text = format_locations_block(dataset.locations, max_rows=15)
    usr_text = format_users_block(dataset.users, max_rows=8)
    conv_text = format_conversations_block(dataset.conversations, max_rows=6)
    msg_text = format_messages_block(dataset.messages, max_rows=4)

    chunks = [txns[i:i + chunk_size] for i in range(0, len(txns), chunk_size)]
    total = len(chunks)
    prompts = []
    for idx, chunk in enumerate(chunks, 1):
        txn_text = format_transactions_block(chunk, max_rows=chunk_size)
        prompt = ANALYST_PROMPT.format(
            agent_name=AGENT_NAME,
            institution=INSTITUTION,
            city=CITY,
            year=SYSTEM_YEAR,
            level=dataset.level,
            transactions=f"(batch {idx}/{total})\n{txn_text}",
            locations=loc_text if idx == 1 else "(see batch 1)",
            users=usr_text if idx == 1 else "(see batch 1)",
            conversations=conv_text if idx == 1 else "(see batch 1)",
            messages=msg_text if idx == 1 else "(see batch 1)",
        )
        prompts.append((idx, prompt))

    results = _run_parallel(model, prompts, session_id, "Analyst")
    return "\n\n".join(f"--- Batch {idx} ---\n{text}" for idx, text in results)


def run_detector(session_id: str, model, analyst_report: str, dataset: LevelDataset) -> str:
    txns = dataset.transactions
    chunk_size = 150
    analyst_summary = analyst_report[:2000] + ("..." if len(analyst_report) > 2000 else "")

    chunks = [txns[i:i + chunk_size] for i in range(0, len(txns), chunk_size)]
    total = len(chunks)
    prompts = []
    for idx, chunk in enumerate(chunks, 1):
        txn_text = format_transactions_compact(chunk, max_rows=chunk_size)
        prompt = DETECTOR_PROMPT.format(
            agent_name=AGENT_NAME,
            institution=INSTITUTION,
            city=CITY,
            year=SYSTEM_YEAR,
            level=dataset.level,
            analyst_report=analyst_summary,
            transactions=f"(batch {idx}/{total})\n{txn_text}",
        )
        prompts.append((idx, prompt))

    results = _run_parallel(model, prompts, session_id, "Detector")
    return "\n\n".join(f"--- Batch {idx} ---\n{text}" for idx, text in results)


def run_strategist(session_id: str, model, detector_report: str, dataset: LevelDataset) -> str:
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


def run_coordinator(
    session_id: str,
    model,
    detector_report: str,
    strategist_report: str,
    dataset: LevelDataset,
) -> str:
    all_ids = get_all_txn_ids(dataset.transactions)
    all_ids_set = set(all_ids)

    detector_summary = detector_report[:2500] + ("..." if len(detector_report) > 2500 else "")
    strategist_summary = strategist_report[:800] + ("..." if len(strategist_report) > 800 else "")

    BATCH_SIZE = 300
    id_chunks = [all_ids[i:i + BATCH_SIZE] for i in range(0, len(all_ids), BATCH_SIZE)]
    all_suspected: List[str] = []
    seen: Set[str] = set()

    prompts = []
    for idx, id_chunk in enumerate(id_chunks, 1):
        prompt = COORDINATOR_PROMPT.format(
            agent_name=AGENT_NAME,
            institution=INSTITUTION,
            city=CITY,
            year=SYSTEM_YEAR,
            level=dataset.level,
            detector_report=detector_summary if idx == 1 else "(see batch 1)",
            strategist_report=strategist_summary if idx == 1 else "(see batch 1)",
            all_txn_ids="\n".join(id_chunk),
        )
        prompts.append((idx, prompt))

    results = _run_parallel(model, prompts, session_id, "Coordinator")

    for _, result in results:
        for bid in _parse_fraud_list(result, all_ids_set):
            if bid not in seen:
                all_suspected.append(bid)
                seen.add(bid)

    return (
        f"===FRAUD_LIST===\n"
        + "\n".join(all_suspected)
        + "\n===END_LIST==="
    )


def run_level_pipeline(
    session_id: str,
    analyst_model,
    detector_model,
    strategist_model,
    coordinator_model,
    dataset: LevelDataset,
) -> LevelReport:
    all_ids = get_all_txn_ids(dataset.transactions)
    all_ids_set = set(all_ids)
    total = len(all_ids_set)

    print(f"  [Analyst]     {len(dataset.transactions)} transactions — parallel batches...")
    analyst_report = run_analyst(session_id, analyst_model, dataset)

    print(f"  [Detector]    scoring with compact fields — parallel batches...")
    detector_report = run_detector(session_id, detector_model, analyst_report, dataset)

    print(f"  [Strategist]  hunting missed fraud...")
    strategist_report = run_strategist(session_id, strategist_model, detector_report, dataset)

    print(f"  [Coordinator] producing final confirmed fraud list...")
    coordinator_report = run_coordinator(
        session_id, coordinator_model, detector_report, strategist_report, dataset
    )

    suspected_ids = _parse_fraud_list(coordinator_report, all_ids_set)

    if len(suspected_ids) == 0:
        logger.warning("  [Warning] Coordinator returned 0 IDs — falling back to Detector FRAUD lines.")
        suspected_ids = _extract_uuids_from_prefixed_lines(
            detector_report, all_ids_set, prefixes=("FRAUD:",)
        )

    if len(suspected_ids) == 0:
        logger.warning("  [Warning] Still 0 IDs — UUID fallback from detector.")
        suspected_ids = _parse_fraud_list(detector_report, all_ids_set)

    if len(suspected_ids) == total:
        logger.warning("  [Warning] All transactions flagged — trimming to top 35%.")
        suspected_ids = suspected_ids[:max(1, int(total * 0.35))]

    print(f"  [Done]        {len(suspected_ids)} confirmed fraud transactions.")

    return LevelReport(
        level=dataset.level,
        analyst_report=analyst_report,
        detector_report=detector_report,
        strategist_report=strategist_report,
        coordinator_report=coordinator_report,
        suspected_ids=suspected_ids,
    )
