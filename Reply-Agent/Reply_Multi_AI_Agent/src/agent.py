import re
from dataclasses import dataclass
from typing import List, Set
from langchain_core.messages import HumanMessage
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


@dataclass
class LevelReport:
    level: int
    analyst_report: str
    detector_report: str
    strategist_report: str
    coordinator_report: str
    suspected_ids: List[str]


def _call_agent(model, prompt: str, session_id: str) -> str:
    handler = CallbackHandler()
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(
        messages,
        config={
            "callbacks": [handler],
            "metadata": {"langfuse_session_id": session_id},
        },
    )
    return response.content


def _parse_fraud_list(coordinator_report: str, all_ids: Set[str]) -> List[str]:
    suspected = []
    in_block = False
    for line in coordinator_report.splitlines():
        stripped = line.strip()
        if stripped == "===FRAUD_LIST===":
            in_block = True
            continue
        if stripped == "===END_LIST===":
            in_block = False
            continue
        if in_block and stripped and stripped in all_ids:
            suspected.append(stripped)

    if not suspected:
        uuid_pattern = re.compile(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
        )
        for match in uuid_pattern.finditer(coordinator_report):
            uid = match.group(0)
            if uid in all_ids and uid not in suspected:
                suspected.append(uid)

    return suspected


@observe()
def run_analyst(session_id: str, model, dataset: LevelDataset) -> str:
    prompt = ANALYST_PROMPT.format(
        agent_name=AGENT_NAME,
        institution=INSTITUTION,
        city=CITY,
        year=SYSTEM_YEAR,
        level=dataset.level,
        transactions=format_transactions_block(dataset.transactions),
        locations=format_locations_block(dataset.locations),
        users=format_users_block(dataset.users),
        conversations=format_conversations_block(dataset.conversations),
        messages=format_messages_block(dataset.messages),
    )
    return _call_agent(model, prompt, session_id)


@observe()
def run_detector(session_id: str, model, analyst_report: str, dataset: LevelDataset) -> str:
    prompt = DETECTOR_PROMPT.format(
        agent_name=AGENT_NAME,
        institution=INSTITUTION,
        city=CITY,
        year=SYSTEM_YEAR,
        level=dataset.level,
        analyst_report=analyst_report,
        transactions=format_transactions_block(dataset.transactions, max_rows=150),
    )
    return _call_agent(model, prompt, session_id)


@observe()
def run_strategist(session_id: str, model, analyst_report: str, detector_report: str, dataset: LevelDataset) -> str:
    prompt = STRATEGIST_PROMPT.format(
        agent_name=AGENT_NAME,
        institution=INSTITUTION,
        city=CITY,
        year=SYSTEM_YEAR,
        level=dataset.level,
        detector_report=detector_report,
        hacker_history=get_hacker_history_text(dataset.level),
    )
    return _call_agent(model, prompt, session_id)


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
    prompt = COORDINATOR_PROMPT.format(
        agent_name=AGENT_NAME,
        institution=INSTITUTION,
        city=CITY,
        year=SYSTEM_YEAR,
        level=dataset.level,
        analyst_report=analyst_report,
        detector_report=detector_report,
        strategist_report=strategist_report,
        all_txn_ids="\n".join(all_ids),
    )
    return _call_agent(model, prompt, session_id)


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

    analyst_report = run_analyst(session_id, analyst_model, dataset)
    detector_report = run_detector(session_id, detector_model, analyst_report, dataset)
    strategist_report = run_strategist(session_id, strategist_model, analyst_report, detector_report, dataset)
    coordinator_report = run_coordinator(
        session_id, coordinator_model, analyst_report, detector_report, strategist_report, dataset
    )

    suspected_ids = _parse_fraud_list(coordinator_report, all_ids_set)

    total = len(all_ids_set)
    if len(suspected_ids) == 0 or len(suspected_ids) == total:
        fraud_lines = re.findall(
            r"FRAUD:\s*([0-9a-f\-]{36})", detector_report + "\n" + strategist_report, re.IGNORECASE
        )
        for fid in fraud_lines:
            if fid in all_ids_set and fid not in suspected_ids:
                suspected_ids.append(fid)

    return LevelReport(
        level=dataset.level,
        analyst_report=analyst_report,
        detector_report=detector_report,
        strategist_report=strategist_report,
        coordinator_report=coordinator_report,
        suspected_ids=suspected_ids,
    )
