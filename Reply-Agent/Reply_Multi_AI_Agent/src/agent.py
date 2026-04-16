import re
from dataclasses import dataclass
from typing import List, Dict
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
)
from src.data import Transaction, LEVEL_PROFILES


@dataclass
class LevelReport:
    level: int
    analyst_report: str
    detector_report: str
    strategist_report: str
    coordinator_report: str
    predictions: Dict[str, str]


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


def _parse_predictions(coordinator_report: str, eval_transactions: List[Transaction]) -> Dict[str, str]:
    predictions: Dict[str, str] = {}
    lines = coordinator_report.splitlines()
    for line in lines:
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                txn_id = parts[0].strip()
                prediction = parts[1].strip().lower()
                if prediction in ("fraudulent", "legitimate"):
                    predictions[txn_id] = prediction
    for txn in eval_transactions:
        if txn.txn_id not in predictions:
            text = coordinator_report.lower()
            pattern = re.escape(txn.txn_id.lower())
            match = re.search(pattern + r".*?(fraudulent|legitimate)", text)
            if match:
                predictions[txn.txn_id] = match.group(1)
            else:
                predictions[txn.txn_id] = "legitimate"
    return predictions


@observe()
def run_analyst(session_id: str, model, training_text: str, level: int) -> str:
    profile = LEVEL_PROFILES[level]
    prompt = ANALYST_PROMPT.format(
        agent_name=AGENT_NAME,
        institution=INSTITUTION,
        city=CITY,
        year=SYSTEM_YEAR,
        level=level,
        level_description=profile["description"],
        additional_data=profile["additional_data"],
        transactions=training_text,
    )
    return _call_agent(model, prompt, session_id)


@observe()
def run_detector(session_id: str, model, analyst_report: str, eval_text: str, level: int) -> str:
    profile = LEVEL_PROFILES[level]
    prompt = DETECTOR_PROMPT.format(
        agent_name=AGENT_NAME,
        institution=INSTITUTION,
        city=CITY,
        year=SYSTEM_YEAR,
        level=level,
        level_description=profile["description"],
        analyst_report=analyst_report,
        transactions=eval_text,
    )
    return _call_agent(model, prompt, session_id)


@observe()
def run_strategist(session_id: str, model, detector_report: str, hacker_history: str, level: int) -> str:
    profile = LEVEL_PROFILES[level]
    prompt = STRATEGIST_PROMPT.format(
        agent_name=AGENT_NAME,
        institution=INSTITUTION,
        city=CITY,
        year=SYSTEM_YEAR,
        level=level,
        level_description=profile["description"],
        detector_report=detector_report,
        hacker_history=hacker_history,
    )
    return _call_agent(model, prompt, session_id)


@observe()
def run_coordinator(
    session_id: str,
    model,
    analyst_report: str,
    detector_report: str,
    strategist_report: str,
    level: int,
) -> str:
    profile = LEVEL_PROFILES[level]
    prompt = COORDINATOR_PROMPT.format(
        agent_name=AGENT_NAME,
        institution=INSTITUTION,
        city=CITY,
        year=SYSTEM_YEAR,
        level=level,
        level_description=profile["description"],
        analyst_report=analyst_report,
        detector_report=detector_report,
        strategist_report=strategist_report,
    )
    return _call_agent(model, prompt, session_id)


@observe()
def run_level_pipeline(
    session_id: str,
    analyst_model,
    detector_model,
    strategist_model,
    coordinator_model,
    training_text: str,
    eval_text: str,
    eval_transactions: List[Transaction],
    hacker_history: str,
    level: int,
) -> LevelReport:
    analyst_report = run_analyst(session_id, analyst_model, training_text, level)
    detector_report = run_detector(session_id, detector_model, analyst_report, eval_text, level)
    strategist_report = run_strategist(session_id, strategist_model, detector_report, hacker_history, level)
    coordinator_report = run_coordinator(
        session_id, coordinator_model, analyst_report, detector_report, strategist_report, level
    )
    predictions = _parse_predictions(coordinator_report, eval_transactions)
    return LevelReport(
        level=level,
        analyst_report=analyst_report,
        detector_report=detector_report,
        strategist_report=strategist_report,
        coordinator_report=coordinator_report,
        predictions=predictions,
    )
