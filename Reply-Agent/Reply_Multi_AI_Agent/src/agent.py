from dataclasses import dataclass
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
)


@dataclass
class WaveReport:
    wave: int
    analyst_report: str
    detector_report: str
    strategist_report: str
    coordinator_report: str


def _call_agent(model, system_prompt: str, session_id: str) -> str:
    handler = CallbackHandler()
    messages = [HumanMessage(content=system_prompt)]
    response = model.invoke(
        messages,
        config={
            "callbacks": [handler],
            "metadata": {"langfuse_session_id": session_id},
        },
    )
    return response.content


@observe()
def run_analyst(session_id: str, model, transactions_text: str) -> str:
    prompt = ANALYST_PROMPT.format(
        agent_name=AGENT_NAME,
        institution=INSTITUTION,
        city=CITY,
        year=SYSTEM_YEAR,
        transactions=transactions_text,
    )
    return _call_agent(model, prompt, session_id)


@observe()
def run_detector(session_id: str, model, analyst_report: str, transaction_ids: str) -> str:
    prompt = DETECTOR_PROMPT.format(
        agent_name=AGENT_NAME,
        institution=INSTITUTION,
        city=CITY,
        year=SYSTEM_YEAR,
        analyst_report=analyst_report,
        transaction_ids=transaction_ids,
    )
    return _call_agent(model, prompt, session_id)


@observe()
def run_strategist(session_id: str, model, detector_report: str, hacker_history: str) -> str:
    prompt = STRATEGIST_PROMPT.format(
        agent_name=AGENT_NAME,
        institution=INSTITUTION,
        city=CITY,
        year=SYSTEM_YEAR,
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
) -> str:
    prompt = COORDINATOR_PROMPT.format(
        agent_name=AGENT_NAME,
        institution=INSTITUTION,
        city=CITY,
        year=SYSTEM_YEAR,
        analyst_report=analyst_report,
        detector_report=detector_report,
        strategist_report=strategist_report,
    )
    return _call_agent(model, prompt, session_id)


@observe()
def run_fraud_pipeline(
    session_id: str,
    analyst_model,
    detector_model,
    strategist_model,
    coordinator_model,
    transactions_text: str,
    transaction_ids: str,
    hacker_history: str,
    wave: int,
) -> WaveReport:
    analyst_report = run_analyst(session_id, analyst_model, transactions_text)
    detector_report = run_detector(session_id, detector_model, analyst_report, transaction_ids)
    strategist_report = run_strategist(session_id, strategist_model, detector_report, hacker_history)
    coordinator_report = run_coordinator(
        session_id, coordinator_model, analyst_report, detector_report, strategist_report
    )
    return WaveReport(
        wave=wave,
        analyst_report=analyst_report,
        detector_report=detector_report,
        strategist_report=strategist_report,
        coordinator_report=coordinator_report,
    )
