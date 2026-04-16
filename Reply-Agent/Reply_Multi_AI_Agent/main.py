from dotenv import load_dotenv

load_dotenv()

from src.config import get_langfuse_client, generate_session_id, AGENT_NAME, CITY, SYSTEM_YEAR
from src.llm import get_analyst_model, get_detector_model, get_strategist_model, get_coordinator_model
from src.data import generate_wave, format_transactions, get_suspicious_ids, get_hacker_history_text, WAVE_PROFILES
from src.agent import run_fraud_pipeline
from src.utils import print_wave_header, print_wave_report, print_session_summary


WAVES_TO_RUN = [1, 2, 3]


def main() -> None:
    print(f"\n{'=' * 70}")
    print(f"  {AGENT_NAME} — MirrorPay Fraud Intelligence System")
    print(f"  Location: {CITY}  |  Year: {SYSTEM_YEAR}")
    print(f"{'=' * 70}")

    langfuse_client = get_langfuse_client()
    session_id = generate_session_id()
    print(f"\n  Session ID: {session_id}")

    analyst_model = get_analyst_model()
    detector_model = get_detector_model()
    strategist_model = get_strategist_model()
    coordinator_model = get_coordinator_model()

    for wave in WAVES_TO_RUN:
        print_wave_header(wave)

        transactions = generate_wave(wave)
        transactions_text = format_transactions(transactions)
        transaction_ids = get_suspicious_ids(transactions)
        hacker_history = get_hacker_history_text(wave)

        print(f"\n  Transactions in batch : {len(transactions)}")
        print(f"  Running 4-agent pipeline (Analyst -> Detector -> Strategist -> Coordinator)...")

        report = run_fraud_pipeline(
            session_id=session_id,
            analyst_model=analyst_model,
            detector_model=detector_model,
            strategist_model=strategist_model,
            coordinator_model=coordinator_model,
            transactions_text=transactions_text,
            transaction_ids=transaction_ids,
            hacker_history=hacker_history,
            wave=wave,
        )

        print_wave_report(report)

    langfuse_client.flush()
    print_session_summary(session_id, len(WAVES_TO_RUN))


if __name__ == "__main__":
    main()
