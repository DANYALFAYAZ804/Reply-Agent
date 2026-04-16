from dotenv import load_dotenv
import os
import logging

_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=_ENV_PATH)

logging.getLogger("opentelemetry.exporter.otlp").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry.sdk").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

from typing import Set


def _check_env() -> None:
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key or key == "your-api-key-here":
        raise SystemExit(
            "\n[SETUP ERROR] OPENROUTER_API_KEY is not set.\n"
            f"Edit the file: {_ENV_PATH}\n"
            "Replace 'your-api-key-here' with your real key from https://openrouter.ai\n"
        )
    lf_pub = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    if not lf_pub or lf_pub == "pk-your-public-key-here":
        raise SystemExit(
            "\n[SETUP ERROR] LANGFUSE_PUBLIC_KEY is not set.\n"
            f"Edit the file: {_ENV_PATH}\n"
            "Replace the placeholder values with your real Langfuse credentials.\n"
        )
from src.config import get_langfuse_client, generate_session_id, AGENT_NAME, CITY, SYSTEM_YEAR, TOTAL_LEVELS
from src.llm import get_analyst_model, get_detector_model, get_strategist_model, get_coordinator_model
from src.data import load_level_dataset, get_all_txn_ids, LevelDataset
from src.agent import run_level_pipeline
from src.submission import save_submission, load_submission
from src.scorer import score_level, build_leaderboard, LevelScore
from src.utils import print_level_header, print_section, print_level_report, print_session_summary

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

LEVEL_NAMES = {
    1: "Brave New World",
    2: "Deus Ex",
    3: "The Truman Show",
    4: "Level 4",
    5: "Level 5",
}

LEVELS_TO_RUN = [1]


def _load_ground_truth(level: int) -> Set[str]:
    path = os.path.join(DATA_DIR, f"level_{level}", "ground_truth.txt")
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def main() -> None:
    _check_env()
    print(f"\n{'=' * 70}")
    print(f"  {AGENT_NAME} — MirrorPay Fraud Intelligence System")
    print(f"  Location: {CITY}  |  Year: {SYSTEM_YEAR}  |  Levels: {TOTAL_LEVELS}")
    print(f"{'=' * 70}")

    langfuse_client = get_langfuse_client()
    session_id = generate_session_id()
    print(f"\n  Session ID: {session_id}")

    analyst_model = get_analyst_model()
    detector_model = get_detector_model()
    strategist_model = get_strategist_model()
    coordinator_model = get_coordinator_model()

    all_scores: list[LevelScore] = []

    for level in LEVELS_TO_RUN:
        name = LEVEL_NAMES.get(level, f"Level {level}")
        print_level_header(level, name=name)

        dataset = load_level_dataset(DATA_DIR, level, name=name)

        if not dataset.transactions:
            print(f"  [SKIP] No transactions.csv found for Level {level} — place data in data/level_{level}/")
            continue

        print(f"  Transactions   : {len(dataset.transactions)}")
        print(f"  Locations      : {len(dataset.locations)}")
        print(f"  Users          : {len(dataset.users)}")
        print(f"  Conversations  : {len(dataset.conversations)}")
        print(f"  Messages       : {len(dataset.messages)}")
        print(f"  Running 4-agent pipeline (Analyst -> Detector -> Strategist -> Coordinator)...")

        report = run_level_pipeline(
            session_id=session_id,
            analyst_model=analyst_model,
            detector_model=detector_model,
            strategist_model=strategist_model,
            coordinator_model=coordinator_model,
            dataset=dataset,
        )

        print_level_report(report)

        save_submission(level, report.suspected_ids)

        final_ids = load_submission(level)
        all_txn_ids = get_all_txn_ids(dataset.transactions)
        ground_truth = _load_ground_truth(level)

        total_txns = len(all_txn_ids)
        confirmed_fraud = final_ids
        confirmed_legit = [t for t in all_txn_ids if t not in set(final_ids)]

        print(f"\n  {'─' * 60}")
        print(f"  RESULTS — Level {level}: {name}")
        print(f"  {'─' * 60}")
        print(f"  Total transactions : {total_txns}")
        print(f"  Confirmed FRAUD    : {len(confirmed_fraud)}")
        print(f"  Confirmed LEGIT    : {len(confirmed_legit)}")
        print(f"  {'─' * 60}")

        if ground_truth:
            score = score_level(level, all_txn_ids, final_ids, ground_truth)
            all_scores.append(score)
            print(
                f"  [SCORE] {score.validity_label}"
                f" | TP={score.tp} FP={score.fp} FN={score.fn}"
                f" | Precision={score.precision:.1%} Recall={score.recall:.1%}"
                f" | F1={score.f1:.1%} Cost={score.asymmetric_cost:.1f}"
            )
        else:
            print(f"  [INFO]  No ground_truth.txt found — submission saved, scoring skipped.")
        print(f"  {'─' * 60}")

    langfuse_client.flush()

    if all_scores:
        print(f"\n\n{'=' * 70}")
        print("  THE EYE — LEADERBOARD")
        print(build_leaderboard(all_scores))

    print_session_summary(session_id, len(LEVELS_TO_RUN))


if __name__ == "__main__":
    main()
