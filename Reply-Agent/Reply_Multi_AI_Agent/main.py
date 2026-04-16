from dotenv import load_dotenv

load_dotenv()

import os
from typing import Set
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

LEVELS_TO_RUN = [lvl for lvl in range(1, TOTAL_LEVELS + 1)
                 if os.path.exists(os.path.join(DATA_DIR, f"level_{lvl}", "transactions.csv"))
                 or os.path.exists(os.path.join(DATA_DIR, f"level_{lvl}", "Transactions.csv"))]


def _load_ground_truth(level: int) -> Set[str]:
    path = os.path.join(DATA_DIR, f"level_{level}", "ground_truth.txt")
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def main() -> None:
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
        print_level_header(level, name=name)

        name = LEVEL_NAMES.get(level, f"Level {level}")
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

        if ground_truth:
            score = score_level(level, all_txn_ids, final_ids, ground_truth)
            all_scores.append(score)
            print(
                f"\n  [SCORE] Level {level}: {score.validity_label}"
                f" | Flagged={score.flagged} TP={score.tp} FP={score.fp} FN={score.fn}"
                f" | Precision={score.precision:.1%} Recall={score.recall:.1%}"
                f" F1={score.f1:.1%} Cost={score.asymmetric_cost:.1f}"
            )
        else:
            print(f"\n  [SCORE] No ground_truth.txt for Level {level} — submission saved, scoring skipped.")
            print(f"  [INFO]  Flagged {len(final_ids)} suspected fraud transactions.")

    langfuse_client.flush()

    if all_scores:
        print(f"\n\n{'=' * 70}")
        print("  THE EYE — LEADERBOARD")
        print(build_leaderboard(all_scores))

    print_session_summary(session_id, len(LEVELS_TO_RUN))


if __name__ == "__main__":
    main()
