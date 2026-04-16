from dotenv import load_dotenv

load_dotenv()

import os
from src.config import get_langfuse_client, generate_session_id, AGENT_NAME, CITY, SYSTEM_YEAR, TOTAL_LEVELS
from src.llm import get_analyst_model, get_detector_model, get_strategist_model, get_coordinator_model
from src.data import (
    generate_level_data, load_transactions_from_csv, save_transactions_to_csv,
    format_transactions, get_all_ids, get_hacker_history_text, LEVEL_PROFILES,
)
from src.agent import run_level_pipeline
from src.submission import save_submission, is_already_submitted, load_submission
from src.scorer import score_level, build_leaderboard, LevelScore
from src.utils import print_level_header, print_section, print_level_report, print_session_summary

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LEVELS_TO_RUN = list(range(1, TOTAL_LEVELS + 1))


def get_or_generate_data(level: int):
    train_path = os.path.join(DATA_DIR, f"level_{level}", "train.csv")
    eval_path = os.path.join(DATA_DIR, f"level_{level}", "eval.csv")

    if os.path.exists(train_path) and os.path.exists(eval_path):
        training = load_transactions_from_csv(train_path)
        evaluation = load_transactions_from_csv(eval_path)
        print(f"  [DATA] Loaded from CSV: {len(training)} training, {len(evaluation)} eval transactions.")
    else:
        print(f"  [DATA] No CSV found — generating synthetic data for Level {level}.")
        training, evaluation = generate_level_data(level)
        save_transactions_to_csv(training, train_path)
        eval_with_labels = evaluation[:]
        eval_no_labels = []
        for t in evaluation:
            import copy
            t_copy = copy.copy(t)
            t_copy.label = None
            eval_no_labels.append(t_copy)
        save_transactions_to_csv(eval_with_labels, eval_path)
        print(f"  [DATA] Saved: {len(training)} training, {len(evaluation)} eval transactions.")

    return training, evaluation


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
        print_level_header(level)

        training, evaluation = get_or_generate_data(level)
        training_text = format_transactions(training, include_label=True)
        eval_text = format_transactions(evaluation, include_label=False)
        hacker_history = get_hacker_history_text(level)

        print(f"  Training transactions : {len(training)}")
        print(f"  Eval transactions     : {len(evaluation)}")
        print(f"  Running 4-agent pipeline (Analyst -> Detector -> Strategist -> Coordinator)...")

        report = run_level_pipeline(
            session_id=session_id,
            analyst_model=analyst_model,
            detector_model=detector_model,
            strategist_model=strategist_model,
            coordinator_model=coordinator_model,
            training_text=training_text,
            eval_text=eval_text,
            eval_transactions=evaluation,
            hacker_history=hacker_history,
            level=level,
        )

        print_level_report(report)

        save_submission(level, report.predictions, evaluation)

        final_predictions = load_submission(level)
        score = score_level(level, evaluation, final_predictions)
        all_scores.append(score)

        print(f"\n  [SCORE] Level {level}: Accuracy={score.accuracy:.1%}  Precision={score.precision:.1%}"
              f"  Recall={score.recall:.1%}  F1={score.f1:.1%}  Threat={score.threat_level}")

    langfuse_client.flush()

    print(f"\n\n{'=' * 70}")
    print("  THE EYE — LEADERBOARD")
    print(build_leaderboard(all_scores))

    print_session_summary(session_id, len(LEVELS_TO_RUN))


if __name__ == "__main__":
    main()
