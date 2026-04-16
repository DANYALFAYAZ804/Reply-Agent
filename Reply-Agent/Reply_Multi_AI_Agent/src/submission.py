import os
import csv
from typing import Dict, List
from src.data import Transaction


SUBMISSIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "submissions")


def _submission_path(level: int) -> str:
    return os.path.join(SUBMISSIONS_DIR, f"level_{level}_submission.csv")


def is_already_submitted(level: int) -> bool:
    return os.path.exists(_submission_path(level))


def save_submission(level: int, predictions: Dict[str, str], eval_transactions: List[Transaction]) -> str:
    path = _submission_path(level)
    if is_already_submitted(level):
        print(f"  [SUBMISSION] Level {level} already submitted. First submission is final — skipping overwrite.")
        return path
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    id_set = {t.txn_id for t in eval_transactions}
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["txn_id", "prediction"])
        for txn_id in sorted(id_set):
            prediction = predictions.get(txn_id, "legitimate")
            writer.writerow([txn_id, prediction])
    print(f"  [SUBMISSION] Level {level} submission saved -> {path}")
    return path


def load_submission(level: int) -> Dict[str, str]:
    path = _submission_path(level)
    result: Dict[str, str] = {}
    if not os.path.exists(path):
        return result
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[row["txn_id"]] = row["prediction"]
    return result
