import os
from typing import List

SUBMISSIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "submissions")


_LEVEL_NAMES = {
    1: "1st_dataset",
    2: "2nd_dataset",
    3: "3rd_dataset",
    4: "4th_dataset",
    5: "5th_dataset",
}


def _submission_path(level: int) -> str:
    name = _LEVEL_NAMES.get(level, f"level_{level}")
    return os.path.join(SUBMISSIONS_DIR, f"{name}_output.txt")


def is_already_submitted(level: int) -> bool:
    return os.path.exists(_submission_path(level))


def save_submission(level: int, suspected_ids: List[str]) -> str:
    path = _submission_path(level)
    if is_already_submitted(level):
        print(f"  [SUBMISSION] Level {level} already submitted — first submission is final, skipping overwrite.")
        return path
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    with open(path, "w", encoding="ascii", errors="replace") as f:
        for txn_id in suspected_ids:
            f.write(txn_id + "\n")
    print(f"  [SUBMISSION] Level {level} output saved -> {path}  ({len(suspected_ids)} suspected fraud TXNs)")
    return path


def load_submission(level: int) -> List[str]:
    path = _submission_path(level)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="ascii", errors="replace") as f:
        return [line.strip() for line in f if line.strip()]
