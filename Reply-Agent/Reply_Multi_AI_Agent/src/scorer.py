from dataclasses import dataclass
from typing import List, Set


@dataclass
class LevelScore:
    level: int
    total_txns: int
    flagged: int
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total_txns if self.total_txns else 0.0

    @property
    def fp_rate(self) -> float:
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) else 0.0

    @property
    def asymmetric_cost(self) -> float:
        fp_cost = self.fp * 1.0
        fn_cost = self.fn * 3.0
        return fp_cost + fn_cost

    @property
    def submission_valid(self) -> bool:
        if self.flagged == 0:
            return False
        if self.flagged == self.total_txns:
            return False
        total_fraud = self.tp + self.fn
        if total_fraud > 0 and self.tp / total_fraud < 0.15:
            return False
        return True

    @property
    def validity_label(self) -> str:
        return "VALID" if self.submission_valid else "INVALID"


def score_level(
    level: int,
    all_txn_ids: List[str],
    suspected_ids: List[str],
    ground_truth_fraud_ids: Set[str],
) -> LevelScore:
    suspected_set = set(suspected_ids)
    all_set = set(all_txn_ids)
    tp = len(suspected_set & ground_truth_fraud_ids)
    fp = len(suspected_set - ground_truth_fraud_ids)
    fn = len(ground_truth_fraud_ids - suspected_set)
    tn = len(all_set - suspected_set - ground_truth_fraud_ids)
    return LevelScore(
        level=level,
        total_txns=len(all_txn_ids),
        flagged=len(suspected_ids),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def build_leaderboard(scores: List[LevelScore]) -> str:
    W = 76
    divider = "=" * W
    lines = [
        divider,
        f"  {'LEVEL':<10} {'VALID':<8} {'FLAGGED':>8} {'TP':>5} {'FP':>5} {'FN':>5}"
        f" {'PREC':>7} {'REC':>7} {'F1':>7} {'COST':>8}",
        divider,
    ]
    total_cost = 0.0
    total_f1 = 0.0
    for s in scores:
        lines.append(
            f"  {'Level '+str(s.level):<10} {s.validity_label:<8} {s.flagged:>8}"
            f" {s.tp:>5} {s.fp:>5} {s.fn:>5}"
            f" {s.precision:>6.1%} {s.recall:>6.1%} {s.f1:>6.1%} {s.asymmetric_cost:>8.1f}"
        )
        total_cost += s.asymmetric_cost
        total_f1 += s.f1
    avg_f1 = total_f1 / len(scores) if scores else 0.0
    lines.append(divider)
    lines.append(
        f"  {'OVERALL':<10} {'':8} {'':>8} {'':>5} {'':>5} {'':>5}"
        f" {'':>7} {'':>7} {avg_f1:>6.1%} {total_cost:>8.1f}"
    )
    lines.append(f"  Note: asymmetric cost = FP*1 + FN*3 (FN penalised 3x)")
    lines.append(divider)
    return "\n".join(lines)
