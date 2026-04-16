from dataclasses import dataclass, field
from typing import Dict, List, Optional
from src.data import Transaction


@dataclass
class LevelScore:
    level: int
    total: int
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0

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
    def threat_level(self) -> str:
        if self.f1 >= 0.90:
            return "GREEN"
        elif self.f1 >= 0.75:
            return "YELLOW"
        elif self.f1 >= 0.55:
            return "ORANGE"
        return "RED"


def score_level(
    level: int,
    eval_transactions: List[Transaction],
    predictions: Dict[str, str],
) -> LevelScore:
    tp = fp = tn = fn = 0
    for txn in eval_transactions:
        true_label = txn.label or "legitimate"
        pred_label = predictions.get(txn.txn_id, "legitimate")
        if true_label == "fraudulent" and pred_label == "fraudulent":
            tp += 1
        elif true_label == "legitimate" and pred_label == "fraudulent":
            fp += 1
        elif true_label == "legitimate" and pred_label == "legitimate":
            tn += 1
        else:
            fn += 1
    return LevelScore(level=level, total=len(eval_transactions), tp=tp, fp=fp, tn=tn, fn=fn)


def build_leaderboard(scores: List[LevelScore]) -> str:
    lines = []
    divider = "=" * 72
    lines.append(divider)
    lines.append(f"  {'LEVEL':<8} {'ACC':>7} {'PREC':>7} {'REC':>7} {'F1':>7} {'THREAT':<10} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}")
    lines.append(divider)
    total_f1 = 0.0
    for s in scores:
        lines.append(
            f"  {'Level '+str(s.level):<8} {s.accuracy:>6.1%} {s.precision:>6.1%}"
            f" {s.recall:>6.1%} {s.f1:>6.1%} {s.threat_level:<10}"
            f" {s.tp:>4} {s.fp:>4} {s.tn:>4} {s.fn:>4}"
        )
        total_f1 += s.f1
    lines.append(divider)
    avg_f1 = total_f1 / len(scores) if scores else 0.0
    lines.append(f"  {'OVERALL':<8} {'':>7} {'':>7} {'':>7} {avg_f1:>6.1%} {'avg F1':<10}")
    lines.append(divider)
    return "\n".join(lines)
