import os
import csv
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Transaction:
    txn_id: str
    amount: float
    merchant: str
    category: str
    hour: int
    region: str
    jurisdiction: str
    frequency_last_hour: int
    citizen_id: str = ""
    citizen_age: int = 0
    citizen_risk_score: float = 0.0
    communication_flags: int = 0
    habit_deviation_score: float = 0.0
    label: Optional[str] = None

    def __str__(self) -> str:
        citizen_info = ""
        if self.citizen_id:
            citizen_info = (
                f" | citizen:{self.citizen_id} age:{self.citizen_age}"
                f" risk:{self.citizen_risk_score:.2f} comm_flags:{self.communication_flags}"
                f" habit_dev:{self.habit_deviation_score:.2f}"
            )
        label_str = f" | label:{self.label}" if self.label else ""
        return (
            f"TXN:{self.txn_id} | ${self.amount:.2f} | {self.merchant} ({self.category})"
            f" | {self.hour:02d}:00h | {self.region}/{self.jurisdiction}"
            f" | freq/hr:{self.frequency_last_hour}{citizen_info}{label_str}"
        )


LEVEL_PROFILES: Dict[int, Dict] = {
    1: {
        "description": "Level 1 — Daytime retail targeting (basic patterns)",
        "fraud_merchants": ["QuickMart-Fake", "NeonGrocer-Clone", "DataFuel Spoof"],
        "fraud_categories": ["retail", "fuel", "grocery"],
        "fraud_hours": list(range(9, 18)),
        "fraud_regions": ["Central District", "East Quarter"],
        "fraud_jurisdictions": ["MR-ALPHA"],
        "fraud_amount_range": (50, 400),
        "fraud_freq_range": (1, 3),
        "legit_merchants": ["QuickMart", "NeonGrocer", "DataFuel Station", "FreshMarket"],
        "legit_categories": ["retail", "fuel", "grocery", "pharmacy"],
        "legit_amount_range": (10, 300),
        "additional_data": "basic demographics",
    },
    2: {
        "description": "Level 2 — Late-night entertainment & cross-border",
        "fraud_merchants": ["SynthBar-Ghost", "HoloCasino-Shadow", "NightStream-Fake"],
        "fraud_categories": ["entertainment", "gambling", "streaming"],
        "fraud_hours": list(range(22, 24)) + list(range(0, 5)),
        "fraud_regions": ["West Fringe", "Border Zone"],
        "fraud_jurisdictions": ["MR-BETA", "MR-GAMMA"],
        "fraud_amount_range": (200, 1500),
        "fraud_freq_range": (3, 8),
        "legit_merchants": ["SynthBar", "CinemaX", "StreamHub", "NightDiner"],
        "legit_categories": ["entertainment", "dining", "streaming", "transport"],
        "legit_amount_range": (20, 800),
        "additional_data": "demographics + communication metadata",
    },
    3: {
        "description": "Level 3 — High-value B2B impersonation",
        "fraud_merchants": ["MirrorLogistics-Fake", "SyntheticSupply Ghost", "DataBroker Rogue"],
        "fraud_categories": ["logistics", "wholesale", "data-brokerage"],
        "fraud_hours": list(range(6, 10)),
        "fraud_regions": ["Finance Spire", "Corporate Arc"],
        "fraud_jurisdictions": ["MR-DELTA", "MR-EPSILON"],
        "fraud_amount_range": (5000, 50000),
        "fraud_freq_range": (1, 2),
        "legit_merchants": ["MirrorLogistics", "SupplyChain Co.", "DataExchange Ltd"],
        "legit_categories": ["logistics", "wholesale", "finance"],
        "legit_amount_range": (1000, 30000),
        "additional_data": "demographics + communications + transaction history",
    },
    4: {
        "description": "Level 4 — Identity morphing & micro-transaction layering",
        "fraud_merchants": ["PayNode-Phantom", "CryptoRelay-X", "MicroVault-Ghost"],
        "fraud_categories": ["crypto", "transfer", "micro-payment"],
        "fraud_hours": list(range(0, 24)),
        "fraud_regions": ["Shadow Sector", "Relay District", "Anonymous Zone"],
        "fraud_jurisdictions": ["MR-ZETA", "MR-ETA", "MR-THETA"],
        "fraud_amount_range": (1, 50),
        "fraud_freq_range": (15, 40),
        "legit_merchants": ["PayNode", "CryptoRelay", "MicroVault"],
        "legit_categories": ["crypto", "transfer", "micro-payment"],
        "legit_amount_range": (1, 200),
        "additional_data": "full demographics + communications + habits + network graph",
    },
    5: {
        "description": "Level 5 — Hybrid multi-vector coordinated attack",
        "fraud_merchants": ["OmniPay-Mirage", "TrustNet-Fake", "ClearPath-Ghost", "NexusBank-Clone"],
        "fraud_categories": ["omni-payment", "trust-transfer", "clearance", "banking"],
        "fraud_hours": list(range(0, 24)),
        "fraud_regions": ["All Sectors"],
        "fraud_jurisdictions": ["MR-IOTA", "MR-KAPPA", "MR-LAMBDA"],
        "fraud_amount_range": (100, 100000),
        "fraud_freq_range": (1, 20),
        "legit_merchants": ["OmniPay", "TrustNet", "ClearPath", "NexusBank"],
        "legit_categories": ["omni-payment", "trust-transfer", "clearance", "banking"],
        "legit_amount_range": (50, 50000),
        "additional_data": "full citizen profile: demographics + comms + habits + network + behavioural biometrics",
    },
}

HACKER_HISTORY: Dict[int, str] = {
    1: "Level 1: small-amount daytime retail clones; single jurisdiction MR-ALPHA; low velocity 1-3/hr.",
    2: "Level 2: late-night entertainment/gambling; cross-border MR-BETA/MR-GAMMA; medium-high velocity 3-8/hr; $200-$1,500.",
    3: "Level 3: early-morning B2B impersonation; high-value $5k-$50k; Corporate Arc; MR-DELTA/MR-EPSILON; very low frequency.",
    4: "Level 4: identity morphing + micro-transaction layering across anonymous zones; ultra-high frequency 15-40/hr; tiny amounts $1-$50.",
}


def _make_txn_id(level: int, index: int, split: str = "train") -> str:
    prefix = "TR" if split == "train" else "EV"
    return f"L{level}-{prefix}-{index:04d}"


def _make_citizen_id(index: int) -> str:
    return f"CIT-{index:05d}"


def _generate_transactions(
    level: int,
    n_legit: int,
    n_fraud: int,
    split: str,
    seed_offset: int = 0,
) -> List[Transaction]:
    profile = LEVEL_PROFILES[level]
    transactions: List[Transaction] = []

    for i in range(n_legit):
        cit_idx = random.randint(1000, 9999)
        transactions.append(Transaction(
            txn_id=_make_txn_id(level, i + 1 + seed_offset, split),
            amount=round(random.uniform(*profile["legit_amount_range"]), 2),
            merchant=random.choice(profile["legit_merchants"]),
            category=random.choice(profile["legit_categories"]),
            hour=random.randint(8, 20),
            region="Central District",
            jurisdiction="MR-ALPHA",
            frequency_last_hour=random.randint(1, 2),
            citizen_id=_make_citizen_id(cit_idx),
            citizen_age=random.randint(18, 75),
            citizen_risk_score=round(random.uniform(0.0, 0.3), 2),
            communication_flags=random.randint(0, 1),
            habit_deviation_score=round(random.uniform(0.0, 0.2), 2),
            label="legitimate",
        ))

    for i in range(n_fraud):
        cit_idx = random.randint(1000, 9999)
        transactions.append(Transaction(
            txn_id=_make_txn_id(level, 500 + i + 1 + seed_offset, split),
            amount=round(random.uniform(*profile["fraud_amount_range"]), 2),
            merchant=random.choice(profile["fraud_merchants"]),
            category=random.choice(profile["fraud_categories"]),
            hour=random.choice(profile["fraud_hours"]),
            region=random.choice(profile["fraud_regions"]),
            jurisdiction=random.choice(profile["fraud_jurisdictions"]),
            frequency_last_hour=random.randint(*profile["fraud_freq_range"]),
            citizen_id=_make_citizen_id(cit_idx),
            citizen_age=random.randint(18, 75),
            citizen_risk_score=round(random.uniform(0.6, 1.0), 2),
            communication_flags=random.randint(2, 8),
            habit_deviation_score=round(random.uniform(0.5, 1.0), 2),
            label="fraudulent",
        ))

    random.shuffle(transactions)
    return transactions


def generate_level_data(level: int, n_train_legit: int = 20, n_train_fraud: int = 8,
                        n_eval_legit: int = 15, n_eval_fraud: int = 6):
    training = _generate_transactions(level, n_train_legit, n_train_fraud, "train")
    evaluation = _generate_transactions(level, n_eval_legit, n_eval_fraud, "eval", seed_offset=200)
    return training, evaluation


def load_transactions_from_csv(filepath: str) -> List[Transaction]:
    transactions = []
    if not os.path.exists(filepath):
        return transactions
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            transactions.append(Transaction(
                txn_id=row.get("txn_id", ""),
                amount=float(row.get("amount", 0)),
                merchant=row.get("merchant", ""),
                category=row.get("category", ""),
                hour=int(row.get("hour", 0)),
                region=row.get("region", ""),
                jurisdiction=row.get("jurisdiction", ""),
                frequency_last_hour=int(row.get("frequency_last_hour", 0)),
                citizen_id=row.get("citizen_id", ""),
                citizen_age=int(row.get("citizen_age", 0)),
                citizen_risk_score=float(row.get("citizen_risk_score", 0)),
                communication_flags=int(row.get("communication_flags", 0)),
                habit_deviation_score=float(row.get("habit_deviation_score", 0)),
                label=row.get("label", None) or None,
            ))
    return transactions


def save_transactions_to_csv(transactions: List[Transaction], filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "txn_id", "amount", "merchant", "category", "hour",
            "region", "jurisdiction", "frequency_last_hour",
            "citizen_id", "citizen_age", "citizen_risk_score",
            "communication_flags", "habit_deviation_score", "label",
        ])
        for t in transactions:
            writer.writerow([
                t.txn_id, t.amount, t.merchant, t.category, t.hour,
                t.region, t.jurisdiction, t.frequency_last_hour,
                t.citizen_id, t.citizen_age, t.citizen_risk_score,
                t.communication_flags, t.habit_deviation_score,
                t.label if t.label else "",
            ])


def format_transactions(transactions: List[Transaction], include_label: bool = True) -> str:
    lines = []
    for t in transactions:
        original_label = t.label
        if not include_label:
            t.label = None
        lines.append(str(t))
        t.label = original_label
    return "\n".join(lines)


def get_all_ids(transactions: List[Transaction]) -> str:
    return ", ".join(t.txn_id for t in transactions)


def get_hacker_history_text(up_to_level: int) -> str:
    lines = []
    for lv in range(1, up_to_level):
        if lv in HACKER_HISTORY:
            lines.append(f"  - {HACKER_HISTORY[lv]}")
    return "\n".join(lines) if lines else "No prior level history available."
