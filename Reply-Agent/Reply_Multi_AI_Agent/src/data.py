import random
from dataclasses import dataclass, field
from typing import List, Dict


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
    is_suspicious: bool = False

    def __str__(self) -> str:
        flag = " [FLAGGED]" if self.is_suspicious else ""
        return (
            f"TXN:{self.txn_id} | ${self.amount:.2f} | {self.merchant} ({self.category})"
            f" | {self.hour:02d}:00h | {self.region}/{self.jurisdiction}"
            f" | freq/hr:{self.frequency_last_hour}{flag}"
        )


WAVE_PROFILES: Dict[int, Dict] = {
    1: {
        "description": "Wave 1 — Daytime retail targeting",
        "merchants": ["QuickMart", "NeonGrocer", "DataFuel Station"],
        "categories": ["retail", "fuel", "grocery"],
        "hours": list(range(9, 18)),
        "regions": ["Central District", "East Quarter"],
        "jurisdictions": ["MR-ALPHA"],
        "amount_range": (50, 400),
        "freq_range": (1, 3),
    },
    2: {
        "description": "Wave 2 — Late-night entertainment & cross-border",
        "merchants": ["SynthBar", "HoloCasino", "NightStream"],
        "categories": ["entertainment", "gambling", "streaming"],
        "hours": list(range(22, 24)) + list(range(0, 5)),
        "regions": ["West Fringe", "Border Zone"],
        "jurisdictions": ["MR-BETA", "MR-GAMMA"],
        "amount_range": (200, 1500),
        "freq_range": (3, 8),
    },
    3: {
        "description": "Wave 3 — High-value B2B impersonation",
        "merchants": ["MirrorLogistics", "SyntheticSupply Co.", "DataBroker Hub"],
        "categories": ["logistics", "wholesale", "data-brokerage"],
        "hours": list(range(6, 10)),
        "regions": ["Finance Spire", "Corporate Arc"],
        "jurisdictions": ["MR-DELTA", "MR-EPSILON"],
        "amount_range": (5000, 50000),
        "freq_range": (1, 2),
    },
}

HACKER_HISTORY: Dict[int, str] = {
    1: (
        "Wave 1 tactics: small-amount daytime retail transactions across Central District; "
        "single jurisdiction (MR-ALPHA); low velocity (1-3 per hour)."
    ),
    2: (
        "Wave 2 tactics: late-night entertainment and gambling charges; "
        "cross-border into MR-BETA and MR-GAMMA; medium-high velocity (3-8 per hour); "
        "amounts escalated to $200-$1,500."
    ),
    3: (
        "Wave 3 tactics: early-morning B2B impersonation; high-value single transactions ($5k-$50k); "
        "Corporate Arc and Finance Spire regions; new jurisdictions MR-DELTA and MR-EPSILON; "
        "very low frequency to avoid velocity triggers."
    ),
}


def _make_txn_id(wave: int, index: int) -> str:
    return f"W{wave}-TXN-{index:04d}"


def generate_wave(wave: int, n_legit: int = 12, n_fraud: int = 5) -> List[Transaction]:
    profile = WAVE_PROFILES[wave]
    transactions: List[Transaction] = []

    legit_profile = WAVE_PROFILES[max(1, wave - 1)]
    for i in range(n_legit):
        transactions.append(Transaction(
            txn_id=_make_txn_id(wave, i + 1),
            amount=round(random.uniform(*legit_profile["amount_range"]), 2),
            merchant=random.choice(legit_profile["merchants"]),
            category=random.choice(legit_profile["categories"]),
            hour=random.choice(list(range(8, 20))),
            region=random.choice(legit_profile["regions"]),
            jurisdiction=random.choice(legit_profile["jurisdictions"]),
            frequency_last_hour=random.randint(1, 2),
            is_suspicious=False,
        ))

    for i in range(n_fraud):
        transactions.append(Transaction(
            txn_id=_make_txn_id(wave, 100 + i + 1),
            amount=round(random.uniform(*profile["amount_range"]), 2),
            merchant=random.choice(profile["merchants"]),
            category=random.choice(profile["categories"]),
            hour=random.choice(profile["hours"]),
            region=random.choice(profile["regions"]),
            jurisdiction=random.choice(profile["jurisdictions"]),
            frequency_last_hour=random.randint(*profile["freq_range"]),
            is_suspicious=True,
        ))

    random.shuffle(transactions)
    return transactions


def format_transactions(transactions: List[Transaction]) -> str:
    return "\n".join(str(t) for t in transactions)


def get_suspicious_ids(transactions: List[Transaction]) -> str:
    return ", ".join(t.txn_id for t in transactions)


def get_hacker_history_text(up_to_wave: int) -> str:
    lines = []
    for w in range(1, up_to_wave):
        if w in HACKER_HISTORY:
            lines.append(f"  - {HACKER_HISTORY[w]}")
    return "\n".join(lines) if lines else "No prior wave history available."
