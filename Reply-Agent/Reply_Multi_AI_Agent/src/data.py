import os
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Transaction:
    transaction_id: str
    sender_id: str
    recipient_id: str
    transaction_type: str
    amount: float
    location: str
    payment_method: str
    sender_iban: str
    recipient_iban: str
    balance: float
    timestamp: str

    def to_text(self) -> str:
        parts = [
            f"TXN_ID:{self.transaction_id}",
            f"sender:{self.sender_id}",
            f"recipient:{self.recipient_id}",
            f"type:{self.transaction_type}",
            f"amount:{self.amount}",
            f"method:{self.payment_method}",
            f"balance:{self.balance}",
            f"time:{self.timestamp}",
        ]
        if self.location:
            parts.append(f"location:{self.location}")
        if self.sender_iban:
            parts.append(f"sender_iban:{self.sender_iban}")
        if self.recipient_iban:
            parts.append(f"recipient_iban:{self.recipient_iban}")
        return " | ".join(parts)


@dataclass
class Location:
    bio_tag: str
    datetime: str
    lat: str
    lng: str

    def to_text(self) -> str:
        return f"BioTag:{self.bio_tag} | time:{self.datetime} | lat:{self.lat} | lng:{self.lng}"


@dataclass
class User:
    raw: Dict[str, str]

    def to_text(self) -> str:
        return " | ".join(f"{k}:{v}" for k, v in self.raw.items() if v)


@dataclass
class Conversation:
    user_id: str
    sms: str

    def to_text(self) -> str:
        return f"UserID:{self.user_id} | SMS:{self.sms[:300]}"


@dataclass
class Message:
    mail: str

    def to_text(self) -> str:
        return f"MAIL:{self.mail[:300]}"


@dataclass
class LevelDataset:
    level: int
    transactions: List[Transaction] = field(default_factory=list)
    locations: List[Location] = field(default_factory=list)
    users: List[User] = field(default_factory=list)
    conversations: List[Conversation] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)


def _open(path: str):
    return open(path, newline="", encoding="utf-8-sig")


def load_transactions(path: str) -> List[Transaction]:
    txns = []
    if not os.path.exists(path):
        return txns
    with _open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not row[0].strip():
                continue
            while len(row) < 12:
                row.append("")
            try:
                amount = float(row[4]) if row[4].strip() else 0.0
            except ValueError:
                amount = 0.0
            try:
                balance = float(row[9]) if row[9].strip() else 0.0
            except ValueError:
                balance = 0.0
            txns.append(Transaction(
                transaction_id=row[0].strip(),
                sender_id=row[1].strip(),
                recipient_id=row[2].strip(),
                transaction_type=row[3].strip(),
                amount=amount,
                location=row[5].strip(),
                payment_method=row[6].strip(),
                sender_iban=row[7].strip(),
                recipient_iban=row[8].strip(),
                balance=balance,
                timestamp=row[11].strip() if len(row) > 11 else "",
            ))
    return txns


def load_locations(path: str) -> List[Location]:
    locs = []
    if not os.path.exists(path):
        return locs
    with _open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            locs.append(Location(
                bio_tag=row.get("BioTag", row.get("bio_tag", "")).strip(),
                datetime=row.get("Datetime", row.get("datetime", "")).strip(),
                lat=row.get("Lat", row.get("lat", "")).strip(),
                lng=row.get("Lng", row.get("lng", "")).strip(),
            ))
    return locs


def load_users(path: str) -> List[User]:
    users = []
    if not os.path.exists(path):
        return users
    with _open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            users.append(User(raw=dict(row)))
    return users


def load_conversations(path: str) -> List[Conversation]:
    convs = []
    if not os.path.exists(path):
        return convs
    with _open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            convs.append(Conversation(
                user_id=row.get("User ID", row.get("user_id", "")).strip(),
                sms=row.get("SMS", row.get("sms", "")).strip(),
            ))
    return convs


def load_messages(path: str) -> List[Message]:
    msgs = []
    if not os.path.exists(path):
        return msgs
    with _open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            msgs.append(Message(mail=row.get("mail", row.get("Mail", "")).strip()))
    return msgs


def load_level_dataset(data_dir: str, level: int) -> LevelDataset:
    level_dir = os.path.join(data_dir, f"level_{level}")
    ds = LevelDataset(level=level)
    ds.transactions = load_transactions(os.path.join(level_dir, "Transactions.csv"))
    ds.locations = load_locations(os.path.join(level_dir, "Locations.csv"))
    ds.users = load_users(os.path.join(level_dir, "Users.csv"))
    ds.conversations = load_conversations(os.path.join(level_dir, "Conversations.csv"))
    ds.messages = load_messages(os.path.join(level_dir, "Messages.csv"))
    return ds


def format_transactions_block(transactions: List[Transaction], max_rows: int = 80) -> str:
    rows = transactions[:max_rows]
    lines = [t.to_text() for t in rows]
    if len(transactions) > max_rows:
        lines.append(f"... ({len(transactions) - max_rows} more transactions truncated)")
    return "\n".join(lines)


def format_locations_block(locations: List[Location], max_rows: int = 30) -> str:
    if not locations:
        return "(no location data)"
    rows = locations[:max_rows]
    return "\n".join(l.to_text() for l in rows)


def format_users_block(users: List[User], max_rows: int = 20) -> str:
    if not users:
        return "(no user data)"
    return "\n".join(u.to_text() for u in users[:max_rows])


def format_conversations_block(conversations: List[Conversation], max_rows: int = 10) -> str:
    if not conversations:
        return "(no conversation data)"
    return "\n".join(c.to_text() for c in conversations[:max_rows])


def format_messages_block(messages: List[Message], max_rows: int = 10) -> str:
    if not messages:
        return "(no message data)"
    return "\n".join(m.to_text() for m in messages[:max_rows])


def get_all_txn_ids(transactions: List[Transaction]) -> List[str]:
    return [t.transaction_id for t in transactions]
