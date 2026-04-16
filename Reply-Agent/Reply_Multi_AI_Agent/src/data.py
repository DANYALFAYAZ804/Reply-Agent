import os
import csv
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


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
    balance_after: float
    description: str
    timestamp: str

    def to_text(self) -> str:
        parts = [
            f"TXN_ID:{self.transaction_id}",
            f"sender:{self.sender_id}",
            f"recipient:{self.recipient_id}",
            f"type:{self.transaction_type}",
            f"amount:{self.amount}",
            f"method:{self.payment_method}",
            f"balance_after:{self.balance_after}",
            f"time:{self.timestamp}",
        ]
        if self.location:
            parts.append(f"location:{self.location}")
        if self.sender_iban:
            parts.append(f"sender_iban:{self.sender_iban}")
        if self.recipient_iban:
            parts.append(f"recipient_iban:{self.recipient_iban}")
        if self.description:
            parts.append(f"desc:{self.description}")
        return " | ".join(parts)


@dataclass
class Location:
    bio_tag: str
    datetime: str
    lat: str
    lng: str
    city: str = ""

    def to_text(self) -> str:
        return (
            f"BioTag:{self.bio_tag} | time:{self.datetime}"
            f" | lat:{self.lat} | lng:{self.lng}"
            + (f" | city:{self.city}" if self.city else "")
        )


@dataclass
class User:
    raw: Dict[str, Any]

    def to_text(self) -> str:
        parts = []
        for k, v in self.raw.items():
            if k == "description":
                parts.append(f"bio:{str(v)[:200]}")
            elif k == "residence" and isinstance(v, dict):
                parts.append(f"city:{v.get('city','')} lat:{v.get('lat','')} lng:{v.get('lng','')}")
            elif v:
                parts.append(f"{k}:{v}")
        return " | ".join(parts)


@dataclass
class Conversation:
    sms: str

    def to_text(self) -> str:
        return f"SMS:{self.sms[:400]}"


@dataclass
class Message:
    mail: str

    def to_text(self) -> str:
        text = self.mail
        for tag in ["<html>", "<head>", "<body>", "<style>", "<script>"]:
            if tag in text.lower():
                import re
                text = re.sub(r"<[^>]+>", " ", text)
                text = " ".join(text.split())
                break
        return f"MAIL:{text[:400]}"


@dataclass
class LevelDataset:
    level: int
    name: str = ""
    transactions: List[Transaction] = field(default_factory=list)
    locations: List[Location] = field(default_factory=list)
    users: List[User] = field(default_factory=list)
    conversations: List[Conversation] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)


def _safe_float(val: str) -> float:
    try:
        return float(val) if val and val.strip() else 0.0
    except ValueError:
        return 0.0


def load_transactions(path: str) -> List[Transaction]:
    txns = []
    if not os.path.exists(path):
        return txns
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            txn_id = (
                row.get("transaction_id")
                or row.get("Transaction ID")
                or ""
            ).strip()
            if not txn_id:
                continue
            txns.append(Transaction(
                transaction_id=txn_id,
                sender_id=(row.get("sender_id") or row.get("Sender ID") or "").strip(),
                recipient_id=(row.get("recipient_id") or row.get("Recipient ID") or "").strip(),
                transaction_type=(row.get("transaction_type") or row.get("Transaction Type") or "").strip(),
                amount=_safe_float(row.get("amount") or row.get("Amount") or ""),
                location=(row.get("location") or row.get("Location") or "").strip(),
                payment_method=(row.get("payment_method") or row.get("Payment Method") or "").strip(),
                sender_iban=(row.get("sender_iban") or row.get("Sender IBAN") or "").strip(),
                recipient_iban=(row.get("recipient_iban") or row.get("Recipient IBAN") or "").strip(),
                balance_after=_safe_float(row.get("balance_after") or row.get("Balance") or row.get("balance") or ""),
                description=(row.get("description") or row.get("Description") or "").strip(),
                timestamp=(row.get("timestamp") or row.get("Timestamp") or "").strip(),
            ))
    return txns


def load_locations_json(path: str) -> List[Location]:
    locs = []
    if not os.path.exists(path):
        return locs
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        locs.append(Location(
            bio_tag=str(item.get("biotag") or item.get("BioTag") or item.get("bio_tag") or ""),
            datetime=str(item.get("timestamp") or item.get("Datetime") or item.get("datetime") or ""),
            lat=str(item.get("lat") or item.get("Lat") or ""),
            lng=str(item.get("lng") or item.get("Lng") or ""),
            city=str(item.get("city") or ""),
        ))
    return locs


def load_users_json(path: str) -> List[User]:
    users = []
    if not os.path.exists(path):
        return users
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        users.append(User(raw=item))
    return users


def load_sms_json(path: str) -> List[Conversation]:
    convs = []
    if not os.path.exists(path):
        return convs
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        text = item.get("sms") or item.get("SMS") or ""
        convs.append(Conversation(sms=str(text)))
    return convs


def load_mails_json(path: str) -> List[Message]:
    msgs = []
    if not os.path.exists(path):
        return msgs
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        text = item.get("mail") or item.get("Mail") or ""
        msgs.append(Message(mail=str(text)))
    return msgs


def load_level_dataset(data_dir: str, level: int, name: str = "") -> LevelDataset:
    level_dir = os.path.join(data_dir, f"level_{level}")
    ds = LevelDataset(level=level, name=name)
    ds.transactions = load_transactions(os.path.join(level_dir, "transactions.csv"))
    if not ds.transactions:
        ds.transactions = load_transactions(os.path.join(level_dir, "Transactions.csv"))
    ds.locations = load_locations_json(os.path.join(level_dir, "locations.json"))
    ds.users = load_users_json(os.path.join(level_dir, "users.json"))
    ds.conversations = load_sms_json(os.path.join(level_dir, "sms.json"))
    ds.messages = load_mails_json(os.path.join(level_dir, "mails.json"))
    return ds


def format_transactions_block(transactions: List[Transaction], max_rows: int = 80) -> str:
    rows = transactions[:max_rows]
    lines = [t.to_text() for t in rows]
    if len(transactions) > max_rows:
        lines.append(f"... ({len(transactions) - max_rows} more transactions not shown)")
    return "\n".join(lines)


def format_locations_block(locations: List[Location], max_rows: int = 30) -> str:
    if not locations:
        return "(no location data)"
    return "\n".join(l.to_text() for l in locations[:max_rows])


def format_users_block(users: List[User], max_rows: int = 20) -> str:
    if not users:
        return "(no user data)"
    return "\n".join(u.to_text() for u in users[:max_rows])


def format_conversations_block(conversations: List[Conversation], max_rows: int = 15) -> str:
    if not conversations:
        return "(no SMS data)"
    return "\n".join(c.to_text() for c in conversations[:max_rows])


def format_messages_block(messages: List[Message], max_rows: int = 10) -> str:
    if not messages:
        return "(no email data)"
    return "\n".join(m.to_text() for m in messages[:max_rows])


def get_all_txn_ids(transactions: List[Transaction]) -> List[str]:
    return [t.transaction_id for t in transactions]
