import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import ulid
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler

PHISHING_PATTERNS = [
    r"urgent",
    r"verify(?:\s+now)?",
    r"suspend(?:ed|sion)?",
    r"locked|lock(?:ed|out)?",
    r"unusual\s+login|suspicious\s+(?:login|sign-?in|activity)",
    r"restore\s+access|avoid\s+.*(?:suspend|lock)",
    r"security",
    r"immediate(?:\s+action)?",
    r"pay\s+now",
    r"update\s+payment",
    r"amaz0n",
    r"paypa1",
    r"netfl1x",
]

TRUSTED_DOMAIN_HINTS = {
    "amazon.com",
    "amazon.co.uk",
    "amazon.de",
    "paypal.com",
    "google.com",
    "spotify.com",
    "edf.fr",
    "gov.uk",
    "deutschebank.com",
    "barclays.co.uk",
    "dhl.com",
    "fedex.com",
    "zoom.us",
}

ECOMMERCE_METHODS = {"paypal", "googlepay", "googlepay", "google_pay", "google pay"}


@dataclass
class TxScore:
    transaction_id: str
    sender_id: str
    timestamp: datetime
    risk_score: float
    reasons: list[str]
    payload: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reply Mirror adaptive fraud detector.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset folder.")
    parser.add_argument("--output", type=Path, required=True, help="Output txt path.")
    parser.add_argument(
        "--verify-output",
        type=Path,
        default=None,
        help="Verify an existing output txt against --dataset transactions and exit.",
    )

    parser.add_argument("--model", type=str, default="meta-llama/llama-3.1-8b-instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM review stage.")

    parser.add_argument("--risk-threshold", type=float, default=None, help="Optional hard threshold.")
    parser.add_argument(
        "--target-flag-rate",
        type=float,
        default=None,
        help="Target output ratio in [0,1]. If omitted, auto by dataset size.",
    )
    parser.add_argument("--min-flagged", type=int, default=3, help="Minimum number of flagged transactions.")
    parser.add_argument(
        "--max-flag-rate",
        type=float,
        default=0.35,
        help="Hard cap ratio for flagged transactions.",
    )

    parser.add_argument("--llm-review", action="store_true", help="Enable LLM add-on review.")
    parser.add_argument("--llm-review-window", type=int, default=150)
    parser.add_argument("--llm-max-add", type=int, default=50)

    return parser.parse_args()


def parse_timestamp(value: str) -> datetime:
    raw = str(value).strip()
    if raw.endswith("Z"):
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    else:
        dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_date_from_sms_record(text: str) -> datetime | None:
    match = re.search(r"Date:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", text)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_date_from_mail_record(text: str) -> datetime | None:
    match = re.search(r"^Date:\s*(.+)$", text, flags=re.MULTILINE)
    if not match:
        return None
    try:
        dt = parsedate_to_datetime(match.group(1).strip())
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def generate_session_id() -> str:
    team = os.getenv("TEAM_NAME", "team").strip().replace(" ", "-")
    return f"{team}-{ulid.new().str}"


def normalize_method(method: str) -> str:
    lowered = str(method or "").strip().lower()
    return lowered.replace("-", "").replace(" ", "")


def phishing_score(text: str) -> float:
    lowered = str(text).lower()
    score = 0.0
    keyword_hits = 0

    for pattern in PHISHING_PATTERNS:
        if re.search(pattern, lowered):
            keyword_hits += 1
            score += 1.0

    url_hits = re.findall(r"https?://[^\s\"'>]+", lowered)
    for url in url_hits:
        host = urlparse(url).netloc.lower().split(":", 1)[0]
        if "amaz0n" in host or "paypa1" in host or "netfl1x" in host:
            score += 1.5
            continue
        if host.startswith("bit.ly") or host.startswith("tinyurl"):
            score += 0.25
            continue
        if any(host == trusted or host.endswith(f".{trusted}") for trusted in TRUSTED_DOMAIN_HINTS):
            score += 0.05
        elif re.search(r"\.(?:xyz|top|click|live|biz|ru|cn)\b", host):
            score += 0.9
        else:
            score += 0.3

    if keyword_hits == 0 and not url_hits:
        return 0.0
    return max(0.0, min(score, 7.0))


class FraudPipeline:
    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self.transactions: list[dict[str, Any]] = []
        self.users: list[dict[str, Any]] = []
        self.locations: list[dict[str, Any]] = []
        self.sms: list[dict[str, Any]] = []
        self.mails: list[dict[str, Any]] = []

        self.sender_profile: dict[str, dict[str, Any]] = {}
        self.cities_by_sender: dict[str, set[str]] = defaultdict(set)
        self.message_events_by_sender: dict[str, list[tuple[datetime, float]]] = defaultdict(list)

    def load(self) -> None:
        required = [
            self.dataset_dir / "transactions.csv",
            self.dataset_dir / "users.json",
            self.dataset_dir / "locations.json",
            self.dataset_dir / "sms.json",
            self.dataset_dir / "mails.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing required files: {missing}")

        with (self.dataset_dir / "transactions.csv").open("r", encoding="utf-8", newline="") as f:
            self.transactions = list(csv.DictReader(f))

        self.users = json.loads((self.dataset_dir / "users.json").read_text(encoding="utf-8"))
        self.locations = json.loads((self.dataset_dir / "locations.json").read_text(encoding="utf-8"))
        self.sms = json.loads((self.dataset_dir / "sms.json").read_text(encoding="utf-8"))
        self.mails = json.loads((self.dataset_dir / "mails.json").read_text(encoding="utf-8"))

        self._build_sender_profile_map()
        self._build_city_map()
        self._build_message_events()

    def _build_sender_profile_map(self) -> None:
        sender_by_iban: dict[str, str] = {}
        for tx in self.transactions:
            sender_id = str(tx.get("sender_id", "")).strip()
            sender_iban = str(tx.get("sender_iban", "")).strip()
            if sender_id and sender_iban and sender_iban not in sender_by_iban:
                sender_by_iban[sender_iban] = sender_id

        self.sender_profile = {}
        for user in self.users:
            iban = str(user.get("iban", "")).strip()
            sender_id = sender_by_iban.get(iban)
            if sender_id:
                self.sender_profile[sender_id] = user

    def _build_city_map(self) -> None:
        self.cities_by_sender = defaultdict(set)
        for rec in self.locations:
            sender_id = str(rec.get("biotag", "")).strip()
            city = str(rec.get("city", "")).strip()
            if sender_id and city:
                self.cities_by_sender[sender_id].add(city)

    def _build_message_events(self) -> None:
        events: dict[str, list[tuple[datetime, float]]] = defaultdict(list)

        token_to_senders: dict[str, set[str]] = defaultdict(set)
        for sender_id, user in self.sender_profile.items():
            first = str(user.get("first_name", "")).strip().lower()
            last = str(user.get("last_name", "")).strip().lower()
            if first:
                token_to_senders[first].add(sender_id)
            if last:
                token_to_senders[last].add(sender_id)

        for row in self.sms:
            text = str(row.get("sms", ""))
            score = phishing_score(text)
            if score < 1.1:
                continue
            event_time = parse_date_from_sms_record(text)
            if event_time is None:
                continue
            lowered = text.lower()
            matched: set[str] = set()
            for token, senders in token_to_senders.items():
                if token and re.search(rf"\b{re.escape(token)}\b", lowered):
                    matched.update(senders)
            for sender_id in matched:
                events[sender_id].append((event_time, score))

        for row in self.mails:
            text = str(row.get("mail", ""))
            score = phishing_score(text)
            if score < 1.1:
                continue
            event_time = parse_date_from_mail_record(text)
            if event_time is None:
                continue
            lowered = text.lower()
            matched: set[str] = set()
            for token, senders in token_to_senders.items():
                if token and re.search(rf"\b{re.escape(token)}\b", lowered):
                    matched.update(senders)
            for sender_id in matched:
                events[sender_id].append((event_time, score))

        self.message_events_by_sender = defaultdict(list)
        for sender_id, values in events.items():
            values.sort(key=lambda item: item[0])
            self.message_events_by_sender[sender_id] = values

    def _recent_message_risk(self, sender_id: str, timestamp: datetime, lookback_days: int = 45) -> float:
        values = self.message_events_by_sender.get(sender_id, [])
        if not values:
            return 0.0

        total = 0.0
        for event_time, score in reversed(values):
            if event_time > timestamp:
                continue
            delta_days = (timestamp - event_time).total_seconds() / 86400.0
            if delta_days > lookback_days:
                break
            decay = math.exp(-delta_days / 14.0)
            total += score * decay
        return min(total, 9.0)

    def score_transactions(self) -> list[TxScore]:
        tx_rows = sorted(self.transactions, key=lambda row: parse_timestamp(str(row.get("timestamp", ""))))

        amount_history: dict[str, list[float]] = defaultdict(list)
        hour_history: dict[str, Counter[int]] = defaultdict(Counter)
        recipient_seen: dict[str, set[str]] = defaultdict(set)
        method_seen: dict[str, set[str]] = defaultdict(set)
        type_seen: dict[str, set[str]] = defaultdict(set)
        sender_recent: dict[str, deque[tuple[datetime, float]]] = defaultdict(deque)

        scores: list[TxScore] = []

        for row in tx_rows:
            tx_id = str(row.get("transaction_id", "")).strip()
            sender_id = str(row.get("sender_id", "")).strip()
            recipient_id = str(row.get("recipient_id", "")).strip()
            tx_type = str(row.get("transaction_type", "")).strip().lower()
            method_raw = str(row.get("payment_method", "")).strip().lower()
            method = normalize_method(method_raw)
            timestamp = parse_timestamp(str(row.get("timestamp", "")))
            amount = safe_float(row.get("amount"), 0.0)
            balance_after = safe_float(row.get("balance_after"), 0.0)
            description = str(row.get("description", ""))
            location = str(row.get("location", "")).strip()

            sender_amounts = amount_history[sender_id]
            amount_z = 0.0
            if len(sender_amounts) >= 8:
                mean = statistics.mean(sender_amounts)
                stdev = statistics.pstdev(sender_amounts)
                if stdev > 1e-6:
                    amount_z = (amount - mean) / stdev
                    amount_z = max(amount_z, 0.0)

            sender_hours = hour_history[sender_id]
            hour = timestamp.hour
            hour_prob = sender_hours[hour] / max(1, sum(sender_hours.values()))
            unusual_hour = len(sender_amounts) >= 25 and hour_prob < 0.025

            novelty_ready = len(sender_amounts) >= 6
            new_recipient = novelty_ready and bool(recipient_id) and recipient_id not in recipient_seen[sender_id]
            new_method = novelty_ready and bool(method) and method not in method_seen[sender_id]
            new_type = novelty_ready and bool(tx_type) and tx_type not in type_seen[sender_id]

            comm_risk = self._recent_message_risk(sender_id, timestamp)

            # 24h velocity spike
            window = sender_recent[sender_id]
            while window and (timestamp - window[0][0]).total_seconds() > 86400:
                window.popleft()
            recent_amount_total = sum(v for _, v in window)
            recent_count = len(window)

            velocity_spike = False
            if recent_count >= 4 and sender_amounts:
                median_amount = statistics.median(sender_amounts)
                if amount > max(2.5 * median_amount, 150.0) and recent_amount_total > 4.0 * max(median_amount, 1.0):
                    velocity_spike = True

            user = self.sender_profile.get(sender_id, {})
            residence_city = str(user.get("residence", {}).get("city", "")).strip()
            tx_city = location.split(" - ", 1)[0].strip() if " - " in location else location
            known_cities = self.cities_by_sender.get(sender_id, set())
            city_mismatch = tx_type == "in-person payment" and bool(tx_city) and tx_city not in known_cities and tx_city != residence_city

            reasons: list[str] = []
            risk = 0.0

            if amount_z >= 4.0:
                risk += 2.0
                reasons.append(f"strong amount outlier z={amount_z:.2f}")
            elif amount_z >= 2.6:
                risk += 1.3
                reasons.append(f"amount outlier z={amount_z:.2f}")
            elif amount_z >= 1.9:
                risk += 0.8
                reasons.append(f"amount deviation z={amount_z:.2f}")

            if new_recipient:
                risk += 0.9
                reasons.append("new recipient")
            if new_method:
                risk += 0.6
                reasons.append("new payment method")
            if new_type:
                risk += 0.5
                reasons.append("new transaction type")
            if unusual_hour:
                risk += 0.9
                reasons.append("unusual hour")
            if hour <= 5:
                risk += 0.35
                reasons.append("night transaction")
            if city_mismatch:
                risk += 1.1
                reasons.append("city mismatch")
            if velocity_spike:
                risk += 1.0
                reasons.append("24h spending velocity spike")

            if comm_risk > 2.8:
                comm_bonus = min(1.3, 0.20 * comm_risk)
                risk += comm_bonus
                reasons.append(f"recent phishing exposure={comm_risk:.2f}")

            if balance_after < 100 and amount > 350:
                risk += 0.7
                reasons.append("large debit leaves low balance")

            desc_phishing = phishing_score(description)
            if desc_phishing > 1.2:
                risk += 0.5
                reasons.append("description contains phishing-like wording")

            ecommerce_method = method in {"paypal", "googlepay"}
            if tx_type == "e-commerce" and ecommerce_method and new_recipient and new_method:
                risk += 0.8
                reasons.append("new ecommerce payee pattern")
            if tx_type == "e-commerce" and ecommerce_method and new_recipient and comm_risk > 2.2:
                risk += 0.7
                reasons.append("new ecommerce recipient near phishing exposure")
            if tx_type == "e-commerce" and ecommerce_method and not description.strip():
                risk += 0.5
                reasons.append("silent ecommerce description")

            if tx_type == "transfer" and not recipient_id and not new_method and not new_type:
                risk -= 0.35
                reasons.append("plain recurring transfer")

            if tx_type == "in-person payment" and amount <= 20:
                risk -= 0.4
                reasons.append("small in-person spend")

            risk = max(0.0, risk)

            payload = {
                "transaction_id": tx_id,
                "sender_id": sender_id,
                "recipient_id": recipient_id,
                "transaction_type": tx_type,
                "payment_method": method_raw,
                "amount": amount,
                "balance_after": balance_after,
                "timestamp": row.get("timestamp"),
                "location": location,
                "risk_score": round(risk, 4),
                "amount_z": round(amount_z, 4),
                "new_recipient": new_recipient,
                "new_method": new_method,
                "new_type": new_type,
                "unusual_hour": unusual_hour,
                "velocity_spike": velocity_spike,
                "communication_risk": round(comm_risk, 4),
                "reasons": reasons,
            }

            scores.append(
                TxScore(
                    transaction_id=tx_id,
                    sender_id=sender_id,
                    timestamp=timestamp,
                    risk_score=risk,
                    reasons=reasons,
                    payload=payload,
                )
            )

            amount_history[sender_id].append(amount)
            hour_history[sender_id][hour] += 1
            if recipient_id:
                recipient_seen[sender_id].add(recipient_id)
            if method:
                method_seen[sender_id].add(method)
            if tx_type:
                type_seen[sender_id].add(tx_type)
            window.append((timestamp, amount))

        return scores


def auto_target_rate(total_transactions: int) -> float:
    if total_transactions <= 300:
        return 0.06
    if total_transactions <= 2000:
        return 0.09
    return 0.11


def clamp_output_size(total: int, target_rate: float, min_flagged: int, max_flag_rate: float) -> tuple[int, int]:
    min_count = max(1, min(min_flagged, total - 1))
    target_count = max(min_count, int(round(total * target_rate)))
    max_count = max(min_count, int(round(total * max_flag_rate)))
    max_count = min(max_count, total - 1)
    target_count = min(max(target_count, min_count), max_count)
    return target_count, max_count


def select_base_candidates(
    scores: list[TxScore],
    risk_threshold: float | None,
    target_count: int,
    max_count: int,
) -> tuple[list[TxScore], list[TxScore]]:
    ranked = sorted(scores, key=lambda item: item.risk_score, reverse=True)

    if risk_threshold is not None:
        selected = [item for item in ranked if item.risk_score >= risk_threshold]
    else:
        selected = ranked[:target_count]

    if len(selected) < target_count:
        selected = ranked[:target_count]

    if len(selected) > max_count:
        selected = selected[:max_count]

    selected_ids = {item.transaction_id for item in selected}
    border_pool = [item for item in ranked if item.transaction_id not in selected_ids][:300]
    return selected, border_pool


@observe()
def llm_additional_review(
    session_id: str,
    model: ChatOpenAI,
    selected: list[TxScore],
    border_pool: list[TxScore],
    max_add: int,
) -> list[str]:
    if not border_pool or max_add <= 0:
        return []

    handler = CallbackHandler()
    compact_selected = [
        {
            "transaction_id": item.transaction_id,
            "risk_score": item.payload["risk_score"],
            "type": item.payload["transaction_type"],
            "method": item.payload["payment_method"],
            "signals": item.reasons[:3],
        }
        for item in selected[:40]
    ]

    compact_border = [
        {
            "transaction_id": item.transaction_id,
            "risk_score": item.payload["risk_score"],
            "type": item.payload["transaction_type"],
            "method": item.payload["payment_method"],
            "signals": item.reasons[:4],
        }
        for item in border_pool[:180]
    ]

    prompt = f"""
You are a fraud investigator assistant.

Goal:
- Add only high-value suspicious transactions from BORDERLINE_CANDIDATES.
- Be conservative on false positives.
- Return JSON only.

Return schema:
{{
  "extra_flagged_transaction_ids": ["id1", "id2"]
}}

Constraints:
- IDs must come only from BORDERLINE_CANDIDATES.
- Return at most {max_add} IDs.

ALREADY_SELECTED (high confidence baseline):
{json.dumps(compact_selected, indent=2)}

BORDERLINE_CANDIDATES:
{json.dumps(compact_border, indent=2)}
""".strip()

    response = model.invoke(
        [HumanMessage(content=prompt)],
        config={
            "callbacks": [handler],
            "metadata": {"langfuse_session_id": session_id, "stage": "llm_additional_review"},
        },
    )

    content = response.content if isinstance(response.content, str) else str(response.content)
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content, flags=re.IGNORECASE)
    if fence_match:
        content = fence_match.group(1).strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return []

    values = parsed.get("extra_flagged_transaction_ids", []) if isinstance(parsed, dict) else []
    if not isinstance(values, list):
        return []

    allowed = {item.transaction_id for item in border_pool}
    cleaned = [str(v).strip() for v in values if str(v).strip() in allowed]

    unique_ids: list[str] = []
    seen: set[str] = set()
    for tx_id in cleaned:
        if tx_id not in seen:
            unique_ids.append(tx_id)
            seen.add(tx_id)
        if len(unique_ids) >= max_add:
            break
    return unique_ids


@observe()
def session_trace_anchor(session_id: str, model: ChatOpenAI) -> str:
    handler = CallbackHandler()
    response = model.invoke(
        [HumanMessage(content="Reply with exactly: ok")],
        config={
            "callbacks": [handler],
            "metadata": {"langfuse_session_id": session_id, "stage": "session_anchor"},
        },
    )
    return response.content if isinstance(response.content, str) else str(response.content)


def ensure_env() -> None:
    load_dotenv(find_dotenv())


def ensure_llm_env() -> None:
    required = ["OPENROUTER_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError("Missing env vars: " + ", ".join(missing))


def ensure_langfuse_auth(client: Langfuse) -> bool:
    try:
        ok = client.auth_check()
        if ok is False:
            print("[WARN] Langfuse auth_check returned False.")
            return False
        return True
    except Exception as exc:
        print(f"[WARN] Langfuse auth_check failed: {exc}")
        return False


def enforce_output_validity(scores: list[TxScore], selected_ids: set[str], min_count: int, max_count: int) -> set[str]:
    ranked = sorted(scores, key=lambda item: item.risk_score, reverse=True)
    total = len(ranked)
    if total == 0:
        return set()

    valid_ids = {item.transaction_id for item in ranked}
    selected_ids = {tx_id for tx_id in selected_ids if tx_id in valid_ids}

    if len(selected_ids) < min_count:
        selected_ids = {item.transaction_id for item in ranked[:min_count]}

    if len(selected_ids) > max_count:
        kept: list[str] = []
        for item in ranked:
            if item.transaction_id in selected_ids:
                kept.append(item.transaction_id)
            if len(kept) >= max_count:
                break
        selected_ids = set(kept)

    if len(selected_ids) >= total:
        selected_ids = {item.transaction_id for item in ranked[: total - 1]}

    return selected_ids


def write_ascii_output(path: Path, ids_sorted: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(f"{tx_id}\n" for tx_id in ids_sorted)
    path.write_bytes(payload.encode("ascii", errors="strict"))


def verify_output_against_dataset(scores: list[TxScore], output_path: Path) -> int:
    if not output_path.exists():
        print(f"[ERROR] File not found: {output_path}")
        return 2

    dataset_ids = {item.transaction_id for item in scores}
    lines = [line.strip() for line in output_path.read_text(encoding="utf-8").splitlines()]
    ids = [line for line in lines if line]
    unique_ids = set(ids)

    missing = sorted(unique_ids - dataset_ids)
    duplicates = len(ids) - len(unique_ids)

    print(f"Verify file: {output_path}")
    print(f"Rows (non-empty): {len(ids)}")
    print(f"Unique IDs: {len(unique_ids)}")
    print(f"Duplicates: {duplicates}")
    print(f"IDs present in dataset: {len(unique_ids) - len(missing)}")
    print(f"IDs missing from dataset: {len(missing)}")

    if missing:
        print("First missing IDs:")
        for tx_id in missing[:10]:
            print(tx_id)
        return 1

    if len(ids) == 0:
        print("[WARN] Output has no IDs.")
        return 1

    if len(unique_ids) >= len(dataset_ids):
        print("[WARN] Output includes all transactions (or more).")
        return 1

    print("[OK] Output structure is coherent with selected dataset.")
    return 0


def main() -> None:
    args = parse_args()
    ensure_env()

    pipeline = FraudPipeline(args.dataset)
    pipeline.load()
    scores = pipeline.score_transactions()

    if args.verify_output is not None:
        code = verify_output_against_dataset(scores, args.verify_output)
        raise SystemExit(code)

    total = len(scores)
    if total == 0:
        raise RuntimeError("No transactions found in dataset.")

    target_rate = args.target_flag_rate if args.target_flag_rate is not None else auto_target_rate(total)
    target_rate = min(max(target_rate, 0.01), 0.95)
    max_flag_rate = min(max(args.max_flag_rate, 0.02), 0.99)

    target_count, max_count = clamp_output_size(
        total=total,
        target_rate=target_rate,
        min_flagged=max(args.min_flagged, 1),
        max_flag_rate=max_flag_rate,
    )

    session_id = generate_session_id()
    print(f"Session ID: {session_id}")
    print(f"Dataset: {args.dataset}")
    print(f"Transactions loaded: {total}")
    print(f"Target flag count: {target_count} (max {max_count})")

    selected, border_pool = select_base_candidates(
        scores=scores,
        risk_threshold=args.risk_threshold,
        target_count=target_count,
        max_count=max_count,
    )

    selected_ids = {item.transaction_id for item in selected}

    if not args.dry_run:
        ensure_llm_env()
        model = ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=args.model,
            temperature=args.temperature,
        )
        langfuse_client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
        )
        ensure_langfuse_auth(langfuse_client)

        if args.llm_review:
            review_window = max(0, args.llm_review_window)
            review_pool = border_pool[:review_window]
            try:
                extra_ids = llm_additional_review(
                    session_id=session_id,
                    model=model,
                    selected=selected,
                    border_pool=review_pool,
                    max_add=max(0, args.llm_max_add),
                )
                selected_ids.update(extra_ids)
                print(f"LLM added transactions: {len(extra_ids)}")
            except Exception as exc:
                print(f"[WARN] LLM review failed: {exc}")
        else:
            try:
                _ = session_trace_anchor(session_id=session_id, model=model)
                print("Trace anchor emitted.")
            except Exception as exc:
                print(f"[WARN] Trace anchor failed: {exc}")

        langfuse_client.flush()

    min_count = max(1, min(args.min_flagged, total - 1))
    selected_ids = enforce_output_validity(
        scores=scores,
        selected_ids=selected_ids,
        min_count=min_count,
        max_count=max_count,
    )

    ranked_by_time = sorted(scores, key=lambda item: item.timestamp)
    output_ids = [item.transaction_id for item in ranked_by_time if item.transaction_id in selected_ids]

    write_ascii_output(args.output, output_ids)

    print("Run complete.")
    print(f"Flagged transactions: {len(output_ids)}")
    print(f"Output file: {args.output.resolve()}")
    print("Use the same session ID above in the submission modal.")


if __name__ == "__main__":
    main()
