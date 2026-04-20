import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
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
    r"socsec",
]

TRUSTED_DOMAIN_HINTS = {
    "audincourt.fr",
    "dietzenbach.de",
    "deutschebank.com",
    "amazon.com",
    "barclays.co.uk",
    "dpd.co.uk",
    "dhl.com",
    "eventbrite.com",
    "zoom.us",
    "calendly.com",
    "figma.com",
    "uber.com",
    "lyft.com",
    "paypal.com",
    "google.com",
    "edf.fr",
}

LIKELY_LEGIT_SENDER_HINTS = [
    "city hall",
    "town hall",
    "council",
    "community center",
    "municipal",
    "events office",
    "city of ",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reply Mirror challenge-day fraud detector.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to one dataset folder.")
    parser.add_argument("--output", type=Path, required=True, help="Output TXT file path.")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/llama-3.1-8b-instruct",
        help="OpenRouter model ID.",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--risk-threshold", type=float, default=2.6)
    parser.add_argument("--max-candidates-per-sender", type=int, default=18)
    parser.add_argument("--max-senders", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM and use heuristic only.")
    return parser.parse_args()


def parse_timestamp(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_city_from_transaction_location(location: str) -> str:
    if not location:
        return ""
    if " - " in location:
        return location.split(" - ", 1)[0].strip()
    return location.strip()


def generate_session_id() -> str:
    team = os.getenv("TEAM_NAME", "team").strip().replace(" ", "-")
    return f"{team}-{ulid.new().str}"


def parse_date_from_sms_record(text: str) -> datetime | None:
    match = re.search(r"Date:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", text)
    if not match:
        return None
    raw = match.group(1)
    try:
        dt = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_date_from_mail_record(text: str) -> datetime | None:
    match = re.search(r"^Date:\s*(.+)$", text, flags=re.MULTILINE)
    if not match:
        return None
    raw = match.group(1).strip()
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def phishing_score(text: str) -> float:
    lowered = text.lower()
    score = 0.0
    keyword_hits = 0

    for pattern in PHISHING_PATTERNS:
        if re.search(pattern, lowered):
            keyword_hits += 1
            score += 1.0

    url_hits = re.findall(r"https?://[^\s\"'>]+", lowered)
    for url in url_hits:
        host = urlparse(url).netloc.lower()
        host = host.split(":", 1)[0]

        if "amaz0n" in host or "paypa1" in host or "netfl1x" in host:
            score += 1.6
            continue
        if host.startswith("bit.ly") or host.startswith("tinyurl"):
            score += 0.15
            continue

        if any(host == trusted or host.endswith(f".{trusted}") for trusted in TRUSTED_DOMAIN_HINTS):
            score += 0.05
        elif re.search(r"\.(?:xyz|top|click|live|biz|ru|cn)\b", host):
            score += 0.9
        else:
            score += 0.25

    if any(hint in lowered for hint in LIKELY_LEGIT_SENDER_HINTS) and keyword_hits <= 1:
        score *= 0.2

    if keyword_hits == 0 and not url_hits:
        return 0.0

    score = min(score, 6.0)
    return max(0.0, score)


@dataclass
class TxFeature:
    transaction_id: str
    sender_id: str
    timestamp: datetime
    amount: float
    risk_score: float
    reasons: list[str]
    payload: dict[str, Any]


class FraudPipeline:
    def __init__(
        self,
        dataset_dir: Path,
        risk_threshold: float,
        max_candidates_per_sender: int,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.risk_threshold = risk_threshold
        self.max_candidates_per_sender = max_candidates_per_sender

        self.users: list[dict[str, Any]] = []
        self.transactions: list[dict[str, Any]] = []
        self.locations: list[dict[str, Any]] = []
        self.sms: list[dict[str, Any]] = []
        self.mails: list[dict[str, Any]] = []

        self.users_by_biotag: dict[str, dict[str, Any]] = {}
        self.message_events_by_user: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
        self.location_cities_by_user: dict[str, set[str]] = defaultdict(set)

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

        self.users_by_biotag = {}

        sender_by_iban: dict[str, str] = {}
        for tx in self.transactions:
            sender_id = str(tx.get("sender_id", "")).strip()
            sender_iban = str(tx.get("sender_iban", "")).strip()
            if sender_id and sender_iban and sender_iban not in sender_by_iban:
                sender_by_iban[sender_iban] = sender_id

        for user in self.users:
            user_iban = str(user.get("iban", "")).strip()
            if not user_iban:
                continue
            sender_id = sender_by_iban.get(user_iban)
            if sender_id:
                self.users_by_biotag[sender_id] = user

        for record in self.locations:
            user_id = record.get("biotag", "")
            city = str(record.get("city", "")).strip()
            if user_id and city:
                self.location_cities_by_user[user_id].add(city)

        self.message_events_by_user = self._build_message_events_by_user()

    def _build_message_events_by_user(self) -> dict[str, list[tuple[datetime, float]]]:
        events: dict[str, list[tuple[datetime, float]]] = defaultdict(list)

        name_to_user_ids: dict[str, list[str]] = defaultdict(list)
        for user_id, user in self.users_by_biotag.items():
            first = str(user.get("first_name", "")).strip().lower()
            last = str(user.get("last_name", "")).strip().lower()
            if first:
                name_to_user_ids[first].append(user_id)
            if last:
                name_to_user_ids[last].append(user_id)

        for row in self.sms:
            text = str(row.get("sms", ""))
            score = phishing_score(text)
            if score <= 0:
                continue
            event_time = parse_date_from_sms_record(text)
            if event_time is None:
                continue
            lowered = text.lower()
            for token, user_ids in name_to_user_ids.items():
                if token and re.search(rf"\b{re.escape(token)}\b", lowered):
                    for user_id in user_ids:
                        events[user_id].append((event_time, score))

        for row in self.mails:
            text = str(row.get("mail", ""))
            score = phishing_score(text)
            if score <= 0:
                continue
            event_time = parse_date_from_mail_record(text)
            if event_time is None:
                continue
            lowered = text.lower()
            for token, user_ids in name_to_user_ids.items():
                if token and re.search(rf"\b{re.escape(token)}\b", lowered):
                    for user_id in user_ids:
                        events[user_id].append((event_time, score))

        for user_id in events:
            events[user_id].sort(key=lambda item: item[0])
        return dict(events)

    def _recent_message_risk(self, sender_id: str, timestamp: datetime, lookback_days: int = 45) -> float:
        events = self.message_events_by_user.get(sender_id, [])
        if not events:
            return 0.0

        total = 0.0
        for event_time, score in reversed(events):
            if event_time > timestamp:
                continue

            delta_days = (timestamp - event_time).total_seconds() / 86400.0
            if delta_days > lookback_days:
                break

            decay = math.exp(-delta_days / 14.0)
            total += score * decay

        return total

    def compute_features(self) -> list[TxFeature]:
        history_amounts: dict[str, list[float]] = defaultdict(list)
        history_hours: dict[str, Counter[int]] = defaultdict(Counter)
        history_recipients: dict[str, set[str]] = defaultdict(set)
        history_methods: dict[str, set[str]] = defaultdict(set)
        history_types: dict[str, set[str]] = defaultdict(set)

        tx_rows = sorted(self.transactions, key=lambda row: parse_timestamp(row["timestamp"]))
        features: list[TxFeature] = []

        for row in tx_rows:
            tx_id = row["transaction_id"]
            sender = row["sender_id"]
            recipient = (row.get("recipient_id") or "").strip()
            tx_type = (row.get("transaction_type") or "").strip().lower()
            method = (row.get("payment_method") or "").strip().lower()
            location = (row.get("location") or "").strip()
            timestamp = parse_timestamp(row["timestamp"])
            amount = safe_float(row.get("amount"), default=0.0)
            balance_after = safe_float(row.get("balance_after"), default=0.0)
            description = str(row.get("description", ""))

            sender_amounts = history_amounts[sender]
            amount_z = 0.0
            if len(sender_amounts) >= 6:
                mean = statistics.mean(sender_amounts)
                stdev = statistics.pstdev(sender_amounts)
                if stdev > 1e-6:
                    amount_z = (amount - mean) / stdev
                    if amount_z < 0:
                        amount_z = 0.0

            sender_hours = history_hours[sender]
            hour = timestamp.hour
            hour_count = sender_hours[hour]
            total_hours = max(1, sum(sender_hours.values()))
            hour_ratio = hour_count / total_hours
            unusual_hour = len(sender_amounts) >= 20 and hour_ratio < 0.03

            novelty_gate = len(sender_amounts) >= 5
            is_new_recipient = novelty_gate and bool(recipient) and recipient not in history_recipients[sender]
            is_new_method = novelty_gate and bool(method) and method not in history_methods[sender]
            is_new_type = novelty_gate and bool(tx_type) and tx_type not in history_types[sender]

            user_profile = self.users_by_biotag.get(sender, {})
            user_job = str(user_profile.get("job", "")).strip().lower()
            residence_city = str(user_profile.get("residence", {}).get("city", "")).strip()
            tx_city = extract_city_from_transaction_location(location)
            known_cities = self.location_cities_by_user.get(sender, set())

            city_mismatch = False
            if tx_type == "in-person payment" and tx_city:
                if tx_city not in known_cities and tx_city != residence_city:
                    city_mismatch = True

            communication_risk = self._recent_message_risk(sender, timestamp)

            risk = 0.0
            reasons: list[str] = []

            if amount_z >= 2.8:
                risk += 1.6
                reasons.append(f"amount outlier z={amount_z:.2f}")
            elif amount_z >= 2.0:
                risk += 1.0
                reasons.append(f"amount deviation z={amount_z:.2f}")

            if is_new_recipient:
                risk += 0.7
                reasons.append("new recipient")
            if is_new_method:
                method_weight = 0.5
                if tx_type == "in-person payment" and amount < 50:
                    method_weight = 0.1
                risk += method_weight
                reasons.append("new payment method")
            if is_new_type:
                type_weight = 0.4
                if tx_type == "in-person payment" and amount < 50:
                    type_weight = 0.1
                risk += type_weight
                reasons.append("new transaction type")
            if unusual_hour:
                risk += 0.8
                reasons.append("unusual hour for sender")
            if hour <= 5:
                risk += 0.3
                reasons.append("night-time transaction")
            if city_mismatch:
                risk += 1.0
                reasons.append("transaction city not seen in user mobility")
            if communication_risk > 1.8:
                bonus = min(1.6, 0.35 * communication_risk)
                risk += bonus
                reasons.append(f"recent phishing exposure score={communication_risk:.2f}")
            if balance_after < 50 and amount > 300:
                risk += 0.5
                reasons.append("large debit into very low remaining balance")
            if phishing_score(description) > 1.2:
                risk += 0.5
                reasons.append("description contains risky wording")
            if (
                tx_type == "e-commerce"
                and method in {"paypal", "googlepay"}
                and is_new_recipient
                and is_new_method
                and is_new_type
            ):
                risk += 0.8
                reasons.append("new online payee and payment pattern break")
            if (
                tx_type == "e-commerce"
                and method in {"paypal", "googlepay"}
                and is_new_recipient
                and is_new_method
                and is_new_type
                and not description.strip()
            ):
                risk += 0.9
                reasons.append("silent paypal/googlepay ecommerce pattern")
            if (
                tx_type == "e-commerce"
                and method in {"paypal", "googlepay"}
                and is_new_recipient
                and communication_risk > 1.0
            ):
                risk += 0.6
                reasons.append("new online payee shortly after phishing-like messages")

            # Context-aware de-risking for common benign spending patterns.
            desc_lower = description.lower()
            if (
                tx_type == "direct debit"
                and "ride-share" in user_job
                and re.search(r"ride.?share|driver app|subscription", desc_lower)
                and amount <= 30
            ):
                risk -= 1.2
                reasons.append("job-consistent low-value driver subscription")

            if (
                tx_type == "in-person payment"
                and amount <= 25
                and "coffee" in location.lower()
                and ("freelance" in user_job or "designer" in user_job)
            ):
                risk -= 1.0
                reasons.append("plausible small local purchase for freelancer routine")

            if risk < 0:
                risk = 0.0

            payload = {
                "transaction_id": tx_id,
                "sender_id": sender,
                "recipient_id": recipient,
                "transaction_type": tx_type,
                "amount": amount,
                "payment_method": method,
                "location": location,
                "city": tx_city,
                "timestamp": row.get("timestamp"),
                "balance_after": balance_after,
                "risk_score": round(risk, 4),
                "amount_z": round(amount_z, 4),
                "is_new_recipient": is_new_recipient,
                "is_new_method": is_new_method,
                "is_new_type": is_new_type,
                "unusual_hour": unusual_hour,
                "communication_risk": round(communication_risk, 4),
                "reasons": reasons,
            }

            features.append(
                TxFeature(
                    transaction_id=tx_id,
                    sender_id=sender,
                    timestamp=timestamp,
                    amount=amount,
                    risk_score=risk,
                    reasons=reasons,
                    payload=payload,
                )
            )

            history_amounts[sender].append(amount)
            history_hours[sender][hour] += 1
            if recipient:
                history_recipients[sender].add(recipient)
            if method:
                history_methods[sender].add(method)
            if tx_type:
                history_types[sender].add(tx_type)

        return features

    def select_candidates(self, features: list[TxFeature]) -> dict[str, list[TxFeature]]:
        by_sender: dict[str, list[TxFeature]] = defaultdict(list)
        for feat in features:
            if feat.risk_score >= self.risk_threshold:
                by_sender[feat.sender_id].append(feat)

        for sender_id in list(by_sender.keys()):
            ranked = sorted(by_sender[sender_id], key=lambda item: item.risk_score, reverse=True)
            by_sender[sender_id] = ranked[: self.max_candidates_per_sender]
        return by_sender


@observe()
def llm_review_sender_candidates(
    session_id: str,
    model: ChatOpenAI,
    sender_id: str,
    sender_profile: dict[str, Any],
    candidates: list[TxFeature],
) -> list[str]:
    handler = CallbackHandler()

    compact_candidates = [
        {
            "transaction_id": c.transaction_id,
            "timestamp": c.payload["timestamp"],
            "transaction_type": c.payload["transaction_type"],
            "amount": c.payload["amount"],
            "payment_method": c.payload["payment_method"],
            "location": c.payload["location"],
            "risk_score": c.payload["risk_score"],
            "signals": c.payload["reasons"][:4],
        }
        for c in candidates
    ]

    prompt = f"""
You are the final fraud decision agent for Reply Mirror.

Your task:
- Review candidate transactions for one sender.
- Return ONLY transactions that are highly likely fraudulent.
- Be conservative on false positives.

Return valid JSON only with this schema:
{{
  "flagged_transaction_ids": ["id1", "id2"],
  "notes": ["short rationale"]
}}

Sender: {sender_id}
Profile:
{json.dumps(sender_profile, indent=2)}

Candidates:
{json.dumps(compact_candidates, indent=2)}

Rules:
- Prefer flagging when multiple risk signals align.
- A single weak signal is not enough.
- Keep output IDs strictly from candidate list.
""".strip()

    response = model.invoke(
        [HumanMessage(content=prompt)],
        config={
            "callbacks": [handler],
            "metadata": {"langfuse_session_id": session_id, "sender_id": sender_id},
        },
    )

    content = response.content
    if not isinstance(content, str):
        content = str(content)

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content, flags=re.IGNORECASE)
    if fence_match:
        content = fence_match.group(1).strip()

    flagged: list[str] = []
    try:
        parsed = json.loads(content)
        values = parsed.get("flagged_transaction_ids", []) if isinstance(parsed, dict) else []
        if isinstance(values, list):
            flagged = [str(item).strip() for item in values if str(item).strip()]
    except json.JSONDecodeError:
        flagged = []

    allowed = {c.transaction_id for c in candidates}
    filtered = [tx_id for tx_id in flagged if tx_id in allowed]
    return filtered


def enforce_valid_output(all_features: list[TxFeature], flagged_ids: set[str]) -> set[str]:
    total = len(all_features)
    if total == 0:
        return set()

    if len(flagged_ids) == 0:
        ranked = sorted(all_features, key=lambda item: item.risk_score, reverse=True)
        fallback_count = max(1, total // 200)
        flagged_ids = {item.transaction_id for item in ranked[:fallback_count]}

    if len(flagged_ids) >= total:
        ranked = sorted(all_features, key=lambda item: item.risk_score, reverse=True)
        keep = max(1, total - 1)
        flagged_ids = {item.transaction_id for item in ranked[:keep]}

    return flagged_ids


def write_ascii_output(path: Path, tx_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(f"{tx_id}\n" for tx_id in tx_ids)
    path.write_bytes(payload.encode("ascii", errors="strict"))


def ensure_env() -> None:
    load_dotenv(find_dotenv())


def ensure_llm_env() -> None:
    required = ["OPENROUTER_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Missing env vars: "
            + ", ".join(missing)
            + ". Fill them in repo root .env or run with --dry-run."
        )


def ensure_langfuse_auth(langfuse_client: Langfuse) -> bool:
    strict = str(os.getenv("LANGFUSE_STRICT_AUTH", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    try:
        ok = langfuse_client.auth_check()
        if ok is False:
            if strict:
                raise RuntimeError("Langfuse auth_check returned False")
            print("[WARN] Langfuse auth_check returned False; continuing because strict auth is disabled.")
            return False
        return True
    except Exception as exc:
        if strict:
            raise RuntimeError(
                "Langfuse authentication failed. The generated session ID will not be valid for submission. "
                "Check LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY in root .env and retry. "
                f"Original error: {exc}"
            ) from exc
        print(
            "[WARN] Langfuse auth_check failed but continuing (strict auth disabled). "
            f"Reason: {exc}"
        )
        return False


def main() -> None:
    args = parse_args()
    ensure_env()

    pipeline = FraudPipeline(
        dataset_dir=args.dataset,
        risk_threshold=args.risk_threshold,
        max_candidates_per_sender=args.max_candidates_per_sender,
    )
    pipeline.load()

    all_features = pipeline.compute_features()
    by_sender_candidates = pipeline.select_candidates(all_features)

    if args.max_senders is not None:
        limited = sorted(by_sender_candidates.keys())[: args.max_senders]
        by_sender_candidates = {sender: by_sender_candidates[sender] for sender in limited}

    session_id = generate_session_id()
    flagged_ids: set[str] = set()

    print(f"Session ID: {session_id}")
    print(f"Dataset: {args.dataset}")
    print(f"Transactions loaded: {len(all_features)}")
    print(f"Candidate senders: {len(by_sender_candidates)}")

    if args.dry_run:
        for candidates in by_sender_candidates.values():
            for item in candidates:
                if item.risk_score >= args.risk_threshold:
                    flagged_ids.add(item.transaction_id)
    else:
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

        # Seed with deterministic high-risk picks, then let LLM refine/add.
        for candidates in by_sender_candidates.values():
            for item in candidates:
                if item.risk_score >= args.risk_threshold:
                    flagged_ids.add(item.transaction_id)

        fallback_count = 0
        for sender_id, candidates in by_sender_candidates.items():
            profile = pipeline.users_by_biotag.get(sender_id, {})
            try:
                llm_flagged = llm_review_sender_candidates(
                    session_id=session_id,
                    model=model,
                    sender_id=sender_id,
                    sender_profile=profile,
                    candidates=candidates,
                )
                flagged_ids.update(llm_flagged)
            except Exception as exc:
                fallback_count += 1
                for item in candidates:
                    if item.risk_score >= args.risk_threshold:
                        flagged_ids.add(item.transaction_id)
                print(f"[WARN] LLM review failed for sender {sender_id}: {exc}")

        langfuse_client.flush()
        print(f"LLM fallback senders: {fallback_count}")

    flagged_ids = enforce_valid_output(all_features, flagged_ids)

    # Sort by timestamp to keep deterministic output.
    flagged_sorted = [
        feat.transaction_id
        for feat in sorted(all_features, key=lambda item: item.timestamp)
        if feat.transaction_id in flagged_ids
    ]

    write_ascii_output(args.output, flagged_sorted)

    print("Run complete.")
    print(f"Flagged transactions: {len(flagged_sorted)}")
    print(f"Output file: {args.output.resolve()}")
    print("Use the same session ID above in the submission modal.")


if __name__ == "__main__":
    main()
