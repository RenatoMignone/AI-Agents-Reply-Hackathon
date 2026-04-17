"""
tools.py - Data ingestion and analysis tools for MirrorPay fraud detection.

Design principles:
  - Tools FETCH and SUMMARIZE data. The LLM DECIDES what is anomalous.
  - Deterministic logic is kept minimal (statistics, indexing).
  - Expensive tools (communications) are clearly marked so the agent uses them sparingly.
  - A global data store (_DATA) is populated once at startup by load_dataset().
"""

import csv
import json
import math
import re
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Global data store – populated once by load_dataset()
# ---------------------------------------------------------------------------
_DATA: dict[str, Any] = {}


def load_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    """Load all dataset files into the global data store. Call once at startup.

    Returns a dict with dataset metadata (citizen count, transaction count, etc.).
    """
    d = Path(dataset_dir)

    # --- Transactions ---
    txns: list[dict[str, Any]] = []
    txn_path = d / "transactions.csv"
    if not txn_path.exists():
        raise FileNotFoundError(f"transactions.csv not found in {d}")
    with open(txn_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            txns.append(row)

    # --- Users ---
    users_path = d / "users.json"
    if not users_path.exists():
        raise FileNotFoundError(f"users.json not found in {d}")
    users: list[dict[str, Any]] = json.loads(users_path.read_text(encoding="utf-8"))

    # --- Locations ---
    locations_path = d / "locations.json"
    if not locations_path.exists():
        raise FileNotFoundError(f"locations.json not found in {d}")
    locations: list[dict[str, Any]] = json.loads(locations_path.read_text(encoding="utf-8"))

    # --- SMS (optional) ---
    sms: list[dict[str, Any]] = []
    sms_path = d / "sms.json"
    if sms_path.exists():
        sms = json.loads(sms_path.read_text(encoding="utf-8"))

    # --- Mails (optional) ---
    mails: list[dict[str, Any]] = []
    mails_path = d / "mails.json"
    if mails_path.exists():
        mails = json.loads(mails_path.read_text(encoding="utf-8"))

    # ===================================================================
    # Build indexes
    # ===================================================================

    # IBAN -> user profile
    iban_to_user: dict[str, dict[str, Any]] = {u["iban"]: u for u in users}

    # Discover citizen biotags by linking sender IBAN to known users.
    # This preserves citizen IDs containing accented characters.
    biotag_to_iban: dict[str, str] = {}
    for t in txns:
        sid = t.get("sender_id", "")
        siban = t.get("sender_iban", "")
        if sid and siban and siban in iban_to_user and sid not in biotag_to_iban:
            biotag_to_iban[sid] = siban

    biotag_to_user: dict[str, dict[str, Any]] = {}
    for biotag, iban in biotag_to_iban.items():
        if iban in iban_to_user:
            biotag_to_user[biotag] = iban_to_user[iban]

    # Transactions grouped by sender / recipient
    txns_by_sender: dict[str, list[dict[str, Any]]] = defaultdict(list)
    txns_by_recipient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in txns:
        sid = t.get("sender_id", "")
        if sid:
            txns_by_sender[sid].append(t)
        rid = t.get("recipient_id", "")
        if rid:
            txns_by_recipient[rid].append(t)

    # Locations grouped by biotag
    locs_by_biotag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for loc in locations:
        biotag = loc.get("biotag", "")
        if biotag:
            locs_by_biotag[biotag].append(loc)

    # --- SMS / Mail linking by citizen name ---
    name_to_biotag: dict[str, str] = {}
    for biotag, user in biotag_to_user.items():
        first_name = user.get("first_name", "")
        last_name = user.get("last_name", "")

        fn = first_name.lower()
        ln = last_name.lower()
        full = f"{fn} {ln}".strip()
        fn_norm = _normalize_text(first_name)
        ln_norm = _normalize_text(last_name)
        full_norm = _normalize_text(f"{first_name} {last_name}".strip())

        # Full-name keys have priority via sorted length matching below.
        if full:
            name_to_biotag[full] = biotag
        if full_norm:
            name_to_biotag[full_norm] = biotag

        if fn and fn not in name_to_biotag:
            name_to_biotag[fn] = biotag
        if ln and ln not in name_to_biotag:
            name_to_biotag[ln] = biotag
        if fn_norm and fn_norm not in name_to_biotag:
            name_to_biotag[fn_norm] = biotag
        if ln_norm and ln_norm not in name_to_biotag:
            name_to_biotag[ln_norm] = biotag

    # Sort by length descending so full names are matched before first names
    sorted_names = sorted(name_to_biotag.keys(), key=len, reverse=True)

    def _match_text_to_biotag(text: str) -> str | None:
        text_lower = text.lower()
        text_normalized = _normalize_text(text)
        for name in sorted_names:
            if name in text_lower or name in text_normalized:
                return name_to_biotag[name]
        return None

    sms_by_biotag: dict[str, list[str]] = defaultdict(list)
    for s in sms:
        bio = _match_text_to_biotag(s.get("sms", ""))
        if bio:
            sms_by_biotag[bio].append(s["sms"])

    mails_by_biotag: dict[str, list[str]] = defaultdict(list)
    for m in mails:
        bio = _match_text_to_biotag(m.get("mail", ""))
        if bio:
            mails_by_biotag[bio].append(m["mail"])

    risk_terms = (
        "urgent",
        "verify",
        "suspicious",
        "security",
        "locked",
        "suspend",
        "wallet",
        "crypto",
        "gift",
        "otp",
        "password",
        "confirm",
        "login",
    )
    typo_markers = ("paypa1", "amaz0n", "ub3r", "netfl1x", "secure-verify")

    comm_risk_by_biotag: dict[str, float] = defaultdict(float)
    for bio, messages in sms_by_biotag.items():
        for txt in messages:
            low = txt.lower()
            comm_risk_by_biotag[bio] += 1.0 if any(t in low for t in risk_terms) else 0.0
            comm_risk_by_biotag[bio] += 0.8 if "http://" in low or "https://" in low else 0.0
            comm_risk_by_biotag[bio] += 1.2 if "bit.ly" in low or "tinyurl" in low else 0.0
            comm_risk_by_biotag[bio] += 1.5 if any(t in low for t in typo_markers) else 0.0

    for bio, messages in mails_by_biotag.items():
        for txt in messages:
            low = _strip_html(txt).lower()
            comm_risk_by_biotag[bio] += 1.0 if any(t in low for t in risk_terms) else 0.0
            comm_risk_by_biotag[bio] += 0.8 if "http://" in low or "https://" in low else 0.0
            comm_risk_by_biotag[bio] += 1.2 if "bit.ly" in low or "tinyurl" in low else 0.0
            comm_risk_by_biotag[bio] += 1.5 if any(t in low for t in typo_markers) else 0.0

    citizen_ids = sorted(biotag_to_user.keys())

    _DATA.update(
        {
            "transactions": txns,
            "users": users,
            "locations": locations,
            "iban_to_user": iban_to_user,
            "biotag_to_iban": biotag_to_iban,
            "biotag_to_user": biotag_to_user,
            "txns_by_sender": dict(txns_by_sender),
            "txns_by_recipient": dict(txns_by_recipient),
            "locs_by_biotag": dict(locs_by_biotag),
            "sms_by_biotag": dict(sms_by_biotag),
            "mails_by_biotag": dict(mails_by_biotag),
            "comm_risk_by_biotag": dict(comm_risk_by_biotag),
            "name_to_biotag": name_to_biotag,
            "citizen_ids": citizen_ids,
            "all_txn_ids": {
                t.get("transaction_id", "") for t in txns if t.get("transaction_id")
            },
            "flagged_transactions": set(),
        }
    )

    return {
        "citizens": len(citizen_ids),
        "transactions": len(txns),
        "location_pings": len(locations),
        "sms": len(sms),
        "mails": len(mails),
    }


# ---------------------------------------------------------------------------
# Helper utilities (not exposed as tools)
# ---------------------------------------------------------------------------


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two GPS coordinates in km."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _strip_html(text: str) -> str:
    """Remove HTML tags for compact mail display."""
    return re.sub(r"<[^>]+>", "", text)


def _normalize_text(text: str) -> str:
    """Normalize text for robust accent-insensitive matching."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()


# ---------------------------------------------------------------------------
# LangChain @tool definitions
# ---------------------------------------------------------------------------


@tool
def list_citizens() -> str:
    """List all citizen IDs (biotags) with their name, job, salary, and transaction counts.
    START HERE to get an overview of the dataset."""
    lines = [f"Dataset: {len(_DATA['citizen_ids'])} citizens, {len(_DATA['transactions'])} total transactions\n"]
    for cid in _DATA["citizen_ids"]:
        user = _DATA["biotag_to_user"].get(cid, {})
        n_sent = len(_DATA["txns_by_sender"].get(cid, []))
        n_recv = len(_DATA["txns_by_recipient"].get(cid, []))
        salary = user.get("salary", "?")
        lines.append(
            f"  {cid}: {user.get('first_name','')} {user.get('last_name','')}, "
            f"Job={user.get('job','?')}, Salary={salary}, "
            f"Sent={n_sent}, Recv={n_recv}"
        )
    return "\n".join(lines)


@tool
def get_citizen_profile(citizen_id: str) -> str:
    """Get full profile for a citizen: name, salary, job, residence, and personal description.
    The description contains behavioral baseline information useful for anomaly detection."""
    user = _DATA["biotag_to_user"].get(citizen_id)
    if not user:
        return f"No profile found for citizen '{citizen_id}'."
    return json.dumps(user, indent=2, ensure_ascii=False)


@tool
def get_citizen_transaction_summary(citizen_id: str) -> str:
    """Get a statistical summary of a citizen's transactions: amounts, types, timing,
    payment methods, recipients, balance trend, and the list of all their transaction IDs.
    This is the PRIMARY investigation tool – call it for every citizen."""
    sent = _DATA["txns_by_sender"].get(citizen_id, [])
    recv = _DATA["txns_by_recipient"].get(citizen_id, [])

    if not sent and not recv:
        return f"No transactions found for citizen '{citizen_id}'."

    user = _DATA["biotag_to_user"].get(citizen_id, {})
    salary = _safe_float(user.get("salary", 0))

    # --- Sent analysis ---
    sent_amounts = [_safe_float(t.get("amount", 0)) for t in sent]
    type_counts: dict[str, int] = defaultdict(int)
    method_counts: dict[str, int] = defaultdict(int)
    location_counts: dict[str, int] = defaultdict(int)
    hour_counts: dict[int, int] = defaultdict(int)

    for t in sent:
        type_counts[t.get("transaction_type", "unknown")] += 1
        pm = t.get("payment_method", "")
        if pm:
            method_counts[pm] += 1
        loc = t.get("location", "")
        if loc:
            location_counts[loc] += 1
        try:
            ts = t.get("timestamp", "")
            if ts:
                hour_counts[datetime.fromisoformat(ts.replace("Z", "+00:00")).hour] += 1
        except (ValueError, KeyError):
            pass

    # Night transactions (00:00–06:00)
    night = sum(v for h, v in hour_counts.items() if 0 <= h < 6)

    # Balance tracking
    balances: list[float] = []
    for t in sorted(sent, key=lambda x: x.get("timestamp", "")):
        b = t.get("balance_after", "")
        if b:
            balances.append(_safe_float(b))

    # Recipient breakdown
    recip_summary: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "total": 0.0})
    for t in sent:
        rid = t.get("recipient_id", "") or t.get("recipient_iban", "unknown")
        recip_summary[rid]["n"] += 1
        recip_summary[rid]["total"] += _safe_float(t.get("amount", 0))

    top_recips = dict(
        sorted(recip_summary.items(), key=lambda x: x[1]["total"], reverse=True)[:7]
    )

    # Large transactions (> 30% of monthly salary equivalent)
    monthly_salary = salary / 12 if salary else 0
    large_txns = (
        [
            {
                "id": t.get("transaction_id", "UNKNOWN"),
                "amount": t.get("amount", 0),
                "to": t.get("recipient_id", ""),
            }
            for t in sent
            if monthly_salary and _safe_float(t.get("amount", 0)) > monthly_salary * 0.5
        ]
        if monthly_salary
        else []
    )

    summary = {
        "citizen_id": citizen_id,
        "annual_salary": salary,
        "monthly_salary_approx": round(monthly_salary, 2),
        "sent_count": len(sent),
        "received_count": len(recv),
        "sent_amounts": {
            "total": round(sum(sent_amounts), 2),
            "mean": round(sum(sent_amounts) / len(sent_amounts), 2) if sent_amounts else 0,
            "min": round(min(sent_amounts), 2) if sent_amounts else 0,
            "max": round(max(sent_amounts), 2) if sent_amounts else 0,
        },
        "transaction_types": dict(type_counts),
        "payment_methods": dict(method_counts),
        "locations": dict(location_counts),
        "night_transactions": night,
        "hour_distribution": dict(sorted(hour_counts.items())),
        "large_transactions": large_txns[:10],
        "unique_recipients": len(recip_summary),
        "top_recipients": top_recips,
        "balance": {
            "first": balances[0] if balances else None,
            "last": balances[-1] if balances else None,
            "min": min(balances) if balances else None,
            "dropped_below_zero": any(b < 0 for b in balances),
        },
        "all_sent_transaction_ids": [
            t.get("transaction_id", "")
            for t in sorted(sent, key=lambda x: x.get("timestamp", ""))
            if t.get("transaction_id")
        ],
    }

    return json.dumps(summary, indent=2)


@tool
def get_citizen_transactions_detail(citizen_id: str) -> str:
    """Get the FULL chronological list of a citizen's transactions with all fields.
    Use ONLY when you need to inspect individual transactions closely after the summary
    flagged something suspicious."""
    sent = _DATA["txns_by_sender"].get(citizen_id, [])
    recv = _DATA["txns_by_recipient"].get(citizen_id, [])

    if not sent and not recv:
        return f"No transactions found for citizen '{citizen_id}'."

    rows: list[dict[str, str]] = []
    for t in sorted(sent, key=lambda x: x.get("timestamp", "")):
        rows.append(
            {
                "id": t.get("transaction_id", "UNKNOWN"),
                "dir": "SENT",
                "counterparty": t.get("recipient_id", ""),
                "type": t.get("transaction_type", ""),
                "amount": t.get("amount", ""),
                "location": t.get("location", ""),
                "method": t.get("payment_method", ""),
                "balance": t.get("balance_after", ""),
                "desc": t.get("description", ""),
                "time": t.get("timestamp", ""),
            }
        )

    for t in sorted(recv, key=lambda x: x.get("timestamp", "")):
        rows.append(
            {
                "id": t.get("transaction_id", "UNKNOWN"),
                "dir": "RECV",
                "counterparty": t.get("sender_id", ""),
                "type": t.get("transaction_type", ""),
                "amount": t.get("amount", ""),
                "desc": t.get("description", ""),
                "time": t.get("timestamp", ""),
            }
        )

    return json.dumps(rows, indent=2)


@tool
def get_citizen_location_summary(citizen_id: str) -> str:
    """Get GPS movement analysis: cities visited, distance from residence,
    unusual far-away pings, and date range of observations."""
    pings = _DATA["locs_by_biotag"].get(citizen_id, [])
    user = _DATA["biotag_to_user"].get(citizen_id, {})

    if not pings:
        return f"No location data for citizen '{citizen_id}'."

    home = user.get("residence", {})
    home_lat = _safe_float(home.get("lat", 0))
    home_lng = _safe_float(home.get("lng", 0))
    home_city = home.get("city", "unknown")

    sorted_pings = sorted(pings, key=lambda p: p.get("timestamp", ""))

    cities: dict[str, int] = defaultdict(int)
    distances: list[float] = []
    for p in sorted_pings:
        cities[p.get("city", "unknown")] += 1
        d = _haversine_km(
            home_lat,
            home_lng,
            _safe_float(p.get("lat", 0)),
            _safe_float(p.get("lng", 0)),
        )
        distances.append(d)

    # Pings far from home (> 50km)
    far_pings = [
        {"city": p.get("city", "?"), "time": p.get("timestamp", "?"), "km_from_home": round(d, 1)}
        for p, d in zip(sorted_pings, distances)
        if d > 50
    ]

    summary = {
        "citizen_id": citizen_id,
        "home_city": home_city,
        "total_pings": len(pings),
        "cities_visited": dict(cities),
        "distance_from_home_km": {
            "mean": round(sum(distances) / len(distances), 1) if distances else 0,
            "max": round(max(distances), 1) if distances else 0,
        },
        "far_locations": far_pings[:15],
        "date_range": {
            "first": sorted_pings[0].get("timestamp", ""),
            "last": sorted_pings[-1].get("timestamp", ""),
        },
    }

    return json.dumps(summary, indent=2)


@tool
def get_citizen_communications(citizen_id: str) -> str:
    """Get SMS and email communications linked to a citizen.
    EXPENSIVE in tokens – only use for citizens already flagged as suspicious
    to look for phishing, social engineering, or fraud coordination evidence."""
    sms_list = _DATA["sms_by_biotag"].get(citizen_id, [])
    mail_list = _DATA["mails_by_biotag"].get(citizen_id, [])

    # Compact SMS
    sms_compact = sms_list[:12]

    # Compact mails – strip HTML, truncate
    mail_compact: list[str] = []
    for m in mail_list[:8]:
        cleaned = _strip_html(m)
        # Collapse whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) > 800:
            cleaned = cleaned[:800] + " ...[truncated]"
        mail_compact.append(cleaned)

    all_msgs = [m.lower() for m in sms_list] + [_strip_html(m).lower() for m in mail_list]
    risk_terms = (
        "urgent",
        "verify",
        "suspicious",
        "security",
        "locked",
        "suspend",
        "wallet",
        "crypto",
        "gift",
        "otp",
        "password",
        "confirm",
        "login",
    )
    typo_markers = ("paypa1", "amaz0n", "ub3r", "netfl1x", "secure-verify")

    risk_counts = {
        "messages_with_risk_terms": sum(1 for m in all_msgs if any(t in m for t in risk_terms)),
        "messages_with_links": sum(1 for m in all_msgs if "http://" in m or "https://" in m),
        "messages_with_short_links": sum(1 for m in all_msgs if "bit.ly" in m or "tinyurl" in m),
        "messages_with_typosquatting_markers": sum(1 for m in all_msgs if any(t in m for t in typo_markers)),
    }

    result = {
        "citizen_id": citizen_id,
        "total_sms": len(sms_list),
        "total_mails": len(mail_list),
        "risk_counts": risk_counts,
        "sms_samples": sms_compact,
        "mail_samples": mail_compact,
    }

    return json.dumps(result, indent=2, ensure_ascii=False)


@tool
def mark_fraudulent_transactions(transaction_ids: str) -> str:
    """Record suspected fraudulent Transaction IDs.
    Pass a COMMA-SEPARATED string of Transaction IDs (UUIDs).
    Call this once you have finished analyzing all citizens and are confident
    in your final list. You can call this multiple times – IDs accumulate."""
    ids = [tid.strip() for tid in transaction_ids.split(",") if tid.strip()]
    valid_ids = _DATA.get("all_txn_ids", set())

    confirmed: list[str] = []
    invalid: list[str] = []
    for tid in ids:
        if tid in valid_ids:
            confirmed.append(tid)
            _DATA["flagged_transactions"].add(tid)
        else:
            invalid.append(tid)

    parts = [f"Recorded {len(confirmed)} fraudulent transaction(s). Total flagged so far: {len(_DATA['flagged_transactions'])}."]
    if invalid:
        parts.append(f"WARNING: {len(invalid)} ID(s) not found in dataset and were skipped: {invalid[:5]}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API for main.py
# ---------------------------------------------------------------------------


def get_flagged_transactions() -> list[str]:
    """Return all flagged transaction IDs (sorted). Called by main.py after the agent finishes."""
    return sorted(_DATA.get("flagged_transactions", set()))


def get_transaction_count() -> int:
    """Return the number of loaded transactions for adaptive run policies."""
    return len(_DATA.get("transactions", []))


def get_all_transaction_ids() -> set[str]:
    """Return all known transaction IDs in the currently loaded dataset."""
    return set(_DATA.get("all_txn_ids", set()))


def rank_risky_transactions(seed_ids: list[str] | None = None) -> list[str]:
    """Deterministically rank transactions from highest to lowest fraud risk."""
    txns: list[dict[str, Any]] = _DATA.get("transactions", [])
    if not txns:
        return []

    id_to_txn = {
        t.get("transaction_id", ""): t
        for t in txns
        if t.get("transaction_id")
    }
    comm_risk_by_biotag: dict[str, float] = _DATA.get("comm_risk_by_biotag", {})

    seed_set = {tid for tid in (seed_ids or []) if tid in id_to_txn}
    seed_txns = [id_to_txn[tid] for tid in seed_set]
    suspicious_senders = {t.get("sender_id", "") for t in seed_txns if t.get("sender_id")}
    suspicious_recipients = {
        t.get("recipient_id", "") for t in seed_txns if t.get("recipient_id")
    }

    sender_amounts: dict[str, list[float]] = defaultdict(list)
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for t in txns:
        sender = t.get("sender_id", "")
        recipient = t.get("recipient_id", "")
        amount = _safe_float(t.get("amount", 0))
        sender_amounts[sender].append(amount)
        if sender and recipient:
            pair_counts[(sender, recipient)] += 1

    sender_stats: dict[str, tuple[float, float]] = {}
    for sender, values in sender_amounts.items():
        if not values:
            sender_stats[sender] = (0.0, 0.0)
            continue
        mean = sum(values) / len(values)
        var = sum((x - mean) ** 2 for x in values) / max(1, len(values))
        sender_stats[sender] = (mean, math.sqrt(var))

    sorted_amounts = sorted(_safe_float(t.get("amount", 0)) for t in txns)

    def _pct(p: float) -> float:
        if not sorted_amounts:
            return 0.0
        idx = min(len(sorted_amounts) - 1, max(0, int((len(sorted_amounts) - 1) * p)))
        return sorted_amounts[idx]

    p90 = _pct(0.90)
    p95 = _pct(0.95)
    p98 = _pct(0.98)

    suspicious_terms = (
        "urgent",
        "verify",
        "security",
        "unlock",
        "suspend",
        "crypto",
        "gift",
        "fee",
        "wire",
        "wallet",
        "otp",
        "password",
    )

    legit_recurring_terms = (
        "rent payment",
        "salary payment",
        "insurance",
        "subscription",
        "phone bill",
        "gas bill",
        "electricity",
        "water bill",
        "loan payment",
        "tuition",
    )

    recurring_prefixes = (
        "HOME",
        "RENT",
        "RES",
        "APP",
        "ABIT",
        "STAN",
    )

    def _score(t: dict[str, Any]) -> tuple[float, float]:
        score = 0.0
        amount = _safe_float(t.get("amount", 0))

        sender = t.get("sender_id", "")
        recipient = t.get("recipient_id", "")
        tx_type = t.get("transaction_type", "")
        desc = str(t.get("description", "")).lower()
        method = str(t.get("payment_method", "")).strip().lower()
        user = biotag_to_user.get(sender, {})
        monthly_salary = _safe_float(user.get("salary", 0)) / 12

        # SEED TRANSACTIONS: already flagged by LLM
        if sender in suspicious_senders:
            score += 5.5
        if recipient and recipient in suspicious_recipients:
            score += 4.0
        if recipient and recipient in suspicious_senders:
            score += 2.5

        # STATISTICAL ANOMALY: outlier in sender's normal distribution
        mean, std = sender_stats.get(sender, (0.0, 0.0))
        if std > 1e-6:
            z = (amount - mean) / std
            if z >= 3.5:
                score += 4.0
            elif z >= 3.0:
                score += 3.5
            elif z >= 2.2:
                score += 2.0
            elif z >= 1.5:
                score += 1.0

        # ECONOMIC IMPACT: high-value mismatch with salary
        if monthly_salary > 0:
            if amount > monthly_salary * 3.0:
                score += 3.0
            elif amount > monthly_salary * 1.5:
                score += 1.5
            elif amount > monthly_salary * 0.8:
                score += 0.4

        # ABSOLUTE HIGH-VALUE: percentile-based
        if amount >= p98:
            score += 3.2
        elif amount >= p95:
            score += 1.8
        elif amount >= p90:
            score += 0.9

        # TRANSACTION TYPE RISK
        if tx_type in {"e-commerce", "withdrawal"}:
            score += 2.0
        elif tx_type == "transfer":
            score += 0.7
        elif tx_type == "in-person payment":
            score += 1.5

        # PAYMENT METHOD RISK (mobile and digital are riskier)
        risky_methods = {"paypal", "google pay", "mobile phone", "mobile device", "smartwatch"}
        if method in risky_methods:
            score += 1.2
        elif method == "debit card":
            score += 0.3

        # TEMPORAL ANOMALY: suspicious timing
        try:
            hour = datetime.fromisoformat(str(t.get("timestamp", "")).replace("Z", "+00:00")).hour
            if tx_type in {"withdrawal", "e-commerce"}:
                if 0 <= hour < 4:
                    score += 1.2
                elif 2 <= hour < 7:
                    score += 0.8
            elif tx_type == "transfer" and 0 <= hour < 6:
                score += 0.5
        except Exception:
            pass

        # DESCRIPTION-BASED RED FLAGS
        if any(term in desc for term in suspicious_terms):
            score += 1.5

        # COMMUNICATION RISK
        score += min(2.2, comm_risk_by_biotag.get(sender, 0.0) / 32.0)

        # RECIPIENT RELATIONSHIP ANALYSIS
        pair_n = pair_counts.get((sender, recipient), 0)
        if recipient:
            if pair_n == 0:
                score += 1.2
            elif pair_n == 1:
                score += 0.9
            elif pair_n == 2:
                score += 0.4

        # BALANCE IMPACT: check if transaction causes significant drop
        try:
            balance_after = _safe_float(t.get("balance_after", 0))
            balance_before = balance_after + amount
            if monthly_salary > 0 and balance_before > 0:
                balance_drop_pct = (amount / balance_before) * 100
                if balance_drop_pct > 80:
                    score += 2.0
                elif balance_drop_pct > 50:
                    score += 1.0
            if balance_after < 0:
                score += 1.5
        except Exception:
            pass

        # NEGATIVE DEDUCTIONS: legitimate patterns
        recurring_legit = False
        if sender and recipient:
            recurring_legit = pair_n >= 6 and (
                any(term in desc for term in legit_recurring_terms)
                or recipient.startswith(recurring_prefixes)
            )

        if tx_type == "transfer" and "salary payment" in desc and sender.startswith("EMP"):
            score -= 8.0
        if recurring_legit:
            score -= 5.5
        if tx_type == "direct debit" and pair_n >= 6:
            score -= 3.0

        return score, amount

    ranked = sorted(txns, key=_score, reverse=True)
    return [t.get("transaction_id", "") for t in ranked if t.get("transaction_id")]


def expand_flagged_transactions(seed_ids: list[str], target_count: int) -> list[str]:
    """Expand LLM-picked fraud seeds into a broader ranked candidate set.

    This helper is used only when coverage is too low; the LLM remains the
    primary decision-maker by generating the initial seed IDs.
    """
    all_ids = get_all_transaction_ids()
    ranked_ids = rank_risky_transactions(seed_ids)

    expanded: list[str] = []
    seen: set[str] = set()

    for tid in seed_ids:
        if tid in all_ids and tid not in seen:
            expanded.append(tid)
            seen.add(tid)

    for tid in ranked_ids:
        if tid in seen:
            continue
        expanded.append(tid)
        seen.add(tid)
        if len(expanded) >= target_count:
            break

    return expanded


def get_all_tools() -> list:
    """Return the list of all tools to register with the agent."""
    return [
        list_citizens,
        get_citizen_profile,
        get_citizen_transaction_summary,
        get_citizen_transactions_detail,
        get_citizen_location_summary,
        get_citizen_communications,
        mark_fraudulent_transactions,
    ]
