import argparse
import csv
import json
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import ulid
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langfuse import observe
from langfuse.langchain import CallbackHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an LLM-driven sandbox solver and write submission output."
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="Public level to run (1, 2, or 3).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("../00_Sandbox_Sample_Material/Submission_Levels"),
        help="Path to Submission_Levels (or Public_Levels) directory.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/llama-3.1-8b-instruct",
        help="OpenRouter model ID.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Model temperature.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output TXT path. Defaults to ./outputs/public_lev_<n>_predictions.txt",
    )
    parser.add_argument(
        "--max-citizens",
        type=int,
        default=None,
        help="Optional cap for fast local tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM calls and run only deterministic baseline for debugging.",
    )
    parser.add_argument(
        "--full-persona",
        action="store_true",
        help="Use the full persona block in prompts (higher token cost).",
    )
    parser.add_argument(
        "--verbose-context",
        action="store_true",
        help="Include verbose recent events/pings in prompts (higher token cost).",
    )
    parser.add_argument(
        "--enable-review",
        action="store_true",
        help="Enable a second-pass review for uncertain or disputed decisions.",
    )
    parser.add_argument(
        "--review-model",
        type=str,
        default=None,
        help="Optional model ID for second-pass review (defaults to primary model).",
    )
    parser.add_argument(
        "--review-low",
        type=float,
        default=0.45,
        help="Lower confidence bound for triggering review.",
    )
    parser.add_argument(
        "--review-high",
        type=float,
        default=0.75,
        help="Upper confidence bound for triggering review.",
    )
    parser.add_argument(
        "--review-on-disagreement",
        action="store_true",
        help="Trigger review when LLM and heuristic decisions disagree.",
    )
    return parser.parse_args()


def generate_session_id() -> str:
    team = os.getenv("TEAM_NAME", "team").strip().replace(" ", "-")
    return f"{team}-{ulid.new().str}"


def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math

    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parse_personas(personas_md_path: Path) -> dict[str, str]:
    text = personas_md_path.read_text(encoding="utf-8")
    pattern = re.compile(r"^##\s+([A-Z0-9]{8})\s+-\s+.+$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    personas: dict[str, str] = {}
    for idx, match in enumerate(matches):
        user_id = match.group(1)
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        personas[user_id] = text[start:end].strip()
    return personas


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.mean(values))


def trend_delta(values: list[float], window: int = 3) -> float:
    if len(values) < 2:
        return 0.0
    left = values[:window]
    right = values[-window:]
    return float(statistics.mean(right) - statistics.mean(left))


@dataclass
class CitizenSnapshot:
    user_id: str
    profile: dict[str, Any]
    persona: str
    status_summary: dict[str, Any]
    location_summary: dict[str, Any]


def summarize_status(events: list[dict[str, Any]]) -> dict[str, Any]:
    sorted_events = sorted(events, key=lambda e: parse_iso(e["Timestamp"]))
    pai = [float(e["PhysicalActivityIndex"]) for e in sorted_events]
    sqi = [float(e["SleepQualityIndex"]) for e in sorted_events]
    eel = [float(e["EnvironmentalExposureLevel"]) for e in sorted_events]

    event_type_counts: dict[str, int] = defaultdict(int)
    for event in sorted_events:
        event_type_counts[event["EventType"]] += 1

    return {
        "events_count": len(sorted_events),
        "event_type_counts": dict(event_type_counts),
        "activity_mean": mean_or_none(pai),
        "sleep_mean": mean_or_none(sqi),
        "exposure_mean": mean_or_none(eel),
        "activity_delta": trend_delta(pai),
        "sleep_delta": trend_delta(sqi),
        "exposure_delta": trend_delta(eel),
        "recent_events": sorted_events[-5:],
    }


def summarize_locations(records: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, Any]:
    sorted_records = sorted(records, key=lambda r: parse_iso(r["timestamp"]))
    home_lat = float(profile["residence"]["lat"])
    home_lng = float(profile["residence"]["lng"])

    distances = [
        haversine_km(home_lat, home_lng, float(r["lat"]), float(r["lng"]))
        for r in sorted_records
    ]
    unique_cities = sorted({r["city"] for r in sorted_records if r.get("city")})

    return {
        "pings_count": len(sorted_records),
        "unique_cities": unique_cities,
        "distance_home_mean_km": mean_or_none(distances),
        "distance_home_max_km": max(distances) if distances else None,
        "recent_pings": sorted_records[-5:],
    }


def build_snapshot(
    user_id: str,
    users_by_id: dict[str, dict[str, Any]],
    personas_by_id: dict[str, str],
    status_by_id: dict[str, list[dict[str, Any]]],
    locations_by_id: dict[str, list[dict[str, Any]]],
) -> CitizenSnapshot:
    profile = users_by_id[user_id]
    persona = personas_by_id.get(user_id, "Persona not available.")
    status_summary = summarize_status(status_by_id.get(user_id, []))
    location_summary = summarize_locations(locations_by_id.get(user_id, []), profile)

    return CitizenSnapshot(
        user_id=user_id,
        profile=profile,
        persona=persona,
        status_summary=status_summary,
        location_summary=location_summary,
    )


def build_prompt(snapshot: CitizenSnapshot) -> str:
    return f"""
You are an anomaly detection specialist for public-health preventive monitoring.

Task:
- Compare observed behavior against the citizen persona baseline.
- Decide if this citizen should be flagged for preventive support activation.

Return valid JSON only with this exact schema:
{{
  "decision": "ANOMALOUS" | "NORMAL",
  "confidence": 0.0-1.0,
  "reasons": ["short reason 1", "short reason 2", "..."]
}}

Guidelines:
- Flag only when evidence indicates meaningful deviation from baseline.
- Avoid over-flagging on weak or isolated signals.
- Consider trends: activity, sleep, environmental exposure, healthcare event patterns, mobility shifts.

Citizen ID: {snapshot.user_id}

Profile:
{json.dumps(snapshot.profile, indent=2)}

Persona baseline:
{snapshot.persona}

Status summary:
{json.dumps(snapshot.status_summary, indent=2)}

Location summary:
{json.dumps(snapshot.location_summary, indent=2)}
""".strip()


def compress_persona(persona_text: str) -> str:
    lines = [line.strip() for line in persona_text.splitlines() if line.strip()]

    header = next((line for line in lines if line.startswith("## ")), "")
    key_fields = [
        line
        for line in lines
        if line.startswith("**Mobility:**")
        or line.startswith("**Health behavior:**")
        or line.startswith("**Social pattern:**")
    ]

    if key_fields:
        return "\n".join(([header] if header else []) + key_fields)

    # Fallback for unexpected persona formats.
    return "\n".join(lines[:8])


def build_compact_status_context(status_summary: dict[str, Any], verbose: bool) -> dict[str, Any]:
    if verbose:
        return status_summary

    return {
        "events_count": status_summary.get("events_count"),
        "event_type_counts": status_summary.get("event_type_counts"),
        "activity_mean": status_summary.get("activity_mean"),
        "sleep_mean": status_summary.get("sleep_mean"),
        "exposure_mean": status_summary.get("exposure_mean"),
        "activity_delta": status_summary.get("activity_delta"),
        "sleep_delta": status_summary.get("sleep_delta"),
        "exposure_delta": status_summary.get("exposure_delta"),
        "recent_events": (status_summary.get("recent_events") or [])[-2:],
    }


def build_compact_location_context(location_summary: dict[str, Any], verbose: bool) -> dict[str, Any]:
    if verbose:
        return location_summary

    return {
        "pings_count": location_summary.get("pings_count"),
        "unique_cities": location_summary.get("unique_cities"),
        "distance_home_mean_km": location_summary.get("distance_home_mean_km"),
        "distance_home_max_km": location_summary.get("distance_home_max_km"),
        "recent_pings": (location_summary.get("recent_pings") or [])[-2:],
    }


def build_optimized_prompt(
    snapshot: CitizenSnapshot,
    heuristic_result: dict[str, Any],
    use_full_persona: bool,
    verbose_context: bool,
) -> str:
    persona_text = snapshot.persona if use_full_persona else compress_persona(snapshot.persona)
    status_context = build_compact_status_context(snapshot.status_summary, verbose=verbose_context)
    location_context = build_compact_location_context(snapshot.location_summary, verbose=verbose_context)

    return f"""
You are the final decision-maker for anomaly detection in preventive monitoring.

Objective:
- maximize detection quality while avoiding unnecessary false positives,
- keep decisions calibrated and robust under noisy signals.

Return valid JSON only:
{{
  "decision": "ANOMALOUS" | "NORMAL",
  "confidence": 0.0-1.0,
  "reasons": ["short reason 1", "short reason 2"]
}}

Citizen ID: {snapshot.user_id}

Profile:
{json.dumps(snapshot.profile, indent=2)}

Persona baseline:
{persona_text}

Status summary:
{json.dumps(status_context, indent=2)}

Location summary:
{json.dumps(location_context, indent=2)}

Deterministic prior (for calibration only, not mandatory):
{json.dumps(heuristic_result, indent=2)}
""".strip()


def extract_text_from_model_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    # Some providers return structured content parts instead of a plain string.
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    chunks.append(text_value)
        return "\n".join(chunks)

    return str(content)


def parse_llm_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Model returned empty content.")

    # Remove code fences if the model wraps JSON in markdown.
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: find the first decodable JSON object in the text.
    decoder = json.JSONDecoder()
    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not parse a JSON object from model output.")


def normalize_decision_payload(parsed: dict[str, Any]) -> dict[str, Any]:
    decision = str(parsed.get("decision", "")).upper().strip()
    if decision not in {"ANOMALOUS", "NORMAL"}:
        raise ValueError(f"Invalid or missing decision: {decision!r}")

    confidence_raw = parsed.get("confidence", 0.5)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    reasons_raw = parsed.get("reasons", [])
    if isinstance(reasons_raw, str):
        reasons = [reasons_raw]
    elif isinstance(reasons_raw, list):
        reasons = [str(r) for r in reasons_raw if str(r).strip()]
    else:
        reasons = []

    if not reasons:
        reasons = ["No explanation provided by model."]

    return {
        "decision": decision,
        "confidence": confidence,
        "reasons": reasons,
    }


def should_run_review(
    llm_result: dict[str, Any],
    heuristic_result: dict[str, Any],
    low: float,
    high: float,
    review_on_disagreement: bool,
) -> bool:
    confidence = float(llm_result.get("confidence", 0.5))
    uncertain = low <= confidence <= high
    disagreement = str(llm_result.get("decision", "")).upper() != str(
        heuristic_result.get("decision", "")
    ).upper()

    return uncertain or (review_on_disagreement and disagreement)


def heuristic_decision(snapshot: CitizenSnapshot) -> dict[str, Any]:
    score = 0
    reasons = []
    s = snapshot.status_summary
    l = snapshot.location_summary

    if (s.get("sleep_delta") or 0) < -7:
        score += 1
        reasons.append("Sleep quality dropped markedly over time.")
    if (s.get("activity_delta") or 0) < -7:
        score += 1
        reasons.append("Physical activity dropped markedly over time.")
    if (s.get("exposure_delta") or 0) > 7:
        score += 1
        reasons.append("Environmental exposure increased over time.")
    if (l.get("distance_home_max_km") or 0) < 1.0 and (l.get("pings_count") or 0) > 30:
        score += 1
        reasons.append("Mobility appears unusually confined.")

    decision = "ANOMALOUS" if score >= 2 else "NORMAL"
    confidence = min(0.95, 0.45 + score * 0.15)
    if not reasons:
        reasons = ["No strong deviation found by baseline heuristic."]

    return {"decision": decision, "confidence": confidence, "reasons": reasons}


@observe()
def llm_decide(
    session_id: str,
    model: ChatOpenAI,
    snapshot: CitizenSnapshot,
    heuristic_result: dict[str, Any],
    use_full_persona: bool,
    verbose_context: bool,
) -> dict[str, Any]:
    handler = CallbackHandler()
    base_prompt = build_optimized_prompt(
        snapshot=snapshot,
        heuristic_result=heuristic_result,
        use_full_persona=use_full_persona,
        verbose_context=verbose_context,
    )

    last_error: Exception | None = None
    for attempt in range(1, 4):
        prompt = base_prompt
        if attempt > 1:
            prompt += (
                "\n\nIMPORTANT: Your previous answer was invalid. "
                "Return ONLY one valid JSON object with keys decision, confidence, reasons."
            )

        response = model.invoke(
            [HumanMessage(content=prompt)],
            config={
                "callbacks": [handler],
                "metadata": {"langfuse_session_id": session_id},
            },
        )

        raw_text = extract_text_from_model_content(response.content)
        try:
            parsed = parse_llm_json(raw_text)
            return normalize_decision_payload(parsed)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed to parse a valid decision after retries for citizen {snapshot.user_id}. "
        f"Last error: {last_error}"
    )


@observe()
def llm_review_decide(
    session_id: str,
    model: ChatOpenAI,
    snapshot: CitizenSnapshot,
    first_result: dict[str, Any],
    heuristic_result: dict[str, Any],
) -> dict[str, Any]:
    handler = CallbackHandler()
    review_prompt = f"""
You are a senior reviewer. Re-evaluate this citizen decision.

Return valid JSON only:
{{
  "decision": "ANOMALOUS" | "NORMAL",
  "confidence": 0.0-1.0,
  "reasons": ["short reason 1", "short reason 2"]
}}

Citizen ID: {snapshot.user_id}

First-pass decision:
{json.dumps(first_result, indent=2)}

Heuristic prior:
{json.dumps(heuristic_result, indent=2)}

Status summary:
{json.dumps(build_compact_status_context(snapshot.status_summary, verbose=False), indent=2)}

Location summary:
{json.dumps(build_compact_location_context(snapshot.location_summary, verbose=False), indent=2)}

If evidence is weak or contradictory, prefer NORMAL.
""".strip()

    last_error: Exception | None = None
    for attempt in range(1, 3):
        prompt = review_prompt
        if attempt > 1:
            prompt += "\n\nIMPORTANT: Return only one valid JSON object."

        response = model.invoke(
            [HumanMessage(content=prompt)],
            config={
                "callbacks": [handler],
                "metadata": {"langfuse_session_id": session_id},
            },
        )

        raw_text = extract_text_from_model_content(response.content)
        try:
            parsed = parse_llm_json(raw_text)
            return normalize_decision_payload(parsed)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Review pass failed for citizen {snapshot.user_id}. Last error: {last_error}"
    )


def load_level(level_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    users_path = level_dir / "users.json"
    locations_path = level_dir / "locations.json"
    status_path = level_dir / "status.csv"
    personas_path = level_dir / "personas.md"

    if not all(path.exists() for path in [users_path, locations_path, status_path, personas_path]):
        raise FileNotFoundError(f"Missing one or more required files in {level_dir}")

    users = json.loads(users_path.read_text(encoding="utf-8"))
    locations = json.loads(locations_path.read_text(encoding="utf-8"))

    status_rows: list[dict[str, Any]] = []
    with status_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status_rows.append(row)

    personas = parse_personas(personas_path)

    users_by_id = {u["user_id"]: u for u in users}

    status_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in status_rows:
        status_by_id[row["CitizenID"]].append(row)

    locations_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in locations:
        locations_by_id[row["user_id"]].append(row)

    return users_by_id, status_by_id, locations_by_id, personas


def is_level_folder(path: Path) -> bool:
    required = ["users.json", "locations.json", "status.csv", "personas.md"]
    return all((path / name).exists() for name in required)


def resolve_level_dir(data_root: Path, level: int) -> Path:
    # Support passing --data-root directly as an extracted level folder.
    if is_level_folder(data_root):
        return data_root

    nested = data_root / f"public_lev_{level}" / f"public_lev_{level}"
    if nested.exists():
        return nested

    flat = data_root / f"public_lev_{level}"
    if flat.exists():
        return flat

    return data_root


def ensure_env() -> None:
    load_dotenv(find_dotenv())


def ensure_llm_env() -> None:
    required = ["OPENROUTER_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Missing LLM-related env vars: "
            + ", ".join(missing)
            + ". Add them to the repo root .env file, or use --dry-run."
        )


def write_output(output_path: Path, flagged_ids: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{citizen_id}\n" for citizen_id in flagged_ids]

    # Challenge requires ASCII output.
    ascii_payload = "".join(lines).encode("ascii", errors="strict")
    output_path.write_bytes(ascii_payload)


def main() -> None:
    args = parse_args()
    ensure_env()

    level_dir = resolve_level_dir(args.data_root, args.level)
    users_by_id, status_by_id, locations_by_id, personas_by_id = load_level(level_dir)

    session_id = generate_session_id()
    output_path = args.output or Path("outputs") / f"public_lev_{args.level}_predictions.txt"

    model = None
    review_model = None
    if not args.dry_run:
        ensure_llm_env()
        model = ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=args.model,
            temperature=args.temperature,
        )
        if args.enable_review:
            review_model = ChatOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                model=args.review_model or args.model,
                temperature=0.0,
            )

    user_ids = sorted(users_by_id.keys())
    if args.max_citizens is not None:
        user_ids = user_ids[: args.max_citizens]

    flagged_ids: list[str] = []
    review_count = 0
    fallback_count = 0

    print(f"Session ID: {session_id}")
    print(f"Running level {args.level} from: {level_dir}")
    print(f"Citizens to analyze: {len(user_ids)}")
    print(f"Mode: {'dry-run heuristic' if args.dry_run else 'LLM'}")

    for user_id in user_ids:
        snapshot = build_snapshot(
            user_id=user_id,
            users_by_id=users_by_id,
            personas_by_id=personas_by_id,
            status_by_id=status_by_id,
            locations_by_id=locations_by_id,
        )

        if args.dry_run:
            result = heuristic_decision(snapshot)
        else:
            heuristic_result = heuristic_decision(snapshot)
            try:
                result = llm_decide(
                    session_id=session_id,
                    model=model,
                    snapshot=snapshot,
                    heuristic_result=heuristic_result,
                    use_full_persona=args.full_persona,
                    verbose_context=args.verbose_context,
                )

                if args.enable_review and review_model and should_run_review(
                    llm_result=result,
                    heuristic_result=heuristic_result,
                    low=args.review_low,
                    high=args.review_high,
                    review_on_disagreement=args.review_on_disagreement,
                ):
                    try:
                        reviewed = llm_review_decide(
                            session_id=session_id,
                            model=review_model,
                            snapshot=snapshot,
                            first_result=result,
                            heuristic_result=heuristic_result,
                        )
                        result = reviewed
                        review_count += 1
                    except Exception as review_exc:
                        print(f"[WARN] Review pass failed for {user_id}: {review_exc}")
            except Exception as exc:
                # Keep the full run alive if one model call fails.
                result = heuristic_decision(snapshot)
                result["reasons"] = [
                    "LLM response failed; fallback heuristic used.",
                    f"Error: {exc}",
                ]
                fallback_count += 1
                print(f"[WARN] Fallback heuristic used for {user_id}: {exc}")

        decision = str(result.get("decision", "NORMAL")).upper()
        confidence = result.get("confidence", "n/a")
        reasons = result.get("reasons", [])

        if decision == "ANOMALOUS":
            flagged_ids.append(user_id)

        print(
            f"[{user_id}] decision={decision} confidence={confidence} "
            f"reasons={'; '.join(reasons[:2]) if reasons else 'n/a'}"
        )

    write_output(output_path, flagged_ids)

    print("\nRun complete.")
    print(f"Flagged citizens: {len(flagged_ids)}")
    if not args.dry_run:
        print(f"Second-pass reviews used: {review_count}")
        print(f"Fallbacks used: {fallback_count}")
    print(f"Output file: {output_path.resolve()}")
    print("Submit this file in the sandbox modal and paste the same session ID shown above.")


if __name__ == "__main__":
    main()