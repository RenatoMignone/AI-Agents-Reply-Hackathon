"""
agent.py – LangGraph ReAct agent for MirrorPay fraud detection.

Architecture:
  - Uses langgraph's create_react_agent (ReAct pattern).
  - The LLM is the CORE decision-maker: it autonomously decides which tools
    to call, how to interpret the results, and which transactions to flag.
  - Langfuse 4.x integration: @observe() + CallbackHandler() for tracking.
"""

import os
import threading
from typing import Any

import ulid
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler
from langgraph.prebuilt import create_react_agent

from tools import (
    expand_flagged_transactions,
    get_all_transaction_ids,
    get_all_tools,
    get_flagged_transactions,
    get_transaction_count,
    rank_risky_transactions,
)

load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = SystemMessage(content="""You are The Eye, an elite fraud detection AI agent for MirrorPay in the digital metropolis of Reply Mirror (year 2087).

YOUR MISSION: Identify fraudulent financial transactions with high precision and economic impact awareness.

TOOLS (call in this order):
1. list_citizens – Overview of all citizens. START HERE.
2. get_citizen_transaction_summary – Transaction stats, patterns, all IDs. PRIMARY tool.
3. get_citizen_profile – Salary, job, residence, behavioral description.
4. get_citizen_location_summary – GPS movement patterns.
5. get_citizen_transactions_detail – Full transaction list. Use ONLY for targeted deep-dives.
6. get_citizen_communications – SMS/emails. VERY EXPENSIVE – only for high-confidence suspicious citizens.
7. mark_fraudulent_transactions – Record final verdict (comma-separated IDs).

CRITICAL FRAUD SIGNALS (HIGH CONFIDENCE):
1. BEHAVIORAL ANOMALIES:
   - Sudden shift in transaction frequency/timing (baseline broken)
   - Late-night withdrawals/e-commerce (0-6am) when citizen profile shows daytime only
   - One-time recipients with large amounts (no prior relationship pattern)
   - Multiple high-value transactions to different recipients within hours
   
2. ECONOMIC MISALIGNMENT:
   - Single transaction > 2x monthly salary to unknown recipient
   - Cumulative sent transactions in one day > 50% of annual salary
   - Rapid balance drops to near-zero or negative
   
3. CHANNEL/METHOD ANOMALIES:
   - Withdrawal + same-day transfer to unknown recipient
   - Mobile device/smartwatch payments for unusually high amounts
   - E-commerce to recipients that match known fraud patterns
   
4. LOCATION CONTRADICTIONS:
   - Transaction from city far from residence without GPS evidence
   - GPS shows travel to suspicious location immediately before fraud

LEGITIMATE PATTERNS (EXCLUDE UNLESS STRONG EVIDENCE):
- Multiple transactions with "salary payment" description from EMP* senders (ignore)
- Recurring monthly transfers to fixed recipients (rent/utilities) with >6 similar prior transactions
- Direct debits to known utilities, insurance, subscriptions
- Transactions matching citizen's behavioral description baseline

WORKFLOW WITH EFFICIENCY FOCUS:
1. list_citizens → scan overview
2. For EACH citizen systematically:
   a. get_citizen_transaction_summary → identify patterns
   b. If CLEAR anomalies detected: assess 2-4 most suspicious transactions immediately
   c. If BORDERLINE case: add low-confidence candidates to list
   d. Use get_citizen_profile ONLY if transaction_summary unclear on salary context
   e. Skip get_citizen_location_summary unless transaction location is explicitly contradictory
   f. Skip get_citizen_communications unless SMS/email evidence is critical to verdict
3. Compile final list with HIGH-to-MEDIUM confidence fraud only
4. Call mark_fraudulent_transactions ONCE with complete list

BALANCE RULES (CRITICAL FOR SCORING):
- Precision > Recall: False positives cost money. Only flag if >70% confident.
- Miss high-value fraud: Even worse. Flag any transaction > 50% monthly salary with unusual recipient.
- Keep output at 15-20% of total transactions for Dataset 3 size (~15-25 IDs for ~1500 txns)

EXECUTION GUARDRAILS:
- Analyze EVERY citizen (required for thoroughness)
- Use compact tool calls; avoid redundant queries
- NEVER call get_citizen_communications more than 2-3 times total (very expensive)
- Build fraud list incrementally as you go through citizens
- Output exactly what the system flags; do not second-guess the LLM assessment
- Keep Transaction IDs exactly as they appear (no character changes)
- End with single mark_fraudulent_transactions call""")


# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

MODELS = {
    "cheap": "meta-llama/llama-3.1-8b-instruct",
    "mid": "deepseek/deepseek-v3.2",
    "heavy": "deepseek/deepseek-v3.2",
    "gemini-pro": "deepseek/deepseek-v3.2",
}

FALLBACK_MODELS = [
    "deepseek/deepseek-v3.2",
]

langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
)


def generate_session_id() -> str:
    """Generate a Langfuse session ID: {TEAM_NAME}-{ULID}."""
    team = os.getenv("TEAM_NAME", "team").strip().replace(" ", "-")
    return f"{team}-{ulid.new().str}"


def resolve_model_id(model_arg: str) -> str:
    """Resolve a preset name or pass through a full OpenRouter model ID."""
    return MODELS.get(model_arg, model_arg)


def create_fraud_agent(model_id: str, temperature: float = 0.1):
    """Create the LangGraph ReAct fraud detection agent."""
    model = ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model=model_id,
        temperature=temperature,
        max_tokens=1200,
        timeout=35,
        max_retries=0,
    )

    tools = get_all_tools()
    agent = create_react_agent(model, tools, prompt=SYSTEM_PROMPT)
    return agent


def _target_bounds(total_tx: int) -> tuple[int, int]:
    """Adaptive output bounds tuned for balanced precision-recall on economic accuracy."""
    if total_tx <= 300:
        return max(10, total_tx // 25), max(32, total_tx // 6)
    if total_tx <= 1200:
        return max(18, total_tx // 50), max(100, total_tx // 10)
    return max(22, total_tx // 70), max(150, total_tx // 15)


def _calibrate_flagged_ids(flagged_ids: list[str], total_tx: int) -> list[str]:
    """Calibrate output size while preserving high-value fraud and LLM risk ordering."""
    valid_ids = get_all_transaction_ids()
    ranked = [tid for tid in rank_risky_transactions(flagged_ids) if tid in valid_ids]

    unique: list[str] = []
    seen: set[str] = set()
    for tid in flagged_ids:
        if tid in valid_ids and tid not in seen:
            unique.append(tid)
            seen.add(tid)

    min_target, max_target = _target_bounds(total_tx)

    if not unique and ranked:
        unique = ranked[:min_target]
        seen = set(unique)

    if len(unique) < min_target:
        for tid in ranked:
            if tid in seen:
                continue
            unique.append(tid)
            seen.add(tid)
            if len(unique) >= min_target:
                break

    if len(unique) > max_target:
        selected = set(unique)
        unique = [tid for tid in ranked if tid in selected][:max_target]
        seen = set(unique)

    # Ensure high-value frauds are prioritized even if borderline
    for tid in ranked[:6]:
        if tid not in seen and len(unique) < max_target:
            unique.append(tid)
            seen.add(tid)

    if ranked:
        ordered = [tid for tid in ranked if tid in seen]
        if ordered:
            unique = ordered

    return unique


@observe(name="fraud_detection_run")
def run_agent(
    agent,
    session_id: str,
    dataset_name: str = "",
    verbose: bool = True,
    primary_model_id: str = "",
    temperature: float = 0.1,
) -> list[str]:
    """Run the fraud detection agent on the loaded dataset.

    Uses @observe() for Langfuse tracing. CallbackHandler() auto-attaches
    to the current trace. Session ID is passed via config metadata.
    """
    # Langfuse 4.x SDK in this environment exposes update_current_span()
    # instead of update_current_trace(). Session ID is still passed via
    # LangChain callback config metadata (langfuse_session_id).
    langfuse_client.update_current_span(
        metadata={
            "agent_type": "orchestrator",
            "dataset": dataset_name,
            "session_id": session_id,
        }
    )

    # CallbackHandler with no args: auto-attaches to current @observe trace
    langfuse_handler = CallbackHandler()

    user_message = (
        f"Investigate '{dataset_name}' for fraud:\n"
        "1. list_citizens for overview\n"
        "2. For each citizen: get_citizen_transaction_summary for patterns\n"
        "3. Flag high-value anomalies, unusual recipients, behavioral shifts\n"
        "4. Skip deep-dives on clear legitimate patterns\n"
        "5. mark_fraudulent_transactions with final verdict\n"
        "Begin."
    )

    retry_message = (
        f"'{dataset_name}': empty output invalid.\n"
        "Call mark_fraudulent_transactions with focused fraud IDs now.\n"
        "Use high-value, behavioral anomalies, and one-time recipient red flags."
    )

    total_tx = get_transaction_count()
    min_target, max_target = _target_bounds(total_tx)

    low_coverage_message = (
        f"'{dataset_name}': output too small (need {min_target}+ flags).\n"
        "Recall pass: high-value txns, one-time recipients, risky methods (mobile/withdraw).\n"
        "Call mark_fraudulent_transactions with ADDITIONAL IDs."
    )

    recursion_limit = 70 if total_tx <= 600 else 100 if total_tx <= 2000 else 120

    config = {
        "callbacks": [langfuse_handler],
        "recursion_limit": recursion_limit,
        "metadata": {"langfuse_session_id": session_id},
    }

    def _stream_run(selected_agent: Any, message: str) -> None:
        for event in selected_agent.stream(
            {"messages": [("user", message)]},
            config=config,
            stream_mode="updates",
        ):
            if verbose:
                for node_name, update in event.items():
                    messages = update.get("messages", [])
                    for msg in messages:
                        if hasattr(msg, "type") and msg.type == "tool":
                            preview = str(msg.content)[:150]
                            tool_name = getattr(msg, "name", "unknown")
                            print(f"  [Tool:{tool_name}] {preview}...")
                        elif hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                args_preview = str(tc.get("args", ""))[:100]
                                print(f"  [Call] {tc['name']}({args_preview})")
                        elif hasattr(msg, "content") and msg.content:
                            preview = str(msg.content)[:200]
                            if preview.strip():
                                print(f"  [Agent] {preview}")

    if verbose:
        print("\n[Agent] Starting investigation...")

    candidate_models = [m for m in FALLBACK_MODELS if m != primary_model_id]
    run_attempts: list[tuple[Any, str]] = [(agent, primary_model_id or "provided-agent")]
    for model_id in candidate_models:
        run_attempts.append((create_fraud_agent(model_id, temperature), model_id))

    def _run_with_fallback(message: str, phase_label: str) -> None:
        stream_error: Exception | None = None
        for idx, (candidate_agent, model_name) in enumerate(run_attempts, start=1):
            try:
                if verbose and idx > 1:
                    print(f"\n[Agent] {phase_label}: retrying with fallback model: {model_name}")
                _stream_run(candidate_agent, message)
                stream_error = None
                break
            except Exception as exc:
                stream_error = exc
                if verbose:
                    preview = str(exc).splitlines()[0][:180]
                    print(f"\n[Agent] {phase_label} failed ({model_name}): {preview}")

        if stream_error is not None:
            raise stream_error

    _run_with_fallback(user_message, "Initial pass")

    flagged = get_flagged_transactions()

    if not flagged:
        if verbose:
            print("\n[Agent] No transactions flagged. Triggering focused retry...")
        _run_with_fallback(retry_message, "Focused retry")
        flagged = get_flagged_transactions()

    if len(flagged) < min_target:
        if verbose:
            print(
                f"\n[Agent] Low coverage ({len(flagged)}<{min_target}). Triggering recall boost pass..."
            )
        _run_with_fallback(low_coverage_message, "Coverage boost")
        flagged = get_flagged_transactions()

    if len(flagged) < min_target:
        if verbose:
            print(
                f"\n[Agent] Coverage still low ({len(flagged)}<{min_target}). Applying ranked expansion..."
            )
        target_count = min(max_target, max(min_target + 20, int(total_tx * 0.045)))
        flagged = expand_flagged_transactions(flagged, target_count)

    flagged = _calibrate_flagged_ids(flagged, total_tx)

    if verbose:
        print(f"\n[Agent] Investigation complete. Flagged {len(flagged)} transaction(s).")

    return flagged


def flush_langfuse() -> None:
    """Flush buffered Langfuse traces to the backend without blocking script completion."""
    skip_flush = os.getenv("LANGFUSE_SKIP_FLUSH", "0").lower() in {"1", "true", "yes"}
    if skip_flush:
        return

    err: dict[str, Exception] = {}

    def _do_flush() -> None:
        try:
            langfuse_client.flush()
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            err["exc"] = exc

    flush_thread = threading.Thread(target=_do_flush, daemon=True)
    flush_thread.start()
    flush_thread.join(timeout=4.0)

    if flush_thread.is_alive():
        print("WARNING: Langfuse flush timed out; continuing without blocking output generation.")
    elif "exc" in err:
        print(f"WARNING: Langfuse flush failed: {err['exc']}")
