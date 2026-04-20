"""
agent.py – LangGraph ReAct agent for MirrorPay fraud detection.

Architecture:
  - Uses langgraph's create_react_agent (ReAct pattern).
  - The LLM is the CORE decision-maker: it autonomously decides which tools
    to call, how to interpret the results, and which transactions to flag.
  - Langfuse 4.x integration: @observe() + CallbackHandler() for tracking.
"""

import os

import ulid
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler
from langgraph.prebuilt import create_react_agent

from tools import get_all_tools, get_flagged_transactions

load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = SystemMessage(content="""You are The Eye, an elite fraud detection AI agent for MirrorPay in the digital metropolis of Reply Mirror (year 2087).

YOUR MISSION: Identify fraudulent financial transactions. Output specific Transaction IDs.

TOOLS (call them in this order):
1. list_citizens – Overview of all citizens. START HERE.
2. get_citizen_transaction_summary – Transaction stats, patterns, all IDs. PRIMARY tool.
3. get_citizen_profile – Salary, job, residence, behavioral description.
4. get_citizen_transactions_detail – Full transaction list. Use for closer inspection.
5. get_citizen_location_summary – GPS movement patterns.
6. get_citizen_communications – SMS/emails. EXPENSIVE – suspicious citizens only.
7. mark_fraudulent_transactions – Record final verdict (comma-separated IDs).

WORKFLOW:
1. list_citizens → see all citizens.
2. For EACH citizen: get_citizen_transaction_summary → review patterns.
3. RED FLAGS:
   - Amounts inconsistent with salary
   - Late-night bursts, sudden frequency changes
   - Transactions far from residence
   - Unusual types/methods for this citizen's profile
   - Rapid balance drops or negative balances
   - Payments to unknown recipients
   - Suspicious descriptions
4. For suspicious citizens: get_citizen_profile + get_citizen_location_summary.
5. ONLY for highly suspicious: get_citizen_communications (phishing evidence).
6. After ALL citizens analyzed: mark_fraudulent_transactions with comma-separated IDs.

RULES:
- Flag TRANSACTIONS, not citizens. Mix of legit and fraud is normal.
- Salary from EMP* = almost always legit.
- Rent payments with "Rent payment" description = typically legit.
- High-value fraud is MORE COSTLY to miss.
- Keep false positives low.
- Consider citizen description for behavioral baseline.
- Cross-reference GPS with transaction locations.

BE THOROUGH: Analyze EVERY citizen.
BE EFFICIENT: Deep-dive only when summary shows clear anomalies.

IMPORTANT:
- Use compact reasoning and avoid redundant narration.
- Use get_citizen_communications only for borderline/high-risk cases.
- End by calling mark_fraudulent_transactions with a comma-separated ID list.""")


# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

MODELS = {
    "cheap": "meta-llama/llama-3.1-8b-instruct",
    "mid": "deepseek/deepseek-v3.2",
    "heavy": "deepseek/deepseek-v3.2",
    "gemini-pro": "google/gemini-2.5-flash",
}

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
        max_tokens=1800,
    )

    tools = get_all_tools()
    agent = create_react_agent(model, tools, prompt=SYSTEM_PROMPT)
    return agent


@observe(name="fraud_detection_run")
def run_agent(
    agent,
    session_id: str,
    dataset_name: str = "",
    verbose: bool = True,
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
        f"Analyze the '{dataset_name}' dataset for fraudulent transactions.\n"
        "Start by listing all citizens, then systematically investigate each one.\n"
        "For each citizen, check their transaction patterns against their profile.\n"
        "When done with ALL citizens, call mark_fraudulent_transactions with your final verdict.\n"
        "Begin now."
    )

    retry_message = (
        f"Re-run analysis for '{dataset_name}'.\n"
        "Your previous result flagged 0 transactions, which is invalid on the platform.\n"
        "You must call mark_fraudulent_transactions with a focused, non-empty set of the"
        " most suspicious transaction IDs based on your tool-driven investigation.\n"
        "Keep false positives low and do not flag all transactions."
    )

    config = {
        "callbacks": [langfuse_handler],
        "recursion_limit": 150,
        "metadata": {"langfuse_session_id": session_id},
    }

    def _stream_run(message: str) -> None:
        for event in agent.stream(
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

    _stream_run(user_message)
    flagged = get_flagged_transactions()

    if not flagged:
        if verbose:
            print("\n[Agent] No transactions flagged. Triggering focused retry...")
        _stream_run(retry_message)
        flagged = get_flagged_transactions()

    if verbose:
        print(f"\n[Agent] Investigation complete. Flagged {len(flagged)} transaction(s).")

    return flagged


def flush_langfuse() -> None:
    """Flush buffered Langfuse traces to the backend."""
    langfuse_client.flush()
