# API and Langfuse Guidelines
# Source: https://cdn.reply.com/documents/challenges/02_26/api_guidelines.html

All costs are tracked via Langfuse session IDs. Score is based on output files only - Langfuse is for tracking and validation.

Python 3.14 is incompatible with Langfuse. Use Python 3.10-3.13.

The .env file lives in the repository root. All scripts must use load_dotenv(find_dotenv()) so that python-dotenv traverses up the directory tree to find it regardless of which subfolder the script runs from.

---

## Token concepts

- 1 token = ~4 chars / ~0.75 words. 1000 tokens = ~750 words
- Input tokens: prompt + system prompt + conversation history
- Output tokens: model response
- Cache tokens: optional, for repeated queries

---

## Install

```bash
pip install langchain langchain-openai langfuse python-dotenv ulid-py langgraph
```

---

## .env

```
OPENROUTER_API_KEY=your-api-key-here
LANGFUSE_PUBLIC_KEY=pk-your-public-key-here
LANGFUSE_SECRET_KEY=sk-your-secret-key-here
LANGFUSE_HOST=https://challenges.reply.com/langfuse
TEAM_NAME=your-team-name
```

---

## Model setup

```python
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv(find_dotenv())  # finds .env in repo root regardless of working directory

model_id = "gpt-4o-mini"
model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model=model_id,
    temperature=0.1,  # Set low for deterministic classification
    max_tokens=1000,
)
```

---

## Langfuse integration - full pattern

How it works:
- @observe() creates a Langfuse trace on each call
- update_current_trace(session_id=...) tags the trace with the session ID so all calls in a run are grouped
- CallbackHandler() created inside the decorated function auto-attaches to the current trace
- Token usage, costs, and latency are captured automatically

```python
import os, ulid
from dotenv import load_dotenv, find_dotenv
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler
from langchain_core.messages import HumanMessage

load_dotenv(find_dotenv())  # finds .env in repo root regardless of working directory

langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")
)

def generate_session_id():
    # Format: {TEAM_NAME}-{ULID}
    return f"{os.getenv('TEAM_NAME', 'tutorial')}-{ulid.new().str}"

def invoke_langchain(model, prompt, langfuse_handler):
    messages = [HumanMessage(content=prompt)]
    return model.invoke(messages, config={"callbacks": [langfuse_handler]}).content

@observe()
def run_llm_call(session_id, model, prompt):
    langfuse_client.update_current_trace(session_id=session_id)
    langfuse_handler = CallbackHandler()  # auto-attaches to current trace
    return invoke_langchain(model, prompt, langfuse_handler)

# Usage
session_id = generate_session_id()
response = run_llm_call(session_id, model, "Your prompt here")
langfuse_client.flush()  # always flush - traces are buffered, not sent instantly
```

CallbackHandler captures automatically: inputs/outputs, token usage (in/out/cache), cost, latency, model metadata.

---

## Trace viewer helper

Use this to inspect session costs and usage against the $40/$120 budget.

```python
from datetime import datetime
from collections import defaultdict

def get_trace_info(session_id: str):
    # Returns: {counts: {model->n}, costs: {model->$}, time: float, input: str, output: str}
    traces, page = [], 1
    while True:
        resp = langfuse_client.api.trace.list(session_id=session_id, limit=100, page=page)
        if not resp.data: break
        traces.extend(resp.data)
        if len(resp.data) < 100: break
        page += 1
    if not traces: return None

    observations = []
    for trace in traces:
        detail = langfuse_client.api.trace.get(trace.id)
        if detail and hasattr(detail, "observations"):
            observations.extend(detail.observations)
    if not observations: return None

    sorted_obs = sorted(observations,
        key=lambda o: o.start_time if hasattr(o, "start_time") and o.start_time else datetime.min)

    counts, costs, total_time = defaultdict(int), defaultdict(float), 0.0
    for obs in observations:
        if hasattr(obs, "type") and obs.type == "GENERATION":
            m = getattr(obs, "model", "unknown") or "unknown"
            counts[m] += 1
            if hasattr(obs, "calculated_total_cost") and obs.calculated_total_cost:
                costs[m] += obs.calculated_total_cost
            if hasattr(obs, "start_time") and hasattr(obs, "end_time"):
                if obs.start_time and obs.end_time:
                    total_time += (obs.end_time - obs.start_time).total_seconds()

    return {
        "counts": dict(counts), "costs": dict(costs), "time": total_time,
        "input": str(sorted_obs[0].input)[:100] if sorted_obs and hasattr(sorted_obs[0], "input") and sorted_obs[0].input else "",
        "output": str(sorted_obs[-1].output)[:100] if sorted_obs and hasattr(sorted_obs[-1], "output") and sorted_obs[-1].output else "",
    }

def print_results(info):
    if not info: print("\nNo traces found.\n"); return
    print("\nTrace Count by Model:")
    for m, n in info["counts"].items(): print(f"  {m}: {n}")
    print("\nCost by Model:")
    total = 0.0
    for m, c in info["costs"].items():
        print(f"  {m}: ${c:.6f}"); total += c
    if total > 0: print(f"  Total: ${total:.6f}")
    print(f"\nTotal Time: {info['time']:.2f}s")
    if info["input"]: print(f"\nInitial Input:\n  {info['input']}")
    if info["output"]: print(f"\nFinal Output:\n  {info['output']}")

# Usage:
# info = get_trace_info("TEAMNAME-01HXYZ...")
# print_results(info)
```

---

## Advanced: metadata tagging

```python
# Tag trace with custom metadata for filtering in the dashboard
langfuse_client.update_current_trace(metadata={"agent_type": "orchestrator", "dataset": "1"})
```

---

## Best practices

1. Always set session IDs - groups all costs under one queryable session for the challenge validators
2. @observe() + CallbackHandler() - CallbackHandler must be created inside the decorated function; wrong nesting breaks auto-attach
3. langfuse_client.flush() - call after every batch; traces are buffered, not sent instantly
4. ULID session IDs - time-sortable, collision-free format: {TEAM_NAME}-{ULID}
5. Optimize prompts - fewer input tokens = lower cost = better score
6. Multi-agent model sizing - use capable models for decisions, cheap models for auxiliary tasks
7. Monitor per session - use get_trace_info() after each run to track cumulative cost against the $40/$120 budget
