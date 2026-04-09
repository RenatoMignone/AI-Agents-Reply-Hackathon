# Submission Tracking - Sandbox
# Reply AI Agent Challenge 2026

This file tracks every submission made during sandbox training.
Record each attempt here regardless of score. This is your experiment log.

---

## How to record a submission

Fill one entry per submission using the template below.
For the session ID: generate it with generate_session_id() before running your agent, print it, and paste it here.
For training submissions: you can submit many times. Record each one.
For evaluation submissions: only one is allowed per level. Mark clearly.

---

## Entry template

```
Level: [1 / 2 / 3]
Dataset type: [training / evaluation]
Date: YYYY-MM-DD HH:MM
Session ID: {TEAM_NAME}-{ULID}
Model used: openrouter model ID
Score received: [value shown on platform]
Citizens flagged: [list of user_ids flagged in output]
Notes: [what changed from previous attempt, observations, what worked/did not]
```

---

## Submissions

(add entries below as you run them)

---

## Langfuse Integration Reference

All submissions require a Langfuse session ID in the upload modal. Without it the submission is invalid.

Session ID format: {TEAM_NAME}-{ULID}
Example: myteam-01HXYZ123ABC456DEF789GHI

The session ID groups all LLM calls from a single run together in the Langfuse dashboard.
Get your keys from the sandbox platform by clicking "View my Keys".

Required environment variables:

```
OPENROUTER_API_KEY=your-key
LANGFUSE_PUBLIC_KEY=pk-your-key
LANGFUSE_SECRET_KEY=sk-your-key
LANGFUSE_HOST=https://challenges.reply.com/langfuse
TEAM_NAME=your-team-name
```

Minimal code to generate a session ID and run a traced call:

```python
import os, ulid
from dotenv import load_dotenv, find_dotenv
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler
from langchain_core.messages import HumanMessage

load_dotenv(find_dotenv())  # finds .env in repo root from any subfolder

langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")
)

def generate_session_id():
    return f"{os.getenv('TEAM_NAME', 'team')}-{ulid.new().str}"

@observe()
def run_llm_call(session_id, model, prompt):
    langfuse_client.update_current_trace(session_id=session_id)
    handler = CallbackHandler()
    return model.invoke([HumanMessage(content=prompt)], config={"callbacks": [handler]}).content

session_id = generate_session_id()
print(f"Session ID (paste this into the upload modal): {session_id}")
# ... run your agent logic ...
langfuse_client.flush()
```

To inspect costs and token usage for a past session:

```python
from datetime import datetime
from collections import defaultdict

def get_trace_info(session_id):
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

    return {"counts": dict(counts), "costs": dict(costs), "time": total_time}

# Usage:
# info = get_trace_info("myteam-01HXYZ...")
# print(info)
```

Full Langfuse integration guide: ../../02_AI_Agents_Challenge/00_How_It_Works/api_guidelines.md