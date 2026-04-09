# Sandbox Training Guide
# How to develop, test, and submit solutions for the Reply AI Agent Challenge sandbox

---

## Overview

This guide walks through the complete workflow from reading the problem to submitting a scored solution.
Follow it for each level (1, 2, 3) independently. Start with level 1 since it has the fewest citizens and is easiest to reason about.

---

## Step 1: Read the problem statement

Open 00_Sandbox_Sample_Material/Sandbox_2026_V3.pdf.
Read it entirely before writing any code.
Understand:
- The exact output format required (what fields, what order, what delimiter)
- The scoring criteria (count-based and economic/value-weighted accuracy)
- What "anomalous" means in this domain (deviating from expected citizen behavior)
- Any constraints on the agentic system (LLM must be the decision-maker, not a heuristic)

---

## Step 2: Understand the data for the target level

For each level, the data is in: 00_Sandbox_Sample_Material/Public_Levels/public_lev_N/public_lev_N/

Read in this order:

1. users.json - who the citizens are (ID, name, age via birth_year, job, home coordinates)
2. personas.md - narrative descriptions of each citizen's expected behavior, mobility, health patterns, and social life
3. status.csv - timestamped health events with PhysicalActivityIndex, SleepQualityIndex, EnvironmentalExposureLevel
4. locations.json - timestamped GPS coordinates for each citizen

The personas.md is the most important file for understanding what is "normal" for each person.
Your agent should compare observed data against the persona baseline to decide if something is anomalous.

Key signals to look for:
- EnvironmentalExposureLevel rising significantly over time
- PhysicalActivityIndex dropping from the expected range for that person
- SleepQualityIndex declining sharply
- Health event types escalating (routine -> specialist -> follow-up in tight succession)
- Location data showing unusual travel or conversely unusual confinement for someone who normally moves freely
- Gaps in expected events or sudden clustering of medical contacts

---

## Step 3: Set up your environment

From the 00_AI_Agents_Learning virtual environment or create a new one:

```bash
cd /path/to/01_AI_Agents_Training/01_Sandbox_Implementations
python3 -m venv .venv
source .venv/bin/activate
pip install langchain langchain-openai langfuse python-dotenv ulid-py langgraph
```

Create a .env file:

```
OPENROUTER_API_KEY=your-sandbox-key
LANGFUSE_PUBLIC_KEY=pk-your-key
LANGFUSE_SECRET_KEY=sk-your-key
LANGFUSE_HOST=https://challenges.reply.com/langfuse
TEAM_NAME=your-team-name
```

Get your keys from the sandbox challenge page by clicking "View my Keys".

In sandbox mode, no credits are provided. Use free models on OpenRouter:
- meta-llama/llama-3.1-8b-instruct (free)
- google/gemma-2-9b-it (free)
- mistralai/mistral-7b-instruct-v0.1 (free)

Check https://openrouter.ai/models?q=free for the current free model list.

---

## Step 4: Build your agentic system

Create your solution in 01_Sandbox_Implementations/. Suggested structure:

```
01_Sandbox_Implementations/
  solution_lev1.py    - solution for level 1
  solution_lev2.py    - solution for level 2
  solution_lev3.py    - solution for level 3
  tools.py            - shared tool functions
  agent.py            - shared agent setup
  utils.py            - data loading helpers
```

Architecture requirements (from the challenge rules):
- The LLM must be the central orchestrator and decision-maker
- Deterministic tools are allowed and encouraged (data parsing, statistics, filtering), but the LLM must control them
- Using only heuristics without LLM reasoning in the loop is not acceptable
- Multi-agent systems are allowed and encouraged (orchestrator + specialist agents)

Suggested architecture for this problem:

```
Orchestrator LLM
  |
  |-- Data Analyst Agent (reads and summarizes data for each citizen)
  |-- Anomaly Detector Agent (compares data against persona baseline)
  |-- Decision Agent (makes final flag/no-flag decision and produces output)
```

Minimal working pattern using the required Langfuse integration:

```python
import os, ulid, json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler

load_dotenv()

model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="meta-llama/llama-3.1-8b-instruct",
    temperature=0.1,
)

langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")
)

def generate_session_id():
    return f"{os.getenv('TEAM_NAME', 'team')}-{ulid.new().str}"

@observe()
def analyze_citizen(session_id, citizen_id, persona, status_events, location_summary):
    langfuse_client.update_current_trace(session_id=session_id)
    handler = CallbackHandler()
    
    prompt = f"""
You are analyzing citizen {citizen_id} for behavioral anomalies.

EXPECTED BEHAVIOR (persona):
{persona}

HEALTH STATUS EVENTS (chronological):
{status_events}

LOCATION SUMMARY:
{location_summary}

Based on the persona, are there anomalies in the health or location data?
Respond with: ANOMALOUS or NORMAL, followed by a brief explanation.
"""
    response = model.invoke([HumanMessage(content=prompt)], config={"callbacks": [handler]})
    return response.content

# Main loop
session_id = generate_session_id()
print(f"Session ID: {session_id}")

# Load data, loop over citizens, call analyze_citizen for each
# Collect flagged citizens, write output file in required format

langfuse_client.flush()
```

---

## Step 5: Generate and validate the output file

Read Sandbox_2026_V3.pdf carefully for the exact output format.
The output is a UTF-8 plain text file listing the flagged transaction/citizen IDs.

Before submitting:
- Verify the file is correctly encoded (UTF-8)
- Verify the format matches the problem statement exactly
- Test with the training dataset first to see a score before using the evaluation slot

---

## Step 6: Submit to the training dataset

On the sandbox challenge page:
1. Go to the Training dataset section on the left side of the page
2. Click the upload area for the appropriate level (public_lev_1.zip, public_lev_2.zip, public_lev_3.zip)
3. Upload your output .txt file
4. Wait for the score to appear
5. Record the result in Submission_Tracking.md

You can repeat this as many times as needed. Each submission shows a score.
Use the score to iterate on your approach.

---

## Step 7: Iterate using the score

If the score is lower than expected:
- Check if the output format is exactly correct (wrong format = zero score)
- Re-examine the personas for citizens you did not flag
- Look at the health event escalation patterns more carefully
- Consider whether your agent is using the location data effectively
- Try adding more specific instructions to the agent about what to look for

Useful debugging steps:
- Run get_trace_info(session_id) to check token usage per run (see api_guidelines.md)
- Print what the LLM concludes for each citizen before writing the output file
- Compare against the persona descriptions manually for a few borderline cases

---

## Step 8: Submit to the evaluation dataset

Only when you are satisfied with your training score:
1. Go to the Evaluation dataset section on the right side of the page
2. Upload your output .txt file for the correct level
3. After upload, you will be prompted to upload your source code as a .zip
4. The zip must contain: all .py files, .env.example (not real .env), requirements or pip list, README with run instructions
5. Submit and record in Submission_Tracking.md

This is a one-way action. You cannot re-submit the evaluation for the same level.

---

## Submission Tracking

Record every submission in 00_Sandbox_Sample_Material/Submission_Tracking.md.
Include: level, dataset type (training/eval), session ID, score, model used, notes on what changed.

---

## Token budget in sandbox

No credits are provided in sandbox. Use only free OpenRouter models.
In the real challenge: $40 for datasets 1-3, $120 more for datasets 4-5 after submitting eval for 1-3.
Practice being efficient: minimize prompt length, avoid re-loading data repeatedly, batch citizen analysis where possible.
