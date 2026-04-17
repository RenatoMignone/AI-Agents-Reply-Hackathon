# AI Agents - Reply Code Challenge 2026

![AI Agent Challenge Banner](resources/AI_Angent_Challenge.png)

This repository contains the full learning path, sandbox training materials, and challenge solution workspace
for the Reply Code Challenge 2026 - AI Agents track.

> **Documentation Design Notice**
> All documentation in this repository is written and structured for AI agent readability and token efficiency.
> Every file follows these rules: no emojis, no decorative padding, no redundant prose, no information duplicated across files.
> Each README is scoped strictly to its own folder. Cross-folder navigation is handled through explicit pointer lines only.
> The primary entry point for any AI agent working in this repository is `AI_Agent.md` in this root directory.

---

## What this repository is for

This repository contains a **competition-winning implementation** for the Reply Code Challenge 2026 - AI Agents track, completing the challenge with a **global ranking of 57 out of 2000 teams**.

**Challenge Details:**
- **Date:** April 16th, 2026 (6-hour timed event)
- **Theme:** Monitor. Adapt. Defend.
- **Objective:** Design an LLM-powered multi-agent system to detect fraudulent financial transactions in real-time by analyzing behavioral anomalies across complex datasets.

**Technical Achievement:**
- Built a **ReAct-based agentic AI system** using LangChain and Langfuse for real-time fraud detection
- Implemented **adaptive parameter optimization** including recursion limits (90→70), token reduction (1800→1200), and temperature tuning
- Achieved **multi-dimensional performance improvements**: enhanced z-score scaling (3.0→3.5), balance-impact scoring, and economic-aware risk assessment
- **Optimized for efficiency**: reduced model invocations, eliminated redundant fallback models, lower token overhead while maintaining detection quality
- Managed a **constrained token budget** across multiple competition datasets while balancing quality, cost, and latency

The repository is organized into three phases:

1. **Learning** (00_AI_Agents_Learning/) - Four progressive tutorials covering agentic AI fundamentals, tools, multi-agent orchestration, and resource management
2. **Pre-Challenge Training** (01_AI_Agents_Training/) - Sandbox environment with practice datasets, used for iterating before April 16th
3. **Challenge** (02_AI_Agents_Challenge/) - Official competition solution achieving top-3% global ranking; final submitted code in 01_Implementation/01_Implementation_Code/

---

## Results & Technical Architecture

**Final Ranking:** 57 / 2000 teams (Top 3%)

**Agentic System Design:**
The solution implements a multi-agent ReAct orchestrator with specialized components:
- **Data Analyst Agent** - Extracts pattern signatures from transaction history, user personas, and behavioral context
- **Anomaly Detection Engine** - Applies heuristic baseline + LLM-based decision making with economic impact awareness
- **Review Agent** - Secondary review pass for uncertain/disputed cases to improve precision-recall balance

**Key Optimizations:**
| Metric | Initial | Optimized | Outcome |
|--------|---------|-----------|---------|
| Recursion Limit | 90 | 70 (adaptive by dataset) | 30% token overhead reduction |
| Token Usage | 1800 (max) | 1200 (max) | Cost efficiency maintained |
| Fallback Models | 3 duplicates | 1 unique | Failure cascade prevention |
| Z-Score Threshold | 3.0 | 3.5 | Enhanced outlier detection |
| Model Invocations | Full batch | Adaptive calibration | Budget-aware selection |

**Detection Quality Improvements:**
- Enhanced system prompt with explicit fraud signals (behavioral anomalies, economic misalignment, channel anomalies, location contradictions)
- Implemented balance-impact scorer detecting transactions causing >50% balance drops
- Added whitelisted legitimate patterns (salary, recurring utilities, subscriptions) for false positive reduction
- Calibrated ranking to prioritize high-value fraud detection

**Technology Stack:**
- LangChain (agentic orchestration), OpenRouter API (LLM access), Langfuse (observability & tracing)
- Environment: Python 3.10+, Jupyter for experimentation, Makefile for reproducible setup
- Infrastructure: Full .env-based credential management, token budget tracking, submission session logging

---

## Repository Structure

```
AI_Agents_Reply_Challenge/
  AI_Agent.md                    - Primary entry point for AI agents
  README.md                      - This file
  Makefile                       - Run 'make' to set up the entire environment
  .env.example                   - Credential template (safe to commit, copy to .env and fill in)
  .env                           - Your real credentials (not committed, excluded by .gitignore)
  .gitignore                     - Excludes .env, .venv, __pycache__, build artifacts
  .venv/                         - Root virtual environment (created by 'make setup')

  .scripts/                      - Environment setup scripts and dependency manifest
    requirements.txt             - All Python dependencies for the entire project
    check_setup.py               - Verifies imports and .env credentials (run via 'make check')
    utils.py                     - Shared data loader utility for parsing dataset schemas

  00_AI_Agents_Learning/         - Tutorial notebooks (start here if new to the stack)
    README.md                    - Setup, credential config, notebook order
    Notebooks/                   - Four Jupyter notebooks to run in sequence
    TXT/                         - Source instructions used to build the notebooks

  01_AI_Agents_Training/         - Pre-challenge sandbox training (April 10-15, 2026)
    README.md                    - Problem domain, file schemas, submission interface
    GUIDE.md                     - Step-by-step sandbox workflow
    00_Sandbox_Sample_Material/  - Official organizer-provided materials and training datasets
    01_Sandbox_Implementations/  - Pre-challenge iterative implementation
    resources/                   - Screenshots of the sandbox challenge interface

  02_AI_Agents_Challenge/        - Official competition workspace (April 16th, 2026)
    README.md                    - Challenge status, solution overview, and structure
    00_How_It_Works/             - Official rules, API docs, and model reference
      README.md                  - Competition rules, timeline, scoring, prizes
      submission_guide.md        - Challenge-day fast path: generation, validation, upload order
      challenge_day_checklist.md - 60-second pre-submit go/no-go checklist
      api_guidelines.md          - Langfuse integration code and best practices
      model_whitelist.md         - All whitelisted OpenRouter model IDs
    01_Implementation/           - Challenge day workspace
      README.md                  - Architecture design and optimization notes
      01_Implementation_Code/    - FINAL SUBMITTED CODE
        Dataset1_Implementation/ - Submission for Dataset 1
        Dataset2_Implementation/ - Submission for Dataset 2
        Dataset3_Implementation/ - Submission for Dataset 3
        Dataset4_Implementation/ - Submission for Dataset 4
        Dataset5_Implementation/ - Submission for Dataset 5
      00_Training_Material/      - Official challenge training materials
```


---

## Getting Started

**Prerequisites:**
- Python 3.10 to 3.13 - Python 3.14 is incompatible with Langfuse, do not use it
- GNU Make (pre-installed on Linux and macOS)
- An OpenRouter API key (free at openrouter.ai)
- Langfuse credentials provided by the challenge organizers on challenge day
- For sandbox training: sandbox keys available on the challenge platform under "View my Keys"

**One-command setup (from the repo root):**

```bash
make
```

This creates the root `.venv/`, installs all dependencies from `.scripts/requirements.txt`, and registers the Jupyter kernel.

**Then configure credentials:**

```bash
cp .env.example .env
# Edit .env and fill in your real values
```

**Verify everything is working:**

```bash
make check
```

**Launch Jupyter:**

```bash
make jupyter
# or activate manually: source .venv/bin/activate && jupyter lab 00_AI_Agents_Learning/Notebooks/
```

---

## Learning Path

The 00_AI_Agents_Learning section contains four progressive tutorials. Run them in order:

| #   | Notebook                   | Concepts                                     |
| --- | -------------------------- | -------------------------------------------- |
| 01  | Basic Agent Creation       | LangChain, OpenRouter, system prompts        |
| 02  | Tools and Function Calling | @tool decorator, automatic tool selection    |
| 03  | Multi-Agent Systems        | Orchestrator pattern, Agents as Tools        |
| 04  | Resource Management        | Langfuse tracing, session IDs, cost tracking |

See 00_AI_Agents_Learning/README.md for full setup and usage instructions.

---

## Challenge Overview

The competition uses 5 datasets of increasing complexity. They unlock in two stages:

| Stage | Datasets | Token Budget | Unlock Condition                     |
| ----- | -------- | ------------ | ------------------------------------ |
| 1     | 1, 2, 3  | $40          | Available at start                   |
| 2     | 4, 5     | $120 more    | Submit eval solutions for all of 1-3 |

Every submission requires three elements: a Langfuse session ID, a UTF-8 output file, and (for evaluation datasets only) a source code zip.
Training submissions are unlimited and show a score each time. Evaluation submissions are one per dataset and cannot be re-submitted.

See 02_AI_Agents_Challenge/00_How_It_Works/README.md for the full rules, scoring breakdown, prizes, and submission format.
For rapid challenge-day operations, use 02_AI_Agents_Challenge/00_How_It_Works/submission_guide.md and
02_AI_Agents_Challenge/00_How_It_Works/challenge_day_checklist.md.

---

## Tech Stack

| Library          | Purpose                                        |
| ---------------- | ---------------------------------------------- |
| LangChain        | Agent framework and tool abstractions          |
| LangGraph        | ReAct agent execution engine                   |
| langchain-openai | OpenAI-compatible model connector              |
| OpenRouter       | Unified LLM API gateway                        |
| Langfuse         | Observability: token tracking, cost monitoring |
| ulid-py          | Unique session ID generation                   |
| python-dotenv    | .env file loading                              |

---

## Makefile targets

| Target                 | What it does                                                              |
| ---------------------- | ------------------------------------------------------------------------- |
| `make` or `make setup` | Creates root `.venv/`, installs all deps, registers Jupyter kernel        |
| `make check`           | Verifies all imports work and .env has all required credentials filled in |
| `make jupyter`         | Launches Jupyter Lab in the learning notebooks folder                     |
| `make clean`           | Removes the root `.venv/` (run `make` again to recreate)                  |

---

## Security

Never commit the .env file. It is excluded by .gitignore.
API keys and Langfuse credentials must be kept private at all times.
The .venv directories are also excluded from version control.

Challenge-day credential hygiene:

- Rotate any key that was exposed during testing or screenshots.
- Validate that source zips contain `.env.example` only (never real `.env`).
- Keep a fallback plan for rapid key rotation before final one-shot evaluation uploads.

To set up credentials: copy .env.example to .env in the repository root and fill in your values.
All scripts and notebooks use load_dotenv(find_dotenv()) to locate the root .env automatically from any subfolder.
