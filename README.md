# AI Agents - Reply Code Challenge 2026

This repository contains the full learning path, sandbox training materials, and challenge solution workspace
for the Reply Code Challenge 2026 - AI Agents track.

> **Documentation Design Notice**
> All documentation in this repository is written and structured for AI agent readability and token efficiency.
> Every file follows these rules: no emojis, no decorative padding, no redundant prose, no information duplicated across files.
> Each README is scoped strictly to its own folder. Cross-folder navigation is handled through explicit pointer lines only.
> The primary entry point for any AI agent working in this repository is `AI_Agent.md` in this root directory.

---

## What this repository is for

This is a competition workspace for the Reply Code Challenge 2026, a timed AI agent engineering event.

**Challenge day:** April 16th, 2026 - 6 hours (15:30 to 21:30 CEST)
**Theme:** Monitor. Adapt. Defend.
**Format:** Build an AI agent system that analyzes citizen behavioral data
and identifies anomalous patterns indicating welfare risk.

The repository is organized into three phases that follow the natural progression from learning to competing:

1. **Learning** - Understand the technology stack through four progressive tutorials
2. **Training** - Practice against sandbox datasets that replicate the real competition mechanics
3. **Challenge** - Build and submit the actual competition solution on April 16th

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

  .setup/                        - Environment setup scripts and dependency manifest
    requirements.txt             - All Python dependencies for the entire project
    check_setup.py               - Verifies imports and .env credentials (run via 'make check')

  00_AI_Agents_Learning/         - Tutorial notebooks (start here if new to the stack)
    README.md                    - Setup, credential config, notebook order
    Notebooks/                   - Four Jupyter notebooks to run in sequence
    TXT/                         - Source instructions used to build the notebooks

  01_AI_Agents_Training/         - Sandbox training environment and practice datasets
    README.md                    - Problem domain, file schemas, submission interface
    GUIDE.md                     - Step-by-step sandbox workflow (8 steps)
    00_Sandbox_Sample_Material/  - Official organizer-provided materials
      Sandbox_2026_V3.pdf        - Full problem statement (read before coding)
      Submission_Tracking.md     - Submission log, results, and Langfuse code reference
      Public_Levels/             - Training datasets for levels 1, 2, and 3
    01_Sandbox_Implementations/  - Write your sandbox solution code here
    resources/                   - Screenshots of the web submission interface

  02_AI_Agents_Challenge/        - The actual competition solution workspace
    README.md                    - Contents overview for this folder
    00_How_It_Works/             - Official rules, API docs, and model reference
      README.md                  - Competition rules, timeline, scoring, prizes
      api_guidelines.md          - Langfuse integration code and best practices
      model_whitelist.md         - All whitelisted OpenRouter model IDs
    01_Implementation/           - Write your challenge day solution here
      README.md                  - Architecture notes and run instructions (fill as you build)
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

This creates the root `.venv/`, installs all dependencies from `.setup/requirements.txt`, and registers the Jupyter kernel.

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

To set up credentials: copy .env.example to .env in the repository root and fill in your values.
All scripts and notebooks use load_dotenv(find_dotenv()) to locate the root .env automatically from any subfolder.
