# Implementation - Reply Code Challenge 2026

This folder is the workspace for the actual competition solution written on challenge day (April 16th, 2026).

It is currently empty. Create your solution files here during the challenge.

---

## Suggested structure

```
01_Implementation/
  README.md           - This file (update with your architecture and run instructions)
  main.py             - Entry point: loads data, runs agents, writes output file
  agent.py            - Agent and model initialization
  tools.py            - Tool functions used by the agents
  utils.py            - Data loading, parsing, and output formatting helpers
  .env                - Your credentials (do not commit)
  requirements.txt    - Or use a pip freeze output
```

---

## Before coding

Read the following files before writing any code:

1. ../00_How_It_Works/README.md - Rules, scoring, submission format, dataset unlock conditions
2. ../00_How_It_Works/api_guidelines.md - Langfuse integration, session ID generation, best practices
3. ../00_How_It_Works/model_whitelist.md - Choose your model by OpenRouter ID

The problem statement PDF will be available on the challenge platform on April 16th.

---

## Submission checklist

Training submissions (unlimited):
- Output file: UTF-8 plain text, format as specified in the problem statement
- Langfuse session ID: enter in the upload modal for every submission

Evaluation submissions (one per dataset, irreversible):
- Output file: same as training
- Langfuse session ID: required
- Source code zip: must contain all code, a requirements list, .env.example (not real .env), and instructions to reproduce the output

---

## Run instructions

Update this section when you have a working solution:

```bash
# Example (fill in when built)
source .venv/bin/activate
python main.py --level 1 --output output_lev1.txt
```
