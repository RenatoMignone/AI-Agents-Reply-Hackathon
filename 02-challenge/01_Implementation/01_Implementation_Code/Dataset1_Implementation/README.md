# Implementation - Reply Code Challenge 2026

This folder now contains the current challenge-day implementation for the Reply Mirror fraud task.

## Current structure

```text
01_Implementation/
  README.md
  main.py
  requirements.txt
  outputs/
```

## What the current code does

- Loads one dataset folder containing:
  - `transactions.csv`
  - `users.json`
  - `locations.json`
  - `sms.json`
  - `mails.json`
- Builds deterministic fraud-risk features per transaction.
- Selects high-risk candidates grouped by sender.
- Uses an LLM as final reviewer for candidates (unless `--dry-run` is used).
- Writes ASCII output with one `transaction_id` per line.
- Prints a submission session ID in `{TEAM_NAME}-{ULID}` format.

## Environment

- Use the repository root `.env`.
- Do not create a local `.env` in this folder.
- Use the root virtual environment (`.venv`).

## Run commands

From repository root:

```bash
source .venv/bin/activate

python 02_AI_Agents_Challenge/01_Implementation/main.py \
  --dataset "02_AI_Agents_Challenge/The+Truman+Show+-+train/The Truman Show - train" \
  --output "02_AI_Agents_Challenge/01_Implementation/outputs/truman_predictions.txt" \
  --dry-run
```

Live LLM run:

```bash
source .venv/bin/activate

python 02_AI_Agents_Challenge/01_Implementation/main.py \
  --dataset "02_AI_Agents_Challenge/Brave+New+World+-+train/Brave New World - train" \
  --output "02_AI_Agents_Challenge/01_Implementation/outputs/brave_predictions.txt" \
  --model "meta-llama/llama-3.1-8b-instruct"
```

## Truman First Challenge (current tuned baseline)

Use this command for the current best dry-run baseline on The Truman Show:

```bash
source .venv/bin/activate

python 02_AI_Agents_Challenge/01_Implementation/main.py \
  --dataset "02_AI_Agents_Challenge/The+Truman+Show+-+train/The Truman Show - train" \
  --output "02_AI_Agents_Challenge/01_Implementation/outputs/truman_submission_push_best.txt" \
  --dry-run \
  --risk-threshold 2.3
```

Current candidate output file:
- `02_AI_Agents_Challenge/01_Implementation/outputs/truman_submission_push_best.txt`

## Notes before submissions

- Output must be plain text with one transaction ID per line.
- Keep and reuse the exact session ID printed by the run when uploading.
- For one-shot evaluation submissions, prepare a reproducible source zip (`code + requirements + .env.example`, never real secrets).
- Langfuse export currently depends on valid challenge credentials/host configuration.

## Troubleshooting: "No traces found for the given session id"

- Do not use a session ID produced by `--dry-run` for submission.
- Use only a session ID from a successful live run (without `--dry-run`).
- The script now fails fast if Langfuse authentication fails, so invalid session IDs are not silently produced.
- If you get `Langfuse authentication failed` with HTTP 404/401, verify `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY` in root `.env` against the values shown in the challenge platform.
