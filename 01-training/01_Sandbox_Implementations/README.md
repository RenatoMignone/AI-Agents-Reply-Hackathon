# Sandbox Implementations

This folder contains a practical starter implementation for the Reply AI Agent sandbox.

## What is included

- `sandbox_agent.py`: CLI runner that
  - loads one public level dataset,
  - builds per-citizen snapshots from `users.json`, `status.csv`, `locations.json`, and `personas.md`,
  - asks an LLM to decide `ANOMALOUS` vs `NORMAL` for each citizen,
  - writes a submission-ready ASCII text file containing one flagged Citizen ID per line,
  - prints a submission session ID in `{TEAM_NAME}-{ULID}` format.

## Prerequisites

Use the repository root environment and `.env` values.

Required environment variables:

- `OPENROUTER_API_KEY`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST`
- `TEAM_NAME`

Notes:

- Keep `LANGFUSE_HOST=https://challenges.reply.com/langfuse`
- For sandbox training, prefer free OpenRouter models.

## Run examples

From this folder:

```bash
/home/ren/PERSONAL_DIRECTORY/CyberSecurity/Others/AI_Agents_Reply_Challenge/.venv/bin/python sandbox_agent.py --level 1 --model meta-llama/llama-3.1-8b-instruct
```

Dry run without LLM/API keys (debug parser and output only):

```bash
/home/ren/PERSONAL_DIRECTORY/CyberSecurity/Others/AI_Agents_Reply_Challenge/.venv/bin/python sandbox_agent.py --level 1 --dry-run
```

Fast sample run on first N citizens:

```bash
/home/ren/PERSONAL_DIRECTORY/CyberSecurity/Others/AI_Agents_Reply_Challenge/.venv/bin/python sandbox_agent.py --level 2 --max-citizens 5
```

Custom output path:

```bash
/home/ren/PERSONAL_DIRECTORY/CyberSecurity/Others/AI_Agents_Reply_Challenge/.venv/bin/python sandbox_agent.py --level 3 --output outputs/my_level3.txt
```

Quality-focused run (adds second-pass review for uncertain/disputed cases):

```bash
/home/ren/PERSONAL_DIRECTORY/CyberSecurity/Others/AI_Agents_Reply_Challenge/.venv/bin/python sandbox_agent.py --level 2 --model meta-llama/llama-3.1-8b-instruct --enable-review --review-on-disagreement
```

Cost/latency-focused run (default compact persona/context):

```bash
/home/ren/PERSONAL_DIRECTORY/CyberSecurity/Others/AI_Agents_Reply_Challenge/.venv/bin/python sandbox_agent.py --level 2 --model meta-llama/llama-3.1-8b-instruct
```

If you want maximum context (more tokens, slower), add:

```bash
--full-persona --verbose-context
```

## Submission checklist

1. Copy the printed session ID.
2. Upload the generated `.txt` output file to the correct level slot.
3. Paste the same session ID in the submission modal.
4. For evaluation submissions only: upload a source code zip after output upload.

## Next improvements

- Add an explicit multi-agent orchestration layer (analyst + risk judge + final arbiter).
- Add result caching to avoid re-querying unchanged citizen snapshots.
- Add prompt variants and majority voting for borderline cases.