# Elia_AgentSpace - MirrorPay Fraud Agent

Agentic fraud detection implementation for the Reply Code Challenge 2026.

## Architecture

- LangGraph ReAct agent (`agent.py`)
- Tool layer for dataset ingestion and summarization (`tools.py`)
- CLI entrypoint and output generation (`main.py`)

The LLM is the decision-maker. Tools only retrieve and summarize data.

## Requirements

- Python 3.10-3.13
- Root `.env` configured with challenge credentials
- Dependencies in this folder's `requirements.txt` (or root `.venv`)

## Run

From repository root:

```bash
source .venv/bin/activate
cd 02_AI_Agents_Challenge/01_Implementation/Elia_AgentSpace
python main.py --dataset "../../datasets/Brave New World - train" --model cheap
```

Validation run example:

```bash
python main.py --dataset "../../datasets/Brave New World - validation" --model heavy --output output_brave_validation.txt --quiet
```

## Model Presets

- `cheap`: `meta-llama/llama-3.1-8b-instruct`
- `mid`: `google/gemini-2.0-flash-001`
- `heavy`: `anthropic/claude-haiku-4.5`
- `gemini-pro`: `google/gemini-2.5-flash`

## Output Format

UTF-8 plain text file with one suspected fraudulent transaction ID per line.

## Submission Notes

- Use the printed session ID for upload (`{TEAM_NAME}-{ULID}`)
- For evaluation uploads include: `agent.py`, `tools.py`, `main.py`, `requirements.txt`, `.env.example`, this `README.md`
- Never include a real `.env` in the source zip
