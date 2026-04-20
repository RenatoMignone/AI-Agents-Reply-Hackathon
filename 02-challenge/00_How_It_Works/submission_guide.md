# Submission Guide - Challenge Day Fast Path

Purpose: minimal, high-signal checklist for generating and submitting valid files under time pressure.

Use this guide for right-column evaluation submissions.

---

## 1. What to read first (token-efficient)

If you are an AI agent, load files in this order:

1. `AI_Agent.md`
2. `02_AI_Agents_Challenge/00_How_It_Works/README.md`
3. `02_AI_Agents_Challenge/00_How_It_Works/submission_guide.md` (this file)

Only load additional files if blocked.

---

## 2. Hard constraints (do not violate)

- Evaluation (right column) is one-shot per level.
- Training (left column) is unlimited.
- Every submission needs a Langfuse Session ID.
- Session ID format: `{TEAM_NAME}-{ULID}`.
- Output file must be plain text with one ID per line.
- Evaluation requires source code zip after output upload.
- Never include real `.env` in zip. Include `.env.example` only.

---

## 3. Critical dataset rule

For final/evaluation outputs, run on evaluation/submission datasets, not on public training datasets.

- Correct source: `01_AI_Agents_Training/00_Sandbox_Sample_Material/Submission_Levels/`
- Wrong source for final upload: `01_AI_Agents_Training/00_Sandbox_Sample_Material/Public_Levels/`

If output IDs come from public levels, platform can reject output as invalid.

Ignore `__MACOSX/` folders in extracted zips.

---

## 4. Generation commands

Run from `01_AI_Agents_Training/01_Sandbox_Implementations/`.

Level 1:

```bash
/home/ren/PERSONAL_DIRECTORY/CyberSecurity/Others/AI_Agents_Reply_Challenge/.venv/bin/python sandbox_agent.py --level 1 --model meta-llama/llama-3.1-8b-instruct --output ../Submissions/Lev_1/public_lev_1_predictions.txt
```

Level 2:

```bash
/home/ren/PERSONAL_DIRECTORY/CyberSecurity/Others/AI_Agents_Reply_Challenge/.venv/bin/python sandbox_agent.py --level 2 --model meta-llama/llama-3.1-8b-instruct --output ../Submissions/Lev_2/public_lev_2_predictions.txt
```

Level 3:

```bash
/home/ren/PERSONAL_DIRECTORY/CyberSecurity/Others/AI_Agents_Reply_Challenge/.venv/bin/python sandbox_agent.py --level 3 --model meta-llama/llama-3.1-8b-instruct --output ../Submissions/Lev_3/public_lev_3_predictions.txt
```

Copy the printed Session ID immediately after each run.

---

## 5. Pre-upload validation (30 seconds)

For each level file:

```bash
file -bi <output_file>
wc -l <output_file>
cat <output_file>
```

Expected:

- text/plain
- non-zero line count
- each line is one citizen/transaction ID

Optional strict check:

- verify every predicted ID exists in that level's `users.json` from `Submission_Levels`.

---

## 6. Source zip requirements (evaluation)

Zip must include:

- `.py` files needed to reproduce output
- `requirements.txt`
- `.env.example`
- `README` with run instructions

Do not include:

- `.env`
- API keys/secrets
- large irrelevant folders (`.venv`, caches, notebooks unless required)

---

## 7. Upload sequence (right column)

Per level:

1. Upload output txt.
2. Paste Session ID from the exact run that generated that txt.
3. Upload source code zip when prompted.
4. Re-check level slot before final confirm.

---

## 8. Fast troubleshooting

Issue: "Output is not valid"

Check in order:

1. Wrong dataset source used (Public_Levels instead of Submission_Levels).
2. Output uploaded to wrong level slot.
3. Wrong Session ID pasted (from another run/level).
4. Output contains IDs not present in that level dataset.
5. File is empty or malformed.

Issue: run crashes on malformed LLM output

- Use current `sandbox_agent.py` with retry + fallback handling.

### Failure playbook (symptom -> cause -> fix)

| Symptom | Most likely cause | Immediate fix |
|---|---|---|
| Output is not valid | Generated from `Public_Levels` instead of `Submission_Levels` | Re-run on `Submission_Levels`, regenerate txt, upload again |
| Output is not valid | Wrong level slot | Upload txt to matching level slot only |
| Output is not valid | Session ID from another run | Paste Session ID from the exact run that produced uploaded txt |
| Output is not valid | IDs not present in target level dataset | Validate IDs against target `users.json` before upload |
| Evaluation upload rejected | Source zip missing required files | Rebuild zip with `.py`, `requirements`, `.env.example`, `README` |
| Runtime stops mid-run | Malformed/empty model response | Keep retry+fallback enabled; rerun with same config |

---

## 9. Quality vs cost strategy

Given weighted scoring:

- optimize both detection quality and efficiency,
- use compact prompts by default,
- use second-pass review only for uncertain/disputed cases,
- keep latency low and architecture clear.

If uncertain on final one-shot submission, test on training first.

---

## 10. Optimization directive (Discord scoring alignment)

On challenge day, treat optimization as multi-objective and weighted:

- Detection quality:
	- count-based accuracy,
	- economic accuracy (high-value fraud recovery matters more).
- System performance:
	- cost sustainability,
	- latency,
	- architecture quality.
- Benchmark and bonus:
	- outperforming benchmark can add credit.
- Dataset difficulty:
	- higher-complexity datasets can carry higher point ceilings.

Decision policy for future agents:

- Prefer the configuration with better weighted quality-per-cost-per-latency, not just best raw hit count.
- Escalate expensive reasoning only for uncertain cases.
- Keep logs compact but complete enough to reproduce and explain architecture decisions.