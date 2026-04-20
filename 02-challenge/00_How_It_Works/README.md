# Reply AI Agent Challenge - Reference
# Source: https://challenges.reply.com/challenges/ai-agent/how-it-works/
# Date: April 16th | Duration: 6 hours | Theme: Monitor. Adapt. Defend.

---

## Competition Basics

- Open to professionals and students (16+) worldwide; Replyers on a separate leaderboard
- Team size: 2-4 members; solo registrants auto-paired at registration close
- Replyers and externals cannot form mixed teams
- Registration deadline: April 15th, 23:59 CEST - must create/join a team to compete
- Leave team: Challenge Page > "Leave team" (no notification sent). Edit profile: login > Profile > "Edit profile"
- Any language allowed; Python and JS/TS most common
- Problem statement questions during challenge: message AIvengers on Discord

---

## Timeline (April 16th)

| Time | Event |
|------|-------|
| 15:30 | Challenge starts; datasets 1-3 available |
| 21:00 | Leaderboard frozen |
| 21:30 | Leaderboard unfrozen; challenge ends |
| +10 workdays | Podium validation and results |

---

## Datasets and Token Budget

| Stage | Datasets | Budget | Unlock condition |
|-------|----------|--------|-----------------|
| 1 | 1, 2, 3 | $40 | Available at start |
| 2 | 4, 5 | $120 | Submit eval solutions for all of datasets 1-3 first |

Token budget is final - cannot be refilled.

---

## Resources Provided

Problem statement, LLM API key (via OpenRouter), training and eval datasets, token monitoring dashboard, Langfuse tracing dashboard.

- Full API and Langfuse integration guide: [api_guidelines.md](api_guidelines.md)
- All whitelisted models with OpenRouter IDs: [model_whitelist.md](model_whitelist.md)
- Challenge-day submission fast path and pitfalls: [submission_guide.md](submission_guide.md)
- 60-second final submission checks: [challenge_day_checklist.md](challenge_day_checklist.md)

---

## Submission Format

Every submission requires 3 elements:

| Element | Required for | Format |
|---------|-------------|--------|
| Langfuse Session ID | All submissions | Entered in upload modal |
| Output file | All submissions | .txt file with list of fraudulent transactions |
| Source code | Eval datasets only | .zip with complete agentic system |

Limits: training = unlimited submissions; evaluation = 1 submission only. Final score based solely on eval dataset.

Submission is invalid if any of the following are missing or broken: output file, source code (eval only), Langfuse session ID, working zip, complete system with deps/config/instructions.

Sandbox mode: practice simulation before challenge day. Shows per-submission scores; no global leaderboard in training area.

---

## Scoring

All metrics benchmarked against an optimal reference solution. Outperforming it earns bonus points. More complex datasets have higher maximum points.

Discord clarification (2026-04-14, organizers): scoring is weighted across multiple criteria and benchmarked against an optimal reference. Bonus credit may be awarded for outperforming benchmark targets.

| Category | Metric | Notes |
|----------|--------|-------|
| Detection quality | Count-based accuracy | Every transaction weighted equally |
| Detection quality | Economic accuracy | Recovering 50k EUR fraud > recovering 5 EUR fraud |
| System performance | Cost | LLM usage sustainability |
| System performance | Latency | Transaction processing speed |
| System performance | Agent architecture quality | Multi-agent system design |

Practical implication for implementation strategy:
- Do not optimize only for raw detection output quality.
- Actively trade off quality, LLM cost, latency, and architecture clarity.
- Prioritize strong performance on higher-difficulty datasets because they have higher maximum points.

---

## LLM Role Requirements

The LLM must be the core decision-making and orchestration layer.

Acceptable: LLM orchestrates the system, decides which tools to call, interprets results, adapts dynamically. Deterministic tools managed by the LLM.

Not acceptable: predominantly deterministic/rule-based system with superficial LLM usage; LLM used only for formatting or trivial tasks.

---

## Mandatory Technical Requirements

1. LLM via API - key provided; see model_whitelist.md
2. Langfuse - mandatory for tracking; session ID must be in every submission
3. Final eval submission - must include output file + source code zip with all deps, config, and run instructions

Platform does not execute code - score is from output files only. AIvengers must be able to run the submitted code for validation.

---

## Langfuse Integration (summary)

Required env vars: OPENROUTER_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST=https://challenges.reply.com/langfuse, TEAM_NAME

Session ID format: {TEAM_NAME}-{ULID} - groups all traces from one run

Key pattern: @observe() decorator + CallbackHandler() inside it + langfuse_client.flush() after calls

Score is from output files only - Langfuse session ID required for submission validation.

Full code, trace viewer helper, and best practices: [api_guidelines.md](api_guidelines.md)

---

## Prizes

| Leaderboard | 1st | 2nd | 3rd |
|-------------|-----|-----|-----|
| External | 2,500 EUR/member | 1,500 EUR/member | 1,000 EUR/member |
| Replyer | PlayStation 5 | AirPods 4 | Reply Shop voucher |
| University League | Reply football table or financial donation | - | - |
| High School League | 2,000 EUR donation + coding course | - | - |

Results announced within 10 working days. AIvengers decisions are final.

---

## University / High School League

- Score counts toward both main leaderboard and institution-specific leagues
- Sign up: enter university or school name when creating/joining a team
- Points added to each institution represented in a mixed-institution team
- Alumni eligible for University League; Replyers not eligible

---

## Fair Play

No solution sharing, no platform overloading, no malware, independent solutions required.
Violations result in instant disqualification. Report to: challenges@reply.com

---

## Key Links

| Resource | URL |
|----------|-----|
| Challenge home | https://challenges.reply.com/challenges/ai-agent/home |
| Learn and Train | https://challenges.reply.com/challenges/ai-agent/learn-train |
| API and Langfuse guide | api_guidelines.md |
| Whitelisted models | model_whitelist.md |
| Terms and Conditions | https://challenges.reply.com/content/a62ca513-fcfa-6d07-7764-9a233f195ec3/TCs_AI_AGENT_CHALLENGE_STD_ED_final.pdf |
| Support | challenges@reply.com / Discord (AIvengers) |
