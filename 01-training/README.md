# AI Agents Training - Sandbox Environment
# Reply Code Challenge 2026

This folder contains all materials for the sandbox training phase before the actual challenge on April 16th.
The sandbox lets you experience the exact challenge mechanics, submission flow, and scoring system against simplified datasets.

---

## What is the Sandbox

The sandbox is a pre-challenge practice environment hosted on the Reply Challenges platform.
It replicates the real challenge structure precisely: same file formats, same submission flow, same scoring criteria, same Langfuse tracking requirement.
The only differences are that no API credits are provided (you use free OpenRouter LLMs), and the datasets are smaller and simpler than the real challenge datasets.

The sandbox is already active and open. You have 6 days remaining from when these screenshots were taken.

---

## Problem Domain: Citizen Monitoring and Anomaly Detection

The challenge is not fraud detection in a financial sense. It is behavioral anomaly detection in a citizen health and welfare monitoring system.

The system tracks citizens over time through:
- GPS location data (where they go, how far, how often)
- Health status events (routine check-ups, screenings, specialist consultations, follow-up assessments, lifestyle coaching sessions)
- Three biometric indices recorded at each health event: PhysicalActivityIndex, SleepQualityIndex, EnvironmentalExposureLevel

The task is to identify which citizens are showing anomalous patterns that could indicate a welfare risk or deteriorating health trajectory. This is what "fraud detection" means in this context - detecting citizens who deviate from their expected behavioral baseline.

---

## Folder Structure

```
01_AI_Agents_Training/
  README.md                         - This file
  GUIDE.md                          - Step-by-step workflow for sandbox training
  00_Sandbox_Sample_Material/       - Official materials provided by the organizers
    Sandbox_2026_V3.pdf              - Full problem statement (read this first)
    Submission_Tracking.md           - Submission log and Langfuse reference
    Public_Levels/
      public_lev_1/                  - Dataset for level 1 (5 citizens, small scale)
        users.json
        locations.json
        status.csv
        personas.md
      public_lev_2/                  - Dataset for level 2 (larger scale)
      public_lev_3/                  - Dataset for level 3 (largest scale, most complex)
  01_Sandbox_Implementations/       - Your solution code goes here
  resources/
    Sandbox_Challenge_1.png          - Screenshot of the sandbox challenge page (overview)
    Sandbox_Challenge_2.png          - Screenshot of the submission interface
```

---

## Dataset Structure

Each level provides exactly 4 files:

**users.json** - Static citizen profiles
- user_id: 8-character alphanumeric identifier (e.g. IAFGUHCK)
- first_name, last_name, birth_year, job
- residence: city, lat, lng (home coordinates)

**locations.json** - Time-series GPS movement log
- user_id, timestamp, lat, lng, city
- One entry per recorded location ping
- Use this to analyze mobility patterns, travel frequency, geographic range

**status.csv** - Time-series health event log
- EventID, CitizenID, EventType, PhysicalActivityIndex, SleepQualityIndex, EnvironmentalExposureLevel, Timestamp
- EventType values: routine check-up, preventive screening, lifestyle coaching session, specialist consultation, follow-up assessment
- Index values are integers; higher or lower than baseline may be significant depending on the citizen
- Increasing EnvironmentalExposureLevel and declining PhysicalActivityIndex/SleepQualityIndex are warning signals

**personas.md** - Narrative behavioral profiles for each citizen
- Written descriptions of each person's expected lifestyle, mobility, health behavior, and social patterns
- Use these as the ground truth baseline for what is "normal" for that individual
- Deviations from the persona description are exactly what your agent should flag

---

## Dataset Size by Level

| Level | Citizens | Location entries | Status events |
|-------|----------|-----------------|---------------|
| 1 | 5 | ~variable | 50 |
| 2 | ~15 | larger | ~150 |
| 3 | ~30 | largest | ~640 |

Level complexity increases not just in size but also in the subtlety of anomalies and the number of citizens with overlapping or ambiguous signals.

---

## Submission Interface (from screenshots)

### Platform URL
The sandbox challenge page is at: challenges.reply.com (login required, navigate to Your Sandbox Challenge Page)

### Available on the page
- Download button for Sandbox_2026_V3.pdf (problem statement)
- Langfuse template button (provides the session tracking template)
- "View my Keys" button (your LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST, and OPENROUTER_API_KEY)
- "How to track your submission" link (explains the session ID requirement)
- Training dataset upload area: 3 slots (public_lev_1, public_lev_2, public_lev_3)
- Evaluation dataset upload area: 3 slots (same levels)

### Training dataset submissions
- Unlimited submissions allowed
- Upload a UTF-8 plain text output file for each level
- You see a score immediately after each submission
- No leaderboard visible to others in sandbox

### Evaluation dataset submissions
- One submission only per level - cannot be undone
- Upload output file first, then source code zip is requested
- Do not submit evaluation until you are confident in your solution

---

## Key Rules for the Sandbox

1. No API credits are provided in sandbox mode. Use free OpenRouter models for practice.
2. You must include a valid Langfuse session ID in every submission, even in sandbox.
3. Training submissions: unlimited, see score each time.
4. Evaluation submissions: exactly one per level, irreversible.
5. The output format must match the problem statement specification exactly (see Sandbox_2026_V3.pdf).
6. Source code zip (evaluation only) must contain everything needed to reproduce the output.

---

## Links

- Challenge platform: https://challenges.reply.com
- Sandbox page: https://challenges.reply.com/challenges/ai-agent/home (navigate to "Your Sandbox Challenge Page")
- Environment setup: run `make` from the repository root, then `cp .env.example .env` and fill in credentials from "View my Keys"
- How it works (rules, scoring): ../02_AI_Agents_Challenge/00_How_It_Works/README.md
- Langfuse and API guide: ../02_AI_Agents_Challenge/00_How_It_Works/api_guidelines.md
- Model whitelist: ../02_AI_Agents_Challenge/00_How_It_Works/model_whitelist.md
