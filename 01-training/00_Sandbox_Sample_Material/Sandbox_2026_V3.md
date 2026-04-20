# Propedeutical Sandbox

## 12 thMarch 2026

Abstract
In the digital ecosystem of Reply Mirror, the public health infrastruc-
ture MirrorLife represents a benchmark for data-driven prevention and
sustainable well-being. Advanced cloud platforms integrate multimodal
information streams — including lifestyle indicators, environmental expo-
sures and longitudinal health signals — to support population-level health
optimization.
In this interconnected environment, prevention is no longer episodic
but continuous: health systems aim to anticipate deviations from optimal
trajectories and promote timely, personalized interventions.
As part of The Eye initiative, you are called to design an intelligent
system capable of monitoring dynamic well-being patterns, identifying
emerging vulnerabilities and supporting proactive preventive strategies.
The objective is not to react to disease, but to optimize well-being
trajectories across time through adaptive, evidence-based intelligence.


### REPLY MIRROR 1 PROBLEM STATEMENT

## 1 Problem Statement

It is 2087 and the digital metropolis of Reply Mirror operates under an inte-
grated prevention framework. Institutions such as MirrorLife coordinate large-
scale, longitudinal datasets to promote well-being optimization across the pop-
ulation.

The system continuously collects and harmonizes data related to daily routines,
physical activity, sleep patterns, stress proxies, environmental context and social
determinants. Within this complex and evolving landscape, subtle deviations
from optimal behavioural or physiological trajectories may signal the need for
preventive action.

You — The Eye — are tasked with developing an adaptive intelligence capable of
identifying these early signals. The goal is to determine when a citizen’s current
trajectory suggests that proactive preventive support could improve long-term
well-being outcomes.

Preventive risk patterns are not static. They evolve due to:

- changes in lifestyle behaviours and social habits;
- seasonal and environmental variations;
- demographic shifts and contextual factors;
- non-linear interactions between multiple low-intensity signals.

A key challenge lies in the temporal evolution of these patterns. Systems trained
on historical data must remain robust when confronted with distributional shifts,
behavioural drifts and structural changes in the environment.

Static predictive approaches are insufficient. The task requires adaptive, contin-
uously learning systems capable of generalizing across time while maintaining
calibrated and stable performance.

Challenge Flow The challenge unfolds in three levels. At each level, your
team receives a training dataset representing the general well-being status of the
citizens of ReplyMirror. In addition to this, you are provided with additional
datasets containing varying levels of information about citizens.

In addition to the training data, each level also includes a corresponding eval-
uation dataset of comparable difficulty. These evaluation datasets are the only
ones used to compute the official score for the level. For each evaluation dataset,
only the first submission will be accepted and will be considered final.

Throughout all levels, you are asked to design a system of cooperative intelli-
gent agents capable of classifying each individual-time instance according to the
following outcome:

- 0 = continuation of standard monitoring (well-being trajectory within
    acceptable optimization range);
- 1 = recommendation for activation of a personalized preventive support
    pathway.

At the end of all three levels, the overall leader-board will reflect the resilience
and adaptability of the system of each team.


### REPLY MIRROR 2 INPUT FORMAT

Challenge Goal You are asked to design an AI-based system capable of:

- identifying dynamic patterns associated with suboptimal well-being tra-
    jectories;
- anticipating emerging preventive needs through longitudinal pattern recog-
    nition;
- adapting to structural and temporal shifts without performance degrada-
    tion;
- maintaining a low rate of unnecessary preventive pathway activations.

Decision errors carry asymmetric implications:

- unnecessary activation of preventive support (false positive) may lead to
    inefficient allocation of resources and reduced system credibility;
- failure to activate support when appropriate (false negative) may result in
    missed opportunities for early optimization and long-term health benefits.

The final score combines predictive performance, temporal robustness and adapt-
ability to evolving well-being dynamics.

## 2 Input Format

The input consists of a series of data set with increasing complexity. You have
access to multiple types of data set, available for download in different formats.
In detail:

Status.csv with T records of well-being monitoring events, including:

- Event ID: a unique identifier of the monitoring event;
- Citizen ID: a unique identifier of the monitored individual;
- Event Type:
    - routine check-up
    - preventive screening
    - lifestyle coaching session
- Physical Activity Index (aggregated daily score);
- Sleep Quality Index (aggregated score);
- Environmental Exposure Level (area-based indicator);
- Timestamp of the recorded event;

Locations with geo-referenced data, whereby the citizen has been located
through GPS-based systems, irrespective of the type of activity performed, in-
cluding:

- BioTag: a unique identifier of the citizen;
- Datetime: date and time of geo-referentiation;
- Lat: latitude coordinate in degrees;
- Lng: longitude coordinate in degrees;

Users with a resume of citizen’s personal data.


### REPLY MIRROR 4 SCORING RULES

## 3 Output Format

The output must be an ASCII text file. Each line should be separated by newline
character and refers to Citizen ID with AI-recommended preventive support.

The format of each line is:
t

where:

- t represents the Citizen ID among the T record received in input.

## 4 Scoring Rules

The evaluation framework is built around a composite scoring model that mea-
sures performance across two key dimensions, with clinical impact serving as the
primary driver. Your goal is not to design a static, high-accuracy model, but to
develop an AI multi-agent system that is economically sustainable, operationally
efficient and capable of operating in a real-world production environment.

Accuracy This component of the score evaluates the system’s capability to
correctly detect potential suboptimal well-being trajectories. The goal is to
achieve a balanced trade-off between identifying emerging preventive needs and
avoiding unnecessary preventive pathway activations.

The metric used is the F 1 score:

### F 1 =

```
2 · P recision· Recall
P recision + Recall
```
with

```
P recision =
```
### T P

### T P + F P

```
, Recall =
```
### T P

### T P + F N

where:

- TP: number of correctly preventive pathway activated;
- FP: number of unnecessary preventive pathway activated;
- TN: number of correctly standard clinical monitoring mantained;
- FN: number of necessary preventive pathway not activated.

Additional Metrics Cost, speed and efficiency serve as complementary eval-
uation metrics. They reward the design of an optimized agent architecture ca-
pable of detecting clinical needs in real time while maintaining low operational
expense.

These dimensions emphasize the system’s ability to perform under real-world
constraints, balancing computational resources, processing latency and infras-
tructure usage. In essence, they measure how effectively your AI agents can
scale, respond and adapt without compromising economic sustainability or re-
sponsiveness.


### REPLY MIRROR 5 CONSTRAINTS

## 5 Constraints

### • T > 0

### • T ∈R


