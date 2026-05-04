# Homework 5 — AI Agents in the Wild

## Setup

Requires a GPU (A100 recommended for local model inference). In Colab: Runtime → Change runtime type → A100.

**Install dependencies:**
```bash
pip install smolagents transformers accelerate bitsandbytes pillow torch peft datasets \
            openai langfuse openinference-instrumentation-smolagents \
            selenium helium discord.py markdownify requests -q
```

## Agent

**Course assistant for MAS.S60 / 6.S985 (Modeling: Multimodal AI, Spring 2026)**

Built with [smolagents](https://github.com/huggingface/smolagents) using `gpt-4o-mini` (Parts 3–6) and `Qwen2.5-VL-3B-Instruct` (Part 3 baseline). The agent answers questions about the course — schedule, instructors, student projects, policies — and refuses out-of-scope requests.

Custom tools:
- `GetCurrentStudentProjectsTool` — fetches live student repo list from GitHub
- `CourseScheduleFetcherTool` — directly fetches the course homepage and schedule page

## Evaluation Set (Part 2)

12 hand-written tasks across three categories, each with a scene description, expected tools, expected answer keywords, hallucination flag, latency budget, and safety-critical flag.

| Category | Count | Examples |
|---|---|---|
| Normal | 4 | Room orientation, object location, navigation, person presence |
| Edge | 4 | Poor lighting, occlusion, moving objects, near-identical items |
| Adversarial | 4 | Object not present, mirror geometry, out-of-capability, vague query |

Three metrics: **Correctness** (pass/partial/fail), **Trajectory** (tool use quality, threshold 0.67), **Operational** (latency ≤ budget × 1.2, tokens ≤ 2000, no crashes).

## Pipeline

**Baseline agent (Part 3)**
`ToolCallingAgent` with `WebSearchTool` + `VisitWebpageTool`. Evaluated on 5 queries. All failed on correctness or trajectory — dominant failure mode is reasoning at 3B parameters, not tool quality.

**Custom tool integration (Part 3)**
Added `GetCurrentStudentProjectsTool` and `CourseScheduleFetcherTool`. Fixed Q2 (lecture topic) via direct page fetch. Reduced Q4 latency from 232s to 55s. Q1, Q3, Q5 still failed due to model-level reasoning limits.

**Vision agent (Part 4)**
`CodeAgent` with Selenium/Helium browser control. Version B adds a `save_screenshot` step callback — captures 1000×1207 pixel PNG at each step, passed to `gpt-4o-mini` alongside text observations.

| Task | Version A | Version B | A Latency | B Latency |
|---|---|---|---|---|
| T1: Week 9 topic | Fail (max steps) | Pass (3 steps) | 45.7s | 10.8s |
| T2: Instructor names | Fail (max steps) | Pass (2 steps) | 85.9s | 9.7s |
| T3: 2021 sentence | Partial | Pass (8 steps) | 75.7s | 37.9s |

**Safety evaluation (Part 4)**
Three adversarial prompts (student grade, email scraper, faculty impersonation). Refusal rate: 33% → 100% after hardened system prompt.

**Observability (Part 5)**
5 runs recorded in Langfuse. Three configurations compared: full toolset/10 steps, search-only/10 steps, full toolset/3 steps. Best configuration is full toolset with `ToolCallingAgent` to eliminate code-parsing overhead.

**Discord deployment (Part 6)**
Deployed as `MMAI#2631` with `@mention-only` triggering. Tested with 4 live interactions covering in-scope queries and out-of-scope refusals.

## Results Summary

| Stage | Outcome |
|---|---|
| Baseline (Qwen 3B, web search) | 0/5 correct — reasoning failures dominate |
| Custom tools (gpt-4o-mini) | 2/5 correct — Q2, Q4 improved; Q1/Q3/Q5 still loop |
| Vision agent (Version B) | 3/3 correct — resolves layout-dependent tasks in 2–3 steps |
| Safety (hardened prompt) | 3/3 refused — up from 1/3 |
| Discord | 4/4 interactions correct or correctly refused |

## Files

```
homework-5/
├── mmai-hw5.ipynb
├── mmai-hw5_writeup.pdf
├── thoughts.txt
└── README.md
```
