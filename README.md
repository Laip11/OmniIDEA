# OmniIDEA

**OmniIDEA** is a multi-agent research workflow for turning open-ended research questions into structured literature surveys, research ideas, critical assessments, and final reports.

The pipeline is designed for **human-in-the-loop** research: you can steer each phase, request re-runs, upload references, or switch to fully automatic mode when you prefer minimal interaction.

---

## Features

- **Multi-phase research pipeline**: Survey → Idea → Assessment → Report, orchestrated by a Planning Agent
- **Human-in-the-loop (HITL)**: pause for feedback after key phases; resume from checkpoint
- **Iterative refinement**: Assessment feedback can loop back to Idea generation
- **Skill system**: modular `Skills/*/SKILL.md` files (deep research, structured search, report formatting, PDF tools, etc.)
- **Workspace file system**: all artifacts saved under `workspace/` with a consistent layout
- **Web & document tools**: Tavily search, web fetch, PDF text extraction, file read/write
- **Checkpointing**: interrupted runs can be resumed via LangGraph memory checkpointer

---

## Architecture

```
                    ┌─────────────┐
                    │   Planner   │◄────────────────────────────┐
                    │  (router)   │                             │
                    └──────┬──────┘                             │
                           │                                     │
         ┌─────────────────┼─────────────────┐                   │
         ▼                 ▼                 ▼                   │
    ┌─────────┐      ┌──────────┐     ┌────────────┐             │
    │ Survey  │      │   Idea   │────►│ Assessment │             │
    └────┬────┘      └────┬─────┘     └─────┬──────┘             │
         │                │                  │ (feedback loop)   │
         └────────────────┴──────────────────┘                   │
                           │                                     │
                           ▼                                     │
                    ┌─────────────┐                              │
                    │   Report    │                              │
                    └──────┬──────┘                              │
                           │                                     │
                           ▼                                     │
                    ┌─────────────┐                              │
                    │ HumanReview │──────────────────────────────┘
                    └─────────────┘
```

| Agent | Role | Typical temperature |
|-------|------|-------------------|
| **Planning Agent** | Routes workflow, interprets user intent, initializes plan | 0.7 |
| **Survey Agent** | Literature & web research, writes `Survey/report.md` | 0.3 |
| **Idea Agent** | Generates research ideas, writes `Innovation/ideas.md` | 0.95 |
| **Assessment Agent** | Generates domain-adaptive rubric, evaluates ideas, structured feedback | 0.7 |
| **Report Agent** | Synthesizes final report → `Outputs/Final_Report.md` | 0.7 |

---

## Requirements

- **Python** 3.10+
- **OpenAI-compatible API** (OpenAI, Azure OpenAI, or compatible gateway)
- **Tavily API key** for web search ([tavily.com](https://tavily.com))
- Optional: **Jina API key** for enhanced web page extraction

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Laip11/OmniIDEA.git
cd OmniIDEA
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | API key for the LLM provider |
| `OPENAI_MODEL` | Yes | Model name (e.g. `gpt-4o-mini`, `gpt-4o`) |
| `TAVILY_API_KEY` | Yes | Tavily search API key |
| `OPENAI_BASE_URL` | No | Custom OpenAI-compatible endpoint |
| `MAX_NEW_TOKENS` | No | Max output tokens (default: `16384`) |
| `JINA_API_KEY` | No | Jina Reader for web fetch |

---

## Quick Start

### Run with a research question (CLI argument)

```bash
python main.py "How can coding agents combine meta-evolution with reinforcement learning?"
```

### Run interactively (prompt for topic)

```bash
python main.py
```

### Use a custom checkpoint thread (separate sessions)

```bash
python main.py "Your research topic" --thread my_session_01
```

### Exit during human review

Type `quit`, `exit`, or `q` at the human review prompt. Progress is saved and can be resumed on the next run with the same `--thread`.

---

## Standard Usage Workflow

### Step 1 — Start a session

Launch `main.py` with your research question. The **Planning Agent** initializes the workspace and may ask clarifying questions or confirm the research plan.

### Step 2 — Survey phase

The **Survey Agent** searches the web (Tavily), fetches relevant pages, and writes a literature-style report to:

```
workspace/Survey/report.md
```

You can say things like *"redo the survey"* or *"focus more on RL methods"* during human review.

### Step 3 — Idea phase

The **Idea Agent** proposes research directions and hypotheses:

```
workspace/Innovation/ideas.md
```

Provide feedback in natural language (e.g. *"emphasize multi-objective rewards"*).

### Step 4 — Assessment phase

The **Assessment Agent** first generates a domain-adaptive evaluation rubric, then evaluates ideas against it and writes structured improvement feedback:

```
workspace/Assessment/rubric.md
workspace/Assessment/rubric_dimensions.json
workspace/Assessment/assessment_feedback.md
```

If ideas need improvement, the workflow can loop **Assessment → Idea** automatically. The loop stops early when the Assessment Agent's Iteration Verdict is `CONVERGED`, or after reaching the configured max iterations.

### Step 5 — Report phase

The **Report Agent** produces the final deliverable:

```
workspace/Outputs/Final_Report.md
```

### Step 6 — Human review & iteration

At any checkpoint, you can:

| Intent | Example phrases |
|--------|-----------------|
| Continue | *"looks good, continue"* |
| Re-run a phase | *"redo survey"*, *"regenerate ideas"* |
| Upload files | Place PDFs in `workspace/Inputs/uploads/` and mention them |
| Auto mode | *"run automatically / no more questions"* |
| Finish | *"finish"* or `quit` |

### Step 7 — Retrieve outputs

After completion, check:

```bash
ls workspace/Outputs/
cat workspace/Outputs/Final_Report.md
```

Version history snapshots are stored under `workspace/History/version_*.json`.

---

## Workspace Layout

```
workspace/
├── Inputs/
│   ├── uploads/          # User-uploaded PDFs and references
│   └── raw/              # Cached web/PDF extractions
├── Planning/
│   ├── search_plan.md
│   └── global_context.json
├── Survey/
│   └── report.md
├── Innovation/
│   └── ideas.md
├── Assessment/
│   ├── rubric.md
│   ├── rubric_dimensions.json   # Rubric source data (adaptive dimensions)
│   ├── assessment_feedback.md
│   └── iteration_history.md     # Idea↔Assessment iteration archive
├── Outputs/
│   └── Final_Report.md
└── History/
    └── version_*.json    # Iteration snapshots
```

Runtime artifacts under `workspace/` are git-ignored by default (except `.gitkeep` placeholders). Your local runs will not pollute the repository.

---

## Skills

Skills live in `Skills/<name>/SKILL.md` and follow the [Agent Skills](https://docs.langchain.com/oss/python/deepagents/skills) convention (YAML frontmatter + markdown instructions).

| Skill | Purpose |
|-------|---------|
| `deep-research` | Multi-angle web research methodology |
| `deep-search` | Structured search strategies |
| `report-formatting` | Final report structure |
| `google-scholar-search` | Scholar-oriented search guidance |
| `pdf` | PDF handling instructions |

Agents load skills automatically via Deep Agents progressive disclosure.

---

## Project Structure

```
OmniIDEA/
├── main.py           # CLI entry point & human-in-the-loop loop
├── graph.py          # LangGraph workflow definition
├── nodes.py          # Agent node implementations
├── agents.py         # Deep Agent builders & LLM config
├── state.py          # Typed workflow state
├── prompts.py        # System prompts per agent role
├── tools.py          # Tavily, web fetch, PDF, file tools
├── storage.py        # Workspace store & path constants
├── cli.py            # Terminal output formatting
├── Skills/           # Agent skill definitions
├── workspace/        # Runtime artifacts (git-ignored)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | LLM API key (required) |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model identifier |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API base URL |
| `TAVILY_API_KEY` | — | Web search (required) |
| `MAX_NEW_TOKENS` | `16384` | Max generation tokens |
| `AGENT_INVOKE_MAX_RETRIES` | `3` | Retries on transient API errors |
| `AGENT_RETRY_BASE_DELAY` | `2` | Exponential backoff base (seconds) |
| `AGENT_RETRY_MAX_DELAY` | `8` | Exponential backoff cap (seconds) |
| `IDEA_ASSESSMENT_MAX_ITERS` | `2` | Max automatic Idea↔Assessment improvement rounds; the loop also stops early when the Assessment Agent's Iteration Verdict is `CONVERGED` |

---

## Troubleshooting

**`OPENAI_API_KEY is required`**

Copy `.env.example` to `.env` and set your API key.

**Tavily search warnings**

Set `TAVILY_API_KEY` in `.env`. Without it, web search tools will fail.

**PDF extraction errors**

Install optional PDF backends (included in `requirements.txt`):

```bash
pip install pdfplumber pypdf
```

**Resume after interrupt**

Re-run the same command with the same `--thread` id. LangGraph restores checkpoint state automatically.

**Context / token limits**

Large PDFs are truncated in tool returns; full text is cached under `workspace/Inputs/raw/pdf_cache/`. Use page-range extraction for long documents.

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgments

Built with [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain), and [Deep Agents](https://github.com/langchain-ai/deepagents).
