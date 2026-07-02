---
name: survey-report
description: Use this skill when writing the survey synthesis report (report.md). Provides the methodology for reading existing workspace data (research plan, collected papers, per-paper analyses) and producing a well-structured Survey/report.md. This skill only writes report.md — it does NOT perform web search or modify other files.
metadata:
  author: DeepResearch
  version: "1.0"
---

# survey-report

## Overview

This skill guides the process of **writing `Survey/report.md`** — the survey synthesis report. It reads existing workspace data that has already been collected and analyzed, then produces a comprehensive, well-structured report.

This skill **only writes `Survey/report.md`**. It does not:
- Perform web searches (see `deep-search` skill)
- Modify `Inputs/raw/collected_papers.md`
- Modify `Survey/surveyed.md`

## Input Data Sources

Read the following workspace files as input for writing the report:

| File | Content | How to Use |
|------|---------|------------|
| `Planning/task_breakdown.json` | Research plan, domain, keywords | Frame the report scope and structure |
| `Planning/global_context.json` | Global context, primary domain | Set domain-specific tone and terminology |
| `Inputs/raw/collected_papers.md` | Structured literature data | Extract paper metadata, statistics, and coverage info |
| `Survey/surveyed.md` | Per-paper deep analysis | Primary content source — synthesize these analyses |
| `Inputs/uploads/*` | User-uploaded reference files | Incorporate if relevant |
| `Survey/human_feedback_survey.md` | User feedback on previous survey | Address feedback in revised report (iteration mode) |

## Workflow

1. **Read All Input Data**
   - Read `Planning/task_breakdown.json` for research scope and keywords
   - Read `Survey/surveyed.md` for per-paper analysis results
   - Read `Inputs/raw/collected_papers.md` for paper metadata and statistics
   - If iterating, read previous `Survey/report.md` and `Survey/human_feedback_survey.md`

2. **Analyze and Organize**
   - Group the per-paper analyses by research direction/theme
   - Identify cross-cutting patterns, connections, and contradictions
   - Map the technology development timeline
   - Identify research gaps and unexplored opportunities

3. **Write report.md**
   - Follow the report structure defined below
   - Ensure every claim is traceable to a specific paper in `surveyed.md`
   - Use quantitative data where available

4. **Save Output**
   - Save the report to `Survey/report.md`

## Report Structure (report.md)

The report must follow this structure:

### 1. Survey Overview
- Research topic and scope
- Data sources summary (total papers, quality distribution, domains covered)
- Search queries used (from `collected_papers.md` search summary section)
- Methodology brief

### 2. Results by Research Direction
For each identified research direction:
- **Direction title and description**
- **Key papers**: List the most relevant papers with their core findings
- **Technical approaches**: Summarize the methods used in this direction
- **Current state of the art**: What is the best-performing approach and why
- **Direction summary**: Key takeaways for this direction

### 3. Core Concepts & Methods Summary
- Cross-cutting themes that appear across multiple directions
- Key techniques and their relationships (e.g., which methods build on which)
- Common assumptions and their validity

### 4. Technical Gaps & Research Opportunities
- Areas with insufficient coverage in the literature
- Contradictions or unresolved debates between papers
- Promising but unexplored combinations of existing methods
- Limitations shared by most current approaches

### 5. Conclusions & Next Steps
- Key takeaways from the survey
- Recommended focus areas for the innovation phase
- Suggested priorities based on gap analysis

## Quality Guidelines

- Every statement must be supported by evidence from the surveyed papers
- Use specific numbers and metrics when available (e.g., "achieved 95.3% accuracy" not "achieved high accuracy")
- Clearly distinguish between established facts and speculative observations
- Maintain a neutral, analytical tone throughout

## Iteration Protocol

When revising an existing report (not first-time):

1. Read the previous `Survey/report.md`
2. Read `Survey/human_feedback_survey.md` for user's specific feedback
3. Identify which sections need revision based on the feedback
4. Revise the report — update affected sections while preserving unaffected content
5. Add a "Revision Notes" section at the top noting what changed and why
