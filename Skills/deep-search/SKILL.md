---
name: deep-search
description: Use this skill when the research topic requires multi-round deep web search with iterative query refinement. Provides a structured methodology for exhaustive literature discovery, including snowball search, citation tracking, and gap analysis.
metadata:
  author: DeepResearch
  version: "1.0"
allowed-tools: tavily_search
---

# deep-search

## Overview

This skill provides an advanced multi-round search methodology for comprehensive literature discovery. Use it when a standard single-round search is insufficient to cover the research topic.

## Instructions

### 1. Initial Broad Search

Execute 2-3 broad queries using the research topic's core keywords:
- Use domain-specific terminology from `Planning/task_breakdown.json`
- Include both English and Chinese keywords for bilingual coverage
- Target recent publications by appending year constraints (e.g., "2024", "2025")

### 2. Snowball Search (Citation Tracking)

From the initial results, identify:
- **Key authors**: Search for their other recent publications
- **Key terms**: Extract novel terminology not in the original query
- **Key references**: Search for frequently cited foundational works

Construct 2-3 new queries based on these discoveries.

### 3. Gap Analysis Search

After rounds 1-2, identify uncovered areas:
- Compare collected topics against the research plan dimensions
- Formulate targeted queries for missing perspectives
- Search for contrarian or critical viewpoints on the topic

### 4. Quality Filtering

For each result, assess quality:
- **High**: Top-tier venues (Nature, Science, ICML, NeurIPS, etc.), >100 citations
- **Medium**: Reputable preprint servers (arXiv), established tech blogs, >10 citations
- **Low**: Unverified claims, outdated materials (>5 years without recent citations)

Prioritize high and medium quality sources. Include low quality only if they cover unique perspectives.

### 5. Output

Save all results following the standard output format defined in the Survey Agent's system prompt.
