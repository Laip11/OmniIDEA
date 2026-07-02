---
name: google-scholar-search
description: Use this skill when you need to search academic papers on Google Scholar. Provides keyword search, author filtering, year range filtering, author profile retrieval, and JSON export. Ideal for systematic literature review and citation tracking.
metadata:
  author: JackKuo666
  version: "0.1.0"
  source: https://termo.ai/skills/google-scholar-search-skill
allowed-tools: tavily_search
---

# Google Scholar Search

AI-Powered Academic Literature Search Tool — search academic papers through Google Scholar with advanced filtering capabilities.

## When to Use

Use this skill when:
- You need to find academic papers on a specific research topic
- You want to filter papers by author, year range, or keywords
- You need to retrieve an author's profile, citation counts, and publication list
- You want structured JSON output for further processing

## Instructions

### 1. Keyword Search

Search papers by keywords, titles, or topics. Construct your `tavily_search` queries targeting Google Scholar:

- Use `site:scholar.google.com` prefix to focus on Scholar results
- Include domain-specific terminology for precision
- Append year constraints for recency (e.g., "2022-2026")

**Examples:**
- `"transformer architecture" site:scholar.google.com`
- `"machine learning in healthcare" site:scholar.google.com 2023-2026`

### 2. Advanced Search with Author Filter

To find papers by a specific author on a topic:

- Combine author name with topic keywords
- Use `author:"Name"` syntax in queries

**Examples:**
- `author:"Yann LeCun" "deep learning" site:scholar.google.com`
- `author:"Andrew Ng" "reinforcement learning" site:scholar.google.com 2020-2025`

### 3. Year Range Filtering

Narrow results to a specific publication period:

- Append year range to queries: `"topic" 2018..2023`
- For recent work only: `"topic" after:2023`

**Examples:**
- `"generative adversarial networks" 2018..2023 site:scholar.google.com`
- `"large language models" after:2024 site:scholar.google.com`

### 4. Author Profile Retrieval

To get detailed information about a researcher:

- Search for `"Author Name" Google Scholar profile`
- Extract: affiliation, research interests, citation counts, h-index, recent publications

**Example:**
- `"Geoffrey Hinton" Google Scholar profile citations publications`

### 5. Output Format

For each paper found, extract and record:

```json
{
  "title": "Paper title",
  "authors": "Author list",
  "year": "Publication year",
  "abstract": "Paper abstract or summary",
  "url": "Direct link to the paper",
  "citations": "Citation count (if available)",
  "source": "google_scholar",
  "venue": "Journal or conference name (if available)"
}
```

For author profiles:

```json
{
  "name": "Author name",
  "affiliation": "Institution",
  "interests": ["Research area 1", "Research area 2"],
  "total_citations": 150000,
  "h_index": 100,
  "recent_papers": [
    {"title": "Paper title", "year": 2024, "citations": 150}
  ]
}
```

### 6. Important Notes

- Google Scholar does not provide an official API — results are obtained via web search
- Search results may vary; for more stable access, combine with Semantic Scholar or PubMed searches
- Always cross-reference findings with other academic databases for completeness
- Use `web_fetch` to access full paper pages when abstracts from search results are insufficient

## Related Skills

- `deep-search` — Multi-round deep web search with iterative query refinement
- Consider combining this skill with `deep-search` for exhaustive literature discovery
