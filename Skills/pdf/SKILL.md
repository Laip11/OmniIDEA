---
name: pdf
description: Use this skill whenever the user uploads PDF files or the task involves reading, extracting text or tables from PDF documents. Call the pdf_extract_text tool to extract content from PDF files, then use the extracted text for analysis, survey, or report writing.
metadata:
  author: DeepResearch
  version: "1.0"
allowed-tools: pdf_extract_text
---

# PDF Content Extraction

## Overview

This skill enables reading and extracting text content from PDF files using the `pdf_extract_text` tool. Use it when:
- The user uploads PDF papers or reference documents to `Inputs/uploads/`
- The research task requires reading PDF literature
- You need to extract text or tables from PDF files for analysis

## Tool: pdf_extract_text

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file_path` | string | ✅ | Path to the PDF file. Can be absolute or relative to workspace (e.g., `Inputs/uploads/paper.pdf`) |
| `start_page` | int | ❌ | First page to extract (1-indexed). Defaults to first page |
| `end_page` | int | ❌ | Last page to extract (1-indexed). Defaults to last page |
| `extract_tables` | bool | ❌ | Whether to also extract tables as structured text. Defaults to false |

### Usage Examples

**Extract all text from a PDF:**
```
pdf_extract_text(file_path="Inputs/uploads/paper.pdf")
```

**Extract specific pages:**
```
pdf_extract_text(file_path="Inputs/uploads/paper.pdf", start_page=1, end_page=5)
```

**Extract text and tables:**
```
pdf_extract_text(file_path="Inputs/uploads/paper.pdf", extract_tables=True)
```

## Workflow

### 1. Identify PDF Files

Check for PDF files in the workspace:
- `Inputs/uploads/` — User-uploaded files
- Any other path the user specifies

### 2. Extract Content

Call `pdf_extract_text` for each PDF file:
- For short papers (< 20 pages): extract all pages at once
- For long documents (> 20 pages): extract in batches (e.g., pages 1-20, then 21-40) to avoid output truncation
- Set `extract_tables=True` if the document contains important data tables

### 3. Process Extracted Text

After extraction, use the text content for:
- **Survey phase**: Analyze the paper's methods, findings, and relevance
- **Idea phase**: Identify key insights and potential innovation directions
- **Report phase**: Cite specific findings with page references

### 4. Auto-Cached Content

The `pdf_extract_text` tool **automatically caches** extracted text to `Inputs/raw/pdf_cache/<filename>.txt`:
- e.g., extracting `Inputs/uploads/paper.pdf` → auto-saved to `Inputs/raw/pdf_cache/paper.txt`
- No manual save needed — caching happens on every extraction call
- Other agents (Idea, Assessment, Report) can read from the cache via `read_file` without re-extracting
- If you need to re-extract (e.g., with different page range or `extract_tables=True`), just call the tool again — the cache will be overwritten

## Tips

- If extraction returns "(No text content on this page)", the PDF may be a scanned image. Note this limitation in your analysis.
- For PDFs with complex layouts (multi-column, figures), the extracted text may not preserve the original formatting perfectly. Focus on the textual content.
- Always note the source file path and page numbers when citing extracted content.
