# DESS Core Modules

This folder contains the core modules for the Department Extraction pipeline.

## Modules

### `nlp.py` - Department Extraction

The primary extraction engine using pattern matching and keyword classification.

**Key functions:**
- `extract_department_information(df)` - Main entry point that populates department and faculty columns
- `populate_faculty_columns(df)` - Extracts faculty classification flags
- `populate_dummy_variables(df)` - Identifies professor types (assistant, associate, full, etc.)
- `populate_department_variables(df)` - Extracts department using regex patterns and keyword matching

**How it works:**
1. Applies regex patterns to identify department mentions in text
2. Matches extracted terms against a keyword whitelist with precision levels
3. Populates columns: `isProfessor`, `isInstructor`, `isEmeritus`, `department_textual`, `department_keyword`, etc.

### `search.py` - Web Scraping (Archived)

Original Selenium-based Google search scraper.

> **Note**: This module was replaced by `cse.py` (Google Custom Search API) due to reliability issues with browser automation. Kept for reference only.

**What it did:**
- Used Selenium WebDriver to automate Google searches
- Extracted search result snippets for each faculty member
- Supported parallel processing with resume capability

### `stats.py` - Progress Tracking Utilities

Helper functions for monitoring pipeline progress and generating statistics.

**Key functions:**
- `get_expected_file_split_stats(df_master, df_c, df_r)` - Shows data split across complete/reprocess/todo
- `get_chunk_processing_stats(df, CHUNK_SIZE)` - Monitors chunk-based processing progress
- `get_dataset_stats(file_path)` - Generates extraction accuracy statistics

## Usage

The main workflow uses these modules through `google_api_workflow.py` or `workflow.ipynb`:

```python
import dess.nlp as nlp

# After populating rawText via CSE API
nlp.extract_department_information(df)
```

## Dependencies

- pandas
- spaCy (for NLP utilities)
- Environment variable: `WHITELIST_FILE_PATH` pointing to keyword whitelist CSV
