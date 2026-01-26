# Reproduction Guide

This guide provides step-by-step instructions for reproducing the DESS department extraction pipeline or extending it with new data.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Input Data Format](#input-data-format)
3. [Running the Pipeline](#running-the-pipeline)
4. [Understanding the Output](#understanding-the-output)
5. [Extending the Dataset](#extending-the-dataset)
6. [Cloud Deployment](#cloud-deployment)
7. [Troubleshooting](#troubleshooting)

---

## Environment Setup

### 1. Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Google Custom Search API Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable the **Custom Search API**
4. Create credentials (API Key)
5. Go to [Programmable Search Engine](https://programmablesearchengine.google.com/)
6. Create a new search engine configured to search the entire web
7. Note the **Search Engine ID**

### 3. Dropbox Setup (Optional)

For cloud backup functionality:

1. Go to [Dropbox App Console](https://www.dropbox.com/developers/apps)
2. Create a new app with "Full Dropbox" access
3. Generate an access token or use OAuth flow
4. Run `python dropbox_auth.py` to set up OAuth tokens

### 4. Environment Configuration

Create a `.env` file in the project root:

```bash
# Storage directory (must exist)
STORAGE_DIR=/path/to/your/storage

# Google Custom Search API
CSE_API_KEY=AIza...your_key
SEARCH_ENGINE_ID=abc123...your_engine_id

# Dropbox (optional)
DROPBOX_APP_KEY=your_app_key
DROPBOX_APP_SECRET=your_app_secret
DROPBOX_REFRESH_TOKEN=your_refresh_token
DROPBOX_FOLDER=DESS_Data
```

### 5. Storage Directory Structure

Create the following directory structure:

```
storage/
├── input.dta                 # Your input Stata file
├── complete.parquet          # Processed records (create empty initially)
├── reprocess.parquet         # Failed records for retry (create empty initially)
├── department-whitelist.pkl  # Keyword whitelist (generated from Excel)
└── dataset/                  # CSE API result cache
```

To create initial empty parquet files:

```python
import pandas as pd

# Define the schema
columns = ['id_text', 'firstname', 'lastname', 'university', 
           'isProfessor', 'rawText', 'department_textual', 'department_keyword']

# Create empty DataFrames
pd.DataFrame(columns=columns).to_parquet('storage/complete.parquet')
pd.DataFrame(columns=columns).to_parquet('storage/reprocess.parquet')
```

---

## Input Data Format

### Required Columns

Your input Stata file must contain:

| Column | Type | Description |
|--------|------|-------------|
| `id_text` | string | Unique identifier in format "Firstname Lastname University" |
| `firstname` | string | Faculty member's first name |
| `lastname` | string | Faculty member's last name |
| `university` | string | University name |

### Example

```
id_text                                    | firstname | lastname  | university
-------------------------------------------|-----------|-----------|------------------
John Smith Harvard University              | John      | Smith     | Harvard University
Jane Doe MIT                               | Jane      | Doe       | MIT
```

### Keyword Whitelist

The department keyword whitelist is stored as a pickle file with precision levels:

```python
# Structure: {precision_level: [keywords]}
{
    1: ['economics', 'biology', 'chemistry', ...],  # High confidence
    2: ['science', 'arts', 'engineering', ...],     # Medium confidence
    3: ['studies', 'research', ...]                 # Low confidence
}
```

To update from Excel:

```python
import dess.nlp as nlp
nlp.create_keyword_dict_file('storage/keywordList.xlsx')
```

---

## Running the Pipeline

### Option 1: Automated Workflow (Recommended)

For batch processing with automatic rate limiting:

```bash
python google_api_workflow.py
```

This will:
1. Load unprocessed records (respecting daily API limits)
2. Query Google CSE API for each faculty member
3. Extract department information using NLP
4. Save results to parquet files
5. Sync to Dropbox (if configured)

### Option 2: Interactive Notebook

For step-by-step control:

```bash
jupyter notebook workflow.ipynb
```

Follow the notebook sections:
1. **Data Preprocessing** - Load and prepare input data
2. **Scraping** - Query CSE API (can run in separate terminal)
3. **Department Extraction** - Apply NLP extraction
4. **Data Merging** - Consolidate results
5. **Post-processing** - Generate stats and exports

### Option 3: Manual Steps

```python
import pandas as pd
import data_pipeline_manager as dpm
import dess.nlp as nlp
import cse

# 1. Load new records
df = dpm.get_new_rows()
df = dpm.prepare_dess_data_structure(df)

# 2. Query CSE API
cse.populate_rawText_col(df)

# 3. Extract departments
nlp.extract_department_information(df)

# 4. Save results
dpm.write_to_file('storage/complete.parquet', df)
```

---

## Understanding the Output

### Output Columns

| Column | Description |
|--------|-------------|
| `rawText` | List of 4 search result snippets |
| `isProfessor` | True if identified as professor/faculty |
| `isInstructor` | True if identified as instructor/lecturer |
| `isEmeritus` | True if emeritus status detected |
| `isAssistantProf` | True if assistant professor |
| `isAssociateProf` | True if associate professor |
| `isFullProf` | True if full professor |
| `isClinicalProf` | True if clinical professor |
| `isResearcher` | True if researcher indicators found |
| `isRetired` | True if retirement/deceased indicators found |
| `teaching_intensity` | Count of teaching-related terms |
| `department_textual` | Raw extracted department text |
| `isPrimaryPattern` | 1 if primary pattern, 0 if backup, -1 if none |
| `department_keyword` | Matched whitelist keyword |
| `keyword_precision` | Precision level (1-3, lower is better) |

### Interpreting Results

- **Successful extraction**: `department_keyword != "MISSING"`
- **Partial extraction**: `department_textual != "MISSING"` but keyword missing
- **No extraction**: Both fields are "MISSING"

### Quality Metrics

Use `stats.py` to evaluate results:

```python
import dess.stats as stats

# Overall progress
stats.get_expected_file_split_stats(df_master, df_complete, df_reprocess)

# Extraction rates
stats.get_dataset_stats('storage/complete.parquet')
```

---

## Extending the Dataset

### Adding New Universities

1. Prepare a new Stata file with faculty data
2. Place in `storage/input.dta` (or update path in code)
3. Run the pipeline - it will automatically identify new records

### Updating Keyword Whitelist

1. Edit `storage/keywordList.xlsx` with new department keywords
2. Assign precision levels (1=high, 2=medium, 3=low confidence)
3. Regenerate the whitelist:

```python
import dess.nlp as nlp
nlp.create_keyword_dict_file('storage/keywordList.xlsx')
```

### Improving Extraction Patterns

Edit `dess/nlp.py` to add new patterns:

```python
DEPARTMENT_PATTERNS = {
    'primary': [
        # Add new high-confidence patterns here
        r'your_new_pattern ([A-Za-z]+)',
    ],
    'backup': [
        # Add new contextual patterns here
    ]
}
```

---

## Cloud Deployment

### GCP VM Setup (Brief)

For unattended batch processing, the pipeline was deployed on a GCP VM:

1. **VM Configuration**: e2-small instance sufficient for API-bound workload
2. **Environment**: Python 3.9+ with dependencies installed
3. **Storage**: Local SSD or persistent disk for storage directory
4. **Execution**: Run `google_api_workflow.py` via cron or systemd timer
5. **Monitoring**: Check logs at `storage/API_WORKFLOW.LOG`

### Key Considerations

- **Rate Limits**: Free tier allows 100 queries/day; paid tier up to 10,000/day
- **Cost**: $5 per 1,000 queries after free tier
- **Reliability**: Use Dropbox sync for backup; restart from last checkpoint on failure
- **Logging**: All operations logged to file for debugging

### Example Cron Setup

```bash
# Run daily at 2 AM
0 2 * * * cd /path/to/dess && /path/to/venv/bin/python google_api_workflow.py >> /var/log/dess.log 2>&1
```

---

## Troubleshooting

### Common Issues

**"API call failed with status code: 403"**
- Check API key is valid and has quota remaining
- Verify Custom Search API is enabled in Google Cloud Console

**"No new rows to process"**
- All records in input.dta already exist in complete.parquet or reprocess.parquet
- Check `id_text` column for exact matches

**"MISSING department extraction"**
- Faculty member may lack web presence
- Search results may not contain department information
- Consider manual review or alternative data sources (e.g., RMP)

**Empty rawText results**
- API returned no results for the query
- Try searching manually to verify faculty web presence

### Debug Mode

Add logging to see detailed execution:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Manual Verification

To verify extraction for a specific faculty member:

```python
import cse
import dess.nlp as nlp
import pandas as pd

# Test single query
test_df = pd.DataFrame({'id_text': ['John Smith Harvard University']})
cse.populate_rawText_col(test_df)
nlp.extract_department_information(test_df)
print(test_df[['id_text', 'department_textual', 'department_keyword']])
```
