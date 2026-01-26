# Rate My Professor API Experiment

> **Status: Successful Supplementary Approach** - This experiment provides an alternative data source for department extraction by querying Rate My Professor's database. It complements the main pipeline (Google Custom Search + NLP) by providing department information for professors who have RMP profiles.

## Overview

This module queries the Rate My Professor API to extract department information for faculty members. It's particularly effective for professors with active RMP profiles and serves as a supplementary data source to the main DESS pipeline.

## The Problem

Efficiently querying RMP for department data across thousands of faculty members.

The naive approach (in `mvp.rs`) makes individual API calls for each professor using `get_teacher_summary_and_save()`. This becomes inefficient and may trigger rate limiting when you have:
- Multiple professors from the same university
- Large batches of queries
- Repeated queries to the same institutions

## The Solution

Use `get_professor_list()` to fetch ALL professors from a university in a single API call, then filter locally.

## Available Programs

### 1. Original MVP (`mvp.rs`)
```bash
cargo run --bin mvp "Queens College" "Ross Greenberg"
```
- Makes individual API calls per professor
- Good for single queries
- ❌ Inefficient for multiple professors from same university

### 2. Efficient Single University (`efficient_mvp.rs`)
```bash
cargo run --bin efficient "Queens College" "Ross Greenberg" "Rebecca Nelson"
```
- Single API call per university
- Local filtering for multiple professors
- ✅ Efficient for multiple professors from same university
- Includes partial name matching
- Set `SHOW_ALL=1` to see all available professors

### 3. Batch Processing (`batch_mvp.rs`)
```bash
cargo run --bin batch sample_queries.csv
```
- Processes CSV files with university,professor pairs
- Groups queries by university automatically
- Single API call per unique university
- ✅ Most efficient for large datasets
- Maintains input order in output
- Includes summary statistics

## Sample CSV Format

```csv
# Comments start with #
Queens College,Ross Greenberg
Queens College,Rebecca Nelson  
City College of New York,Douglas Troeger
Hunter College,Maria Rodriguez
```

## Key Benefits

1. **Reduced API Calls**: Instead of N calls for N professors, make 1 call per unique university
2. **Rate Limiting Avoidance**: Batch requests reduce server load
3. **Offline Filtering**: Fast local name matching after initial data fetch
4. **Fuzzy Matching**: Handles slight name variations
5. **Scalability**: Efficient for hundreds or thousands of queries

## API Efficiency Comparison

| Scenario | Original | Efficient | Batch |
|----------|----------|-----------|-------|
| 1 professor | 1 API call | 1 API call | 1 API call |
| 5 professors, same university | 5 API calls | 1 API call | 1 API call |
| 100 professors, 10 universities | 100 API calls | 10 API calls | 10 API calls |

## Performance Tips

1. **Group by University**: The batch processor automatically groups queries
2. **Respect Rate Limits**: Built-in 500ms delays between university queries
3. **Cache Results**: Consider saving `get_professor_list()` results locally for repeated use
4. **Name Normalization**: All matching is case-insensitive with whitespace handling

## Dependencies

```toml
rate_my_professor_api_rs = "0.1.5"
tokio = { version = "1", features = ["full"] }
anyhow = "1"
```

## Python Data Pipeline: `run_rmp_workflow.py`

The `run_rmp_workflow.py` script provides a complete end-to-end data pipeline that orchestrates the RMP data processing workflow, integrating the efficient Rust API calls with file management and cloud storage.

### Overview

This Python script automates the entire RMP data enrichment process:
1. **Data Discovery**: Automatically finds Stata (.dta) files to process
2. **Format Conversion**: Converts Stata files to Parquet for efficient processing
3. **API Processing**: Runs the optimized Rust binary to fetch RMP data
4. **Result Merging**: Combines API results back into the original Stata format
5. **Cloud Storage**: Uploads processed files to Dropbox for team access

### Prerequisites

Before running the workflow, ensure you have:

#### Required Files and Setup
- **`.dta` files**: Place your Stata files containing faculty data in the `storage/` directory
- **Environment Configuration**: Create a `.env` file in the project root with:
  ```bash
  DROPBOX_FOLDER=your_dropbox_folder_name
  ```
- **Dropbox Authentication**: Set up Dropbox OAuth tokens (handled by `data_pipeline_manager.py`)
- **Rust Binary**: The script will automatically compile the Rust binary if not present

#### Required Columns in Input Data
Your `.dta` files must contain these columns:
- `university`: University name
- `firstname`: Professor's first name
- `lastname`: Professor's last name

#### Dependencies
```bash
pip install pandas dropbox python-dotenv pyarrow
```

### Usage

#### Basic Usage
```bash
python run_rmp_workflow.py
```

The script will automatically:
1. Authenticate with Dropbox
2. Find all `.dta` files in the `storage/` directory
3. Process each file through the complete pipeline
4. Upload results back to Dropbox

#### What the Script Does

For each `.dta` file found, the script performs these steps:

1. **File Conversion**
   ```python
   # Converts .dta to .parquet with required columns
   df = pd.read_stata(dta_path)
   df_to_process = df[['university', 'firstname', 'lastname']]
   df_to_process.to_parquet(parquet_path)
   ```

2. **Rust Processing**
   ```bash
   # Runs the efficient Rust binary on the parquet file
   cargo run --release --bin batch [parquet_file]
   ```
   This leverages the optimized batch processing that groups professors by university.

3. **Result Merging**
   - Uses `export.py` functions to merge RMP data back into original Stata format
   - Adds RMP columns (ratings, department info, etc.) to the original dataset
   - Saves as `{original_name}_v2.dta`

4. **Cloud Upload**
   ```python
   # Uploads processed file to Dropbox
   upload_large_file(dbx, output_dta_path, dropbox_file_path)
   ```

### Output Files

The script generates several output files for each input:

| File | Location | Description |
|------|----------|-------------|
| `{name}.parquet` | `storage/` | Temporary Parquet file for processing |
| `{name}_v2.dta` | `storage/export/` | Processed Stata file with RMP data |
| `{name}_v2.dta` | Dropbox | Cloud backup of processed file |

### Error Handling

The script includes comprehensive error handling:
- **File Processing Errors**: Continues with next file if one fails
- **API Rate Limits**: Handled by the Rust binary's built-in delays
- **Network Issues**: Dropbox operations include retry logic
- **Compilation**: Automatically compiles Rust binary if missing

### Integration with DESS Project

This script integrates with the larger DESS (Department Extraction using Search and spaCy) project:

- Uses shared Dropbox authentication from `data_pipeline_manager.py`
- Complements the web scraping department extraction workflow
- Maintains consistent file naming and storage patterns
- Leverages the same cloud storage infrastructure

### Monitoring and Logs

The script provides detailed console output:
- Progress indicators for each processing step
- File conversion status
- Upload confirmations
- Error messages with context

### Performance Considerations

- **Batch Processing**: Groups professors by university for efficient API usage
- **Parallel Ready**: Each file is processed independently (could be parallelized)
- **Memory Efficient**: Uses Parquet format for large datasets
- **Resume Capability**: Can be restarted if interrupted (will skip already processed files)

### Troubleshooting

**Common Issues:**
1. **"Storage directory not found"**: Ensure `storage/` directory exists
2. **"No .dta files found"**: Check file extensions and location
3. **"Dropbox authentication failed"**: Verify OAuth setup in `data_pipeline_manager.py`
4. **"Rust compilation failed"**: Ensure Rust toolchain is installed

**Debug Mode:**
Set environment variable `DEBUG=1` for verbose output from the Rust binary.

## Documentation References

- [Rate My Professor API Rust Crate](https://docs.rs/rate_my_professor_api_rs/0.1.5/rateMyProfessorApi_rs/methods/struct.RateMyProfessor.html)
- Key method: `get_professor_list()` returns `Vec<ProfessorList>` with department info
- Each `ProfessorList` contains: `first_name`, `last_name`, `department`, `avg_rating`, etc.

This approach scales from single queries to enterprise-level batch processing while being respectful to the Rate My Professor infrastructure. 