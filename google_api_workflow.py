import os
import pandas as pd
import dess.nlp as nlp
import data_pipeline_manager as dpm
import cse
import datetime
from dotenv import load_dotenv

# ========================================
# CONFIG
load_dotenv()
STORAGE_DIR = 'storage'
ERROR_FILE = f'{STORAGE_DIR}/errors.csv'
PROCESSED_FILE = f'{STORAGE_DIR}/processed.csv'
LOG_FILE = f'{STORAGE_DIR}/API_WORKFLOW_shishir.LOG'
FILE_PATH = f'{STORAGE_DIR}/dataset/DepartmenttoSearch_January2025_0129.dta' # with rows to be scraped
MASTER_FILE = f'{STORAGE_DIR}/DepartmenttoSearch_January2025_0129.dta' # rows constantly updated with results
# ========================================

def _get_next_chunk_for_api_call():
    df = pd.read_stata(FILE_PATH)
    rows_per_day = min(len(df),4)
    today_df = df.iloc[:rows_per_day]
    remaining_df = df.iloc[rows_per_day:]
    print(len(df), len(remaining_df))
    remaining_df.to_stata(FILE_PATH, version=118)
    return today_df

def end_to_end_workflow():
    # 1. Get today's chunk [constrained by rate limits and remaning count]
    df = _get_next_chunk_for_api_call()
    
    # # 2. Make API Calls
    cse.populate_rawText_col(df)
    df_errors = df[df['rawText'].isna()] 
    
    write_header_error = not os.path.exists(ERROR_FILE)
    df_errors.to_csv(ERROR_FILE,mode='a',index=False,header=write_header_error)
    
    print("[COMPLETE] Phase 1: API Calls")

    # 3. Run department extraction methodology
    df_non_errors = df.dropna(subset=['rawText'])
    nlp.extract_department_information(df_non_errors)

    print("[COMPLETE] Phase 2: Populate Department Variables")

    # 4. Update out files
    dpm.update_stata_file(df_non_errors, MASTER_FILE)

    # 5. Cloud Sync and local cleanup
    dbx = dpm.dropbox_oauth()
    dpm.push_new_dataset_files_to_dropbox(dbx)
    
    # 6. Logging & Metrics
    today_date = datetime.now().strftime("%Y-%m-%d")
    processed_count = len(df_non_errors)
    with open(LOG_FILE, "a") as log:
        log.write(f"{today_date}: Processed {processed_count} rows. Error {len(df) - processed_count} rows.")
    
if __name__== "__main__":
    end_to_end_workflow()
