import os
import pandas as pd
import dess.nlp as nlp
import data_pipeline_manager as dpm
import cse
from datetime import datetime
from dotenv import load_dotenv

# ========================================
# CONFIG
load_dotenv()
STORAGE_DIR = 'storage'
ERROR_FILE = f'{STORAGE_DIR}/errors.csv'
PROCESSED_FILE = f'{STORAGE_DIR}/processed.csv'
LOG_FILE = f'status.txt'
FILE_PATH = f'{STORAGE_DIR}/dataset/test_cron copy.parquet'
# ========================================
#TODO: Create column headers (snippet_1 -> snippet_4 first time before running this script)
def _get_next_chunk_for_api_call():
    df = pd.read_parquet(FILE_PATH)
    # df['rawText']=None
    empty_df = df[df['snippet_1'].isna() | (df['snippet_1'] == '')]
    rows_per_day = min(len(empty_df),4)
    today_df = empty_df.iloc[:rows_per_day]
    # remaining_df = empty_df.iloc[rows_per_day:]
    return today_df

def end_to_end_workflow():
    # 1. Get today's chunk [constrained by rate limits and remaning count]
    df = _get_next_chunk_for_api_call()
    with open(LOG_FILE, "a") as log:
        log.write("=====[Starting] Phase 1: API Calls========\n")
    # # 2. Make API Calls
    cse.populate_rawText_col(df)
    df_errors = df[df['rawText'].isna()] 
    
    write_header_error = not os.path.exists(ERROR_FILE)
    df_errors.to_csv(ERROR_FILE,mode='a',index=False,header=write_header_error)
    with open(LOG_FILE, "a") as log:
        log.write("=====[COMPLETE] Phase 1: API Calls========\n")
    # 3. Run department extraction methodology
    df_non_errors = df.dropna(subset=['rawText'])
    with open(LOG_FILE, "a") as log:
        log.write("=====[Starting] Phase 2: Populate Department Variables========\n")
    nlp.extract_department_information(df_non_errors)
    with open(LOG_FILE, "a") as log:
        log.write("=====[COMPLETE] Phase 2: Populate Department Variables========\n")
    # 4. Update out files
    with open(LOG_FILE, "a") as log:
        log.write("=====[Starting] Phase 3: Updating and uploading parquet to drop box========\n")
    dpm.update_parquet_file(df_non_errors, FILE_PATH)

    # 5. Cloud Sync and local cleanup
    dbx = dpm.dropbox_oauth()
    dpm.push_new_dataset_files_to_dropbox(dbx)
    with open(LOG_FILE, "a") as log:
        log.write("=====[COMPLETE] Phase 2: Updating and uploading parquet to drop box========\n")
    # 6. Logging & Metrics
    today_date = datetime.now().strftime("%Y-%m-%d")
    processed_count = len(df_non_errors)
    with open(LOG_FILE, "a") as log:
        log.write(f"{today_date}: Processed {processed_count} rows. Error {len(df) - processed_count} rows.\n")
    
if __name__== "__main__":
    load_dotenv()
    end_to_end_workflow()
