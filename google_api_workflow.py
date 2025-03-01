import os
import pandas as pd
import dess.nlp as nlp
import data_pipeline_manager as dpm
import cse
from dotenv import load_dotenv
import logging

# ========================================
# CONFIG
load_dotenv()
STORAGE_DIR = '/Users/akhil/Desktop/RA-Scraping/DESS/storage'
ERROR_FILE = f'{STORAGE_DIR}/errors.csv'
LOG_FILE = f'status.log'
FILE_PATH = f'{STORAGE_DIR}/dataset/akhil-toSearch-2025-02-11.parquet'
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# ========================================

def _get_next_chunk_for_api_call():
    df = pd.read_parquet(FILE_PATH)
    empty_df = df[df['snippet_1'].isna() | (df['snippet_1'] == '')]
    rows_per_day = min(len(empty_df), 100)
    today_df = empty_df.iloc[:rows_per_day]
    return today_df

def end_to_end_workflow():
    try:
        # 1. Get today's chunk [constrained by rate limits and remaning count]
        df = _get_next_chunk_for_api_call()
        
        # 2. Make API Calls
        logging.info("Starting Phase 1: Custom Search Engine API calls...")
        cse.populate_rawText_col(df)
        df_errors = df[df['rawText'].isna()] 
        
        write_header_error = not os.path.exists(ERROR_FILE)
        df_errors.to_csv(ERROR_FILE,mode='a',index=False,header=write_header_error)
        
        logging.info("[COMPLETE] Phase 1: API Calls")
        logging.info("Starting Phase 2: Department Extraction...")

        # 3. Run department extraction methodology
        df_non_errors = df.dropna(subset=['rawText'])
        nlp.extract_department_information(df_non_errors)

        logging.info("[COMPLETE] Phase 2: Populate Department Variables")

        # 4. Update out files
        dpm.update_parquet_file(df_non_errors, FILE_PATH)

        # 5. Cloud Sync and local cleanup
        logging.info("Starting Phase 3: Uploading to dropbox...")
        dbx = dpm.dropbox_oauth()
        dpm.push_new_dataset_files_to_dropbox(dbx)
        logging.info("[COMPLETE] Phase 3: Dropbox sync")
        
        # 6. Logging & Metrics
        processed_count = len(df_non_errors)
        logging.info(f"Processed {processed_count} rows. Error {len(df) - processed_count} rows.")
    except:
        folder = f'{STORAGE_DIR}/dataset'
        for file in os.listdir(folder):
            if file.endswith('.csv'):
                os.remove(os.path.join(folder, file))
    
if __name__== "__main__":
    end_to_end_workflow()