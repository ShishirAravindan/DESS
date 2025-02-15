import cse
import pandas as pd
import numpy as np
import os
import ast
import dess.nlp as nlp
import datetime
import sys
import data_pipeline_manager as dpm
from dotenv import load_dotenv
import dropbox

load_dotenv()
ERROR_FILE = 'storage/errors.csv'
PROCESSED_FILE = 'storage/processed.csv'
LOG_FILE = 'status.txt'
TEMP_STATA = 'storage/test_cron_copy.dta'
def _get_dataframe_from_strata(filepath):
    df = pd.read_stata(filepath)
    return df

def end_to_end_workflow(filepath):
    df_stata = _get_dataframe_from_strata(filepath)
    #Exit strategy for the cronjob
    if len(df_stata) == 0:
        sys.exit(0)        
    rows_per_day = min(len(df_stata),2)
    today_df = df_stata.iloc[:rows_per_day]
    remaining_df = df_stata.iloc[rows_per_day:]
    with open(LOG_FILE, "a") as log:
        log.write("=======Starting with rawText population of the batch====== \n")
    cse.populate_rawText_col(today_df)
    print('Done with phase 1')
    with open(LOG_FILE, "a") as log:
        log.write("Raw Text population complete \n")
    df_errors = today_df[today_df['rawText'].isna()] 
    write_header_error = not os.path.exists(ERROR_FILE)
    df_errors.to_csv(ERROR_FILE,mode='a',index=False,header=write_header_error)
    df_non_errors = today_df.dropna(subset=['rawText'])
    #df_non_errors['rawText']= df_non_errors['rawText'].apply(ast.literal_eval)
    with open(LOG_FILE, "a") as log:
        log.write("Proceeding with department extraction \n")
    nlp.extract_department_information(df_non_errors)
    dpm.update_stata_file(df_non_errors,TEMP_STATA)
    dbx = dpm.dropbox_oauth()
    dpm.push_new_dataset_files_to_dropbox(dbx)
    remaining_df.to_stata(filepath)
    today_date = datetime.now().strftime("%Y-%m-%d")
    error_count = len(df_errors)
    processed_count = len(df_non_errors)
    with open(LOG_FILE, "a") as log:
        log.write(f"{today_date}: Processed {processed_count} rows check processed.csv \n Errored {error_count} rows check errors.csv \n ")
    
if __name__== "__main__":
    end_to_end_workflow(filepath="storage/test_cron.dta")

    

    
    
    
    
    
    
    
    