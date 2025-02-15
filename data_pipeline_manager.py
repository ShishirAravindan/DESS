import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import dropbox
from dropbox import DropboxOAuth2FlowNoRedirect

load_dotenv()

STORAGE_DIR = "storage/"

def get_new_rows():
    """Reads the master (stata) dataset and returns new rows not present in 'complete' or 'reprocess' files."""
    df_master = pd.read_stata(f"{STORAGE_DIR}/input.dta")
    df_c = pd.read_parquet(f"{STORAGE_DIR}/complete.parquet")
    df_r = pd.read_parquet(f"{STORAGE_DIR}/reprocess.parquet")
    # Combine id_text values from df_c and df_r into a single set
    id_text_set = set(df_c['id_text'].astype(str)).union(set(df_r['id_text'].astype(str)))

    df_u = df_master[~df_master['id_text'].astype(str).isin(id_text_set)]

    return df_u

def write_to_file(file_path: str, df: pd.DataFrame, overwrite: bool = False):
    """Writes DataFrame to a Parquet file, either overwriting or 
    appending to the file if it exists."""
    if overwrite or not os.path.exists(file_path):
        action = "CREATING NEW FILE"
    else:
        action = "APPENDING TO FILE"
        existing_df = pd.read_parquet(file_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    print(f"{action}: {file_path}")
    df.to_parquet(file_path)

def prepare_dess_data_structure(df: pd.DataFrame):
    """Adds custom DESS-related columns to the given DataFrame. This may change
    but generally includes: 'isProfessor', 'isProfessor2', 'rawText', and 'department'."""
    print(df.columns)
     # Normalize formatting in the 'id_text' column
    df['id_text'] = df['id_text'].str.strip()
    # Add new empty columns directly to the DataFrame
    df['isProfessor'] = None
    df['isProfessor2'] = None
    df['rawText'] = None           
    df['department'] = ""
    return df

def get_merged_data_from_parallel_scrape(df1: pd.DataFrame, df2: pd.DataFrame, split_ratio: float =0.5):
    """Merges two DataFrames that were scraped in parallel to populate different sections of rawText."""
    if df1.shape != df2.shape or set(df1.columns) != set(df2.columns):
        raise ValueError("DataFrames must have the same shape and columns to be merged.")
    
    split_index = int(len(df1) * split_ratio)
    merged_df = df1.copy()
    merged_df.loc[split_index:, 'rawText'] = df2.loc[split_index:, 'rawText']
    return merged_df

def update_internal_files(df_c: pd.DataFrame, df_r: pd.DataFrame, df_u: pd.DataFrame):
    """Updates the internal files with the given DataFrames."""
    # TODO: Test
    new_non_empty_rawText_rows = df_u[df_u['rawText'].apply(lambda x: isinstance(x, np.ndarray) and len(x) > 0)]
    new_empty_rawText_rows = df_u[~df_u['id_text'].isin(new_non_empty_rawText_rows['id_text'])]

    # Merging to complete.parquet + error checking
    updated_df_c, completed_conflicts = _safe_merge(df_c, new_non_empty_rawText_rows)
    if len(completed_conflicts):
        error_file_path = os.path.join(STORAGE_DIR, 'completed_conflicts.csv')
        completed_conflicts.to_csv(error_file_path, index = False)
        print(f"{len(completed_conflicts)} conflicts found updating complete.parquet. Conflicting rows saved to {error_file_path}.")

    # Merging to reprocess.parquet + error checking
    updated_df_r, reprocess_conflicts = _safe_merge(df_r, new_empty_rawText_rows)
    if len(reprocess_conflicts):
        error_file_path = os.path.join(STORAGE_DIR, 'reprocess_conflicts.csv')
        reprocess_conflicts.to_csv(error_file_path, index = False)
        print(f"{len(reprocess_conflicts)} conflicts found updating complete.parquet. Conflicting rows saved to {error_file_path}.")

    return updated_df_c, updated_df_r

def _safe_merge(df_master: pd.DataFrame, df: pd.DataFrame, col_name: str = 'id_text'):
    """Concatenates df to df_master, avoiding duplicate id_text entries."""
    conflicting_ids = df[col_name].isin(df_master[col_name])
    conflicts = df.loc[conflicting_ids, col_name]
    
    # Filter df to only non-conflicting rows
    df_to_add = df[~df[col_name].isin(conflicts)]
    
    # Concatenate safely
    df_combined = pd.concat([df_master, df_to_add], ignore_index=True)
    
    return df_combined, conflicts

def dropbox_oauth():
    auth_flow = DropboxOAuth2FlowNoRedirect(os.getenv("DROPBOX_APP_KEY"), os.getenv("DROPBOX_APP_SECRET"))

    authorize_url = auth_flow.start()
    print("1. Go to: " + authorize_url)
    print("2. Click \"Allow\" (you might have to log in first).")
    print("3. Copy the authorization code.")
    auth_code = input("Enter the authorization code here: ").strip()

    try:
        oauth_result = auth_flow.finish(auth_code)
    except Exception as e:
        print('Error: %s' % (e,))
        exit(1)

    with dropbox.Dropbox(oauth2_access_token=oauth_result.access_token) as dbx:
        dbx.users_get_current_account()
        print("Successfully set up client!")
        return dbx

def orchestrate_upload_workflow(overwrite=False, client=None):
    for file_name in os.listdir(STORAGE_DIR):
        if file_name == "input.dta" or file_name.startswith("."):
            print(f"Skipping: {file_name}")
            continue
        print(f'Uploading: {file_name}')
        file_path = os.path.join(STORAGE_DIR, file_name)
        _upload_file_to_dropbox(client, file_path, overwrite)

def _upload_file_to_dropbox(client, file_path, overwrite=False):
    """Uploads a file to Dropbox."""
    # Finding upload path
    dropbox_folder = os.getenv("DROPBOX_FOLDER")
    file_name = os.path.basename(file_path)
    upload_path = f"/{dropbox_folder}/{file_name}"

    if not dropbox_folder:
        raise ValueError("Dropbox folder must be set in the .env file.")

    # Configuring dropbox client
    if client: # if OAuth is successful
        dbx = client
    else:
        access_token = os.getenv("DROPBOX_ACCESS_TOKEN")
        if not access_token:
            raise ValueError("Access token must be set in the .env file.")
        dbx = dropbox.Dropbox(access_token) # dropbox client

    # Check if the file already exists in Dropbox
    try:
        dbx.files_get_metadata(upload_path)
        if overwrite:
            confirm = input(f"{file_name} already exists. Do you want to overwrite it? (y/n): ").strip().lower()
            if confirm != 'y':
                print(f"Skipping upload of {file_name}.")
                return
    except dropbox.exceptions.ApiError as e:
        print(f"NEW FILE FOUND: {file_name}")

    try:
        with open(file_path, 'rb') as f:
            mode = dropbox.files.WriteMode.overwrite if overwrite else dropbox.files.WriteMode.add
            dbx.files_upload(f.read(), upload_path, mode=mode)
        print(f"\tSuccessfully uploaded {file_name} to {upload_path}")
    except Exception as e:
        print(f"Error uploading {file_name} to Dropbox: {e}")
    
def create_stata_output_file(df,file_name):
    """Reads the dataframe and does some post-processing to ensure stata conversion is optimized."""
    snippet_1, snippet_2, snippet_3, snippet_4 = zip(*[rawText for rawText in df['rawText'] if len(rawText) == 4])
    
    df = df.drop(columns='rawText')
    df[['snippet_1', 'snippet_2', 'snippet_3', 'snippet_4']] = list(zip(snippet_1, snippet_2, snippet_3, snippet_4))
    
    stata_file_path = os.path.join(STORAGE_DIR, file_name)
    df.to_stata(stata_file_path, version=118)
    print(f"Successfully generated {stata_file_path}")
    return df

def import_files_from_dropbox(client=None):
    """Imports files from Dropbox into the storage directory."""
    if client:
        dbx = client
    else:
        access_token = os.getenv("DROPBOX_ACCESS_TOKEN")
        # Read & download files from Dropbox
        dbx = dropbox.Dropbox(access_token)
    
    dropbox_folder = os.getenv("DROPBOX_FOLDER")
    files = dbx.files_list_folder(f'/{dropbox_folder}/')
    for file in files.entries:
        if file.name.endswith('.parquet'):
            print(f"Downloading {file.path_lower}")
            dbx.files_download_to_file(os.path.join(STORAGE_DIR, file.name), file.path_lower)

def generate_sample_output_file(filename='sample.xlsx', n_samples=200, onlyIsProfessor=False):
    """Reads the complete Parquet file, randomly samples n_samples rows, and writes to an Excel file.
    If onlyIsProfessor is True, samples only from rows where isProfessor is True.
    """
    df = pd.read_parquet(f"{STORAGE_DIR}/complete.parquet")
    
    if onlyIsProfessor:
        df = df[df['isProfessor'] == True]
    
    sample_df = df.sample(n=n_samples)
    sample_df.to_excel(os.path.join(STORAGE_DIR, filename), index=False)
    print(f"Successfully generated {filename} with {n_samples} samples.")