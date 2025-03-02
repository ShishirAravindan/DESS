import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
import dropbox
from dropbox.files import WriteMode

load_dotenv()

STORAGE_DIR = os.getenv("STORAGE_DIR")
PARQUET_FILE_NAME = "shishir-toSearch-2025-02-11.parquet"

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
    """Create Dropbox client using refresh token stored in environment"""
    try:
        dbx = dropbox.Dropbox(
            app_key=os.getenv("DROPBOX_APP_KEY"),
            app_secret=os.getenv("DROPBOX_APP_SECRET"),
            oauth2_refresh_token=os.getenv("DROPBOX_REFRESH_TOKEN")
        )
        
        # Test the connection
        dbx.users_get_current_account()
        return dbx
        
    except Exception as e:
        print(f'Error creating Dropbox client: {e}')
        raise

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
            mode = WriteMode.overwrite if overwrite else WriteMode.add
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
    files = dbx.files_list_folder(f'/{dropbox_folder}/data-files/')
    for file in files.entries:
        if file.name.endswith('.parquet'):
            print(f"Downloading {file.path_lower}")
            dbx.files_download_to_file(os.path.join(STORAGE_DIR, 'dataset', file.name), file.path_lower)

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

def upload_large_file(dbx, file_path, dropbox_file_path):
    CHUNK_SIZE = 4 * 1024 * 1024  # 4MB chunks
    file_size = os.path.getsize(file_path)

    with open(file_path, "rb") as f:
        if file_size <= CHUNK_SIZE:
            # If the file is small enough, upload it directly
            dbx.files_upload(f.read(), dropbox_file_path, mode=WriteMode.overwrite)
        else:
            # Chunked upload
            upload_session_start_result = dbx.files_upload_session_start(f.read(CHUNK_SIZE))
            session_id = upload_session_start_result.session_id
            cursor = dropbox.files.UploadSessionCursor(session_id=session_id, offset=f.tell())
            commit = dropbox.files.CommitInfo(path=dropbox_file_path, mode=WriteMode.overwrite)

            # Initialize progress bar
            with tqdm(total=file_size, unit='B', unit_scale=True, desc="Uploading") as pbar:
                pbar.update(CHUNK_SIZE)  # Update for the first chunk

                while f.tell() < file_size:
                    if (file_size - f.tell()) <= CHUNK_SIZE:
                        dbx.files_upload_session_finish(f.read(CHUNK_SIZE), cursor, commit)
                    else:
                        dbx.files_upload_session_append(f.read(CHUNK_SIZE), cursor.session_id, cursor.offset)
                        cursor.offset = f.tell()
                    
                    pbar.update(CHUNK_SIZE)  # Update progress bar after each chunk

def push_new_dataset_files_to_dropbox(dbx):
    """Pushes CSV file generated from API calls to the dropbox folder and empties local cache"""
    # Define Dropbox folder and local cache path
    dropbox_folder = os.getenv("DROPBOX_FOLDER")
    local_cache_path = f"{STORAGE_DIR}/dataset"

    # Check for CSV files in the local cache
    csv_files = [f for f in os.listdir(local_cache_path) if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the local cache!")
    
    # Upload all CSV files from local cache with progress bar
    with tqdm(total=len(csv_files), desc="Uploading CSVs", unit="file") as pbar:
        for file_name in csv_files:
            local_file_path = os.path.join(local_cache_path, file_name)
            safe_file_name = file_name.replace(" ", "_")
            dropbox_file_path = os.path.join(dropbox_folder, "dataset", safe_file_name)

            with open(local_file_path, "rb") as f:
                dbx.files_upload(f.read(), dropbox_file_path, mode=WriteMode.add)
            
            # Update progress bar after each successful upload
            pbar.set_postfix(file=file_name)
            pbar.update(1)

    # Upload [updating] parquet file without deleting it
    file_path = os.path.join(STORAGE_DIR, 'dataset', PARQUET_FILE_NAME)
    dropbox_file_path = os.path.join(dropbox_folder, 'dataset', PARQUET_FILE_NAME)

    upload_large_file(dbx, file_path, dropbox_file_path)

    print("Upload complete! Removing local CSV files...")

    # clean up local cache once upload finishes
    for file_name in csv_files:
        local_file_path = os.path.join(local_cache_path, file_name)
        os.remove(local_file_path)

    print("Sync Complete!")

def update_parquet_file(df: pd.DataFrame, parquet_file_path: str):
    """
    Updates a Parquet file with information from the provided DataFrame, matching on id_text.
    
    Args:
        df (pd.DataFrame): DataFrame containing the new information
        parquet_file_path (str): Path to the Parquet file to be updated
    """
    # Read the existing Parquet file
    parquet_df = pd.read_parquet(parquet_file_path)
    
    # Ensure id_text is string type in both DataFrames
    parquet_df['id_text'] = parquet_df['id_text'].astype(str)
    df['id_text'] = df['id_text'].astype(str)
    
    # Convert rawText lists directly to snippet columns
    df[['snippet_1', 'snippet_2', 'snippet_3', 'snippet_4']] = pd.DataFrame(df['rawText'].tolist(), index=df.index)
    df = df.drop(columns='rawText')
    
    # Create a mapping of id_text to row updates
    update_dict = df.set_index('id_text').to_dict('index')
    
    # Update matching rows
    for idx, row in parquet_df.iterrows():
        if row['id_text'] in update_dict:
            for col, value in update_dict[row['id_text']].items():
                if col in parquet_df.columns:  # Update if column exists
                    parquet_df.at[idx, col] = value
                else:  # Add new column if it doesn't exist
                    parquet_df[col] = None  # Initialize new column with None
                    parquet_df.at[idx, col] = value  # Set the value for the new column
    
    # Save the updated DataFrame back to Parquet format
    parquet_df.to_parquet(parquet_file_path, index=False)
