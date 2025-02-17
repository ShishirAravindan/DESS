import os
import pandas as pd
import dess.nlp as nlp
from data_pipeline_manager import import_files_from_dropbox, dropbox_oauth, upload_large_file

LOCAL_DATASET_DIR = "storage/dataset"
DROPBOX_DATASET_DIR = os.path.join(os.getenv("DROPBOX_FOLDER"), 'dataset')
OUTPUT_FILE_NAME = "merged_data.dta"

def _convert_boolean_columns(df):
    """Convert boolean columns to int8 (0/1)."""
    bool_columns = [
        'isProfessor', 'isInstructor', 'isEmeritus', 'isAssistantProf',
        'isAssociateProf', 'isFullProf', 'isClinicalProf', 'isResearcher',
        'isRetired'
    ]
    for col in bool_columns:
        df[col] = df[col].astype('int8')
    return df

def _convert_float_columns(df):
    """Convert float32 columns to float64 (Stata's double)."""
    df['professor'] = df['professor'].astype('float64')
    return df

def _process_raw_text(df):
    """Process rawText column into separate snippet columns."""
    snippet_1, snippet_2, snippet_3, snippet_4 = zip(*[
        rawText for rawText in df['rawText'] if len(rawText) == 4
    ])
    
    df = df.drop(columns='rawText')
    df[['snippet_1', 'snippet_2', 'snippet_3', 'snippet_4']] = list(
        zip(snippet_1, snippet_2, snippet_3, snippet_4)
    )
    return df

def _process_string_columns(df):
    """Process string columns to comply with Stata requirements."""
    string_columns = [
        'university', 'lastname', 'firstname', 'id_text', 'department_textual',
        'department_keyword', 'snippet_1', 'snippet_2', 'snippet_3', 'snippet_4'
    ]
    
    for col in string_columns:
        # Truncate to Stata's maximum length (244 characters)
        df[col] = df[col].astype(str).str.slice(0, 244)
        
        # Clean special characters
        df[col] = (df[col]
                  .str.replace(r'[^\x00-\x7F]+', '', regex=True)  # Remove non-ASCII
                  .str.replace(r'[\r\n\t]+', ' ', regex=True))    # Replace newlines/tabs
    return df

def merge_parquet_files():
    """Merges all parquet files in the storage directory."""
    parquet_files = [f for f in os.listdir(LOCAL_DATASET_DIR) if f.endswith('.parquet')]
    if not parquet_files:
        raise FileNotFoundError("No parquet files found in the storage directory.")

    # Read and merge all parquet files
    df_list = [pd.read_parquet(os.path.join(LOCAL_DATASET_DIR, file)) for file in parquet_files]
    merged_df = pd.concat(df_list, ignore_index=True)
    return merged_df

def convert_to_stata(df):
    """Converts the DataFrame to Stata format with proper data type handling."""
    # Create a copy to avoid modifying the original dataframe
    df_stata = df.copy()
    
    df_stata = _convert_boolean_columns(df_stata)
    df_stata = _convert_float_columns(df_stata)
    df_stata = _process_raw_text(df_stata)
    df_stata = _process_string_columns(df_stata)
    
    return save_to_stata(df_stata)

def save_to_stata(df):
    """Save DataFrame to Stata format."""
    stata_file_path = os.path.join(LOCAL_DATASET_DIR, OUTPUT_FILE_NAME)
    df.to_stata(
        stata_file_path,
        version=118,
        write_index=False
    )
    return stata_file_path
    
def main():
    """Main function to orchestrate the conversion process."""
    dbx = dropbox_oauth()
    # Step 1: Import files from Dropbox
    import_files_from_dropbox(dbx)

    # Step 2: Merge parquet files
    merged_df = merge_parquet_files()

    nlp.extract_department_information(merged_df)

    # Step 3: Convert to Stata format
    stata_file_path = convert_to_stata(merged_df)
    print(stata_file_path)

    # Step 4: Upload the Stata file to Dropbox
    DROPBOX_UPLOAD_FILE_PATH = os.path.join(DROPBOX_DATASET_DIR, OUTPUT_FILE_NAME)
    upload_large_file(dbx, stata_file_path, DROPBOX_UPLOAD_FILE_PATH)

if __name__ == "__main__":
    main()
