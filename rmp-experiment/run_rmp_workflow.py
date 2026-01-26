#!/Users/shishiraravindan/Documents/work-RA/dess/.dessVenv/bin/python3

import os
import pandas as pd
import subprocess
import dropbox
from dotenv import load_dotenv
import sys

# Resolve important paths and ensure env is loaded from repo root
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(PROJECT_ROOT)


from data_pipeline_manager import dropbox_oauth, upload_large_file
from export import add_rmp_column_to_stata_file

load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
RUST_STORAGE_DIR = os.path.join(SCRIPT_DIR, "storage")
DROPBOX_FOLDER = os.getenv("DROPBOX_FOLDER")
RMP_RUST_BINARY = os.path.join(SCRIPT_DIR, "target", "release", "batch")

def list_dta_files_from_dropbox(dbx):
    """
    Lists .dta files from the specified Dropbox folder.
    """
    try:
        path = f"/{DROPBOX_FOLDER}/data-files/"
        result = dbx.files_list_folder(path)
        dta_files = [
            entry
            for entry in result.entries
            if isinstance(entry, dropbox.files.FileMetadata) and entry.name.endswith(".dta")
        ]
        return dta_files
    except dropbox.exceptions.ApiError as err:
        print(f"*** Dropbox API error: {err}")
        return []

def download_from_dropbox(dbx, dropbox_path, local_path):
    """
    Downloads a file from Dropbox.
    """
    try:
        dbx.files_download_to_file(local_path, dropbox_path)
        print(f"  Downloaded {dropbox_path} to {local_path}")
    except dropbox.exceptions.ApiError as err:
        print(f"*** Dropbox API error: {err}")
        raise

def convert_dta_to_parquet(dta_path, parquet_path):
    """
    Converts a .dta file to a .parquet file with the required columns.
    """
    try:
        df = pd.read_stata(dta_path)
        df_to_process = df[['university', 'firstname', 'lastname']]
        df_to_process.to_parquet(parquet_path)
        print(f"  Converted {dta_path} to {parquet_path}")
    except Exception as e:
        print(f"  Error converting {dta_path} to parquet: {e}")
        raise

def run_rmp_rust_binary(parquet_path):
    """
    Compile the Rust crate if needed, then run the batch binary on the given parquet file.
    Streams output directly.
    """
    if not os.path.exists(RMP_RUST_BINARY):
        print("  Rust binary not found, compiling...")
        subprocess.run(
            ["cargo", "build", "--release", "--manifest-path", os.path.join(SCRIPT_DIR, "Cargo.toml")],
            check=True,
        )
    print(f"  Running RMP Rust binary on {parquet_path}...")
    subprocess.run([RMP_RUST_BINARY, parquet_path], check=True)

def main():
    """
    Main function to orchestrate the RMP data pipeline.
    """
    print("Starting RMP data pipeline...")

    # Authenticate with Dropbox
    dbx = dropbox_oauth()

    # Discover .dta files in local storage directory
    if not os.path.isdir(RUST_STORAGE_DIR):
        print(f"Storage directory not found: {RUST_STORAGE_DIR}")
        return

    dta_files = [f for f in os.listdir(RUST_STORAGE_DIR) if f.lower().endswith('.dta')]
    if not dta_files:
        print("No .dta files found in local storage directory.")
        return

    print(f"Found {len(dta_files)} .dta files to process:")


    for name in dta_files:
        try:
            print(f"Processing {name}...")
            base, _ = os.path.splitext(name)

            local_dta_path = os.path.join(RUST_STORAGE_DIR, name)
            local_parquet_path = os.path.join(RUST_STORAGE_DIR, f"{base}.parquet")
            output_dta_path = os.path.join(RUST_STORAGE_DIR, f"export/{base}_v2.dta")

            # 1) Convert to parquet
            convert_dta_to_parquet(local_dta_path, local_parquet_path)

            # 2) Run Rust binary
            run_rmp_rust_binary(local_parquet_path)

            # 3) Merge results and save updated .dta
            add_rmp_column_to_stata_file(local_dta_path, local_parquet_path, output_dta_path)

            # 4) Upload to Dropbox data-files folder with v2 suffix
            dropbox_file_path = f"{DROPBOX_FOLDER}/data-files/{base}_v2.dta"
            print(f"  Uploading to Dropbox: {dropbox_file_path}")
            upload_large_file(dbx, output_dta_path, dropbox_file_path)
            print(f"Finished {name}.")

        except Exception as e:
            print(f"Error processing {name}: {e}")

if __name__ == "__main__":
    main()