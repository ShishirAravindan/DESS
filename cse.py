import os
import re
import pandas as pd
from dotenv import load_dotenv
import requests
import csv
import logging

load_dotenv()

logger = logging.getLogger(__name__)
BASE_URL = "https://www.googleapis.com/customsearch/v1"
DATASET_DIR = f"{os.getenv('STORAGE_DIR')}/dataset"

def _build_payload(search_query, date_restrict):
    """Constructs the payload for the Google Custom Search API request."""
    return {
        'key': os.getenv('CSE_API_KEY'), 
        'cx': os.getenv('SEARCH_ENGINE_ID'),
        'q': search_query
        # 'dateRestrict':date_restrict,
        # 'sort': "date:r:20100101:20101231"
    }

def make_API_CALL(search_query, date_restrict=None):
    response = requests.get(BASE_URL, params=_build_payload(search_query, date_restrict))
    
    if response.status_code == 200:  # Save CSV only if response is successful
        results = response.json()
        df_results = pd.json_normalize(results.get('items', []))
        file_name = '_'.join(search_query.split(" "))
        file_name = file_name.replace('/', '_').replace('\\', '_')
        df_results.to_csv(f"{DATASET_DIR}/{file_name}.csv", index=False, quoting=csv.QUOTE_ALL)
        
        return f"{DATASET_DIR}/{file_name}.csv"
    else:
        raise Exception(f"API call failed with status code: {response.status_code}")

def _clean_strings(text):
    """Cleans a string by escaping unescaped double quotes and handling potential invalid characters."""
    text = str(text)
    text = re.sub(r'(?<!\\)"', '""', text)
    text = re.sub(r"(?<!\\)'", "'", text)
    text = text.encode('unicode_escape').decode('utf-8') # Handle potential invalid characters
    return text

def _get_rawText(filePath):
    """Reads a CSV file and returns list of combined title/snippet text or None if any failure occurs."""
    try:
        df = pd.read_csv(filePath, on_bad_lines='warn', encoding='utf-8')
        
        # Check if DataFrame is empty
        if df.empty:
            logger.warning(f"CSV file {filePath} contains no data")
            return None
        
        if 'title' not in df.columns or 'snippet' not in df.columns:
            logger.warning(f"Missing required columns in {filePath}")
            return None
            
        df['title'] = df['title'].apply(_clean_strings)
        df['snippet'] = df['snippet'].apply(_clean_strings)
        df['combined'] = df['title'].fillna('') + ' ' + df['snippet'].fillna('')
        
        rawText = df['combined'].tolist()[:4]
        return rawText if rawText else None
    except Exception as e:
        logger.error(f"Failed to process file {filePath}: {e}")
        return None

def populate_rawText_col(df):
    """
    Populates the rawText column in the DataFrame. Any failure (API error, empty results, 
    file processing error) will result in None values that can be filtered with isna().

    Args:
        df (pd.DataFrame): DataFrame containing an 'id_text' column.

    Returns:
        pd.DataFrame: DataFrame with populated 'rawText' column, None for any failures.
    """
    df['rawText'] = None

    # Log only at the beginning of processing
    logger.info(f"Starting to process {len(df)} rows for API calls")

    for index, row in df.iterrows():
        id_text = row['id_text']
        try:
            file_name = make_API_CALL(id_text)
            raw_text = _get_rawText(file_name)
            df.at[index, 'rawText'] = raw_text
        except Exception as e:
            logger.error(f"Error processing row {index} ({id_text}): {e}")

    return df
        
if __name__=='__main__':
    search_query = 'Ngoyi Bukonda northern illinois university'
    make_API_CALL(search_query, "y15")
    test_df = pd.DataFrame({
        'id_text': ['Donald Kluemperjr louisiana state university and agricultural & mechanical college']
    })
    df2 = populate_rawText_col(test_df)
    df_errors = df2[df2['rawText'].isna()]
    df_errors[['id_text']].to_csv('errors.csv', mode='a', index=False)
        