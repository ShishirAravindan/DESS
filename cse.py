import os
import re
import pandas as pd
from dotenv import load_dotenv
import requests
import csv

load_dotenv()

BASE_URL = "https://www.googleapis.com/customsearch/v1"
DATASET_DIR = "storage/dataset"

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
    """
    Reads a CSV file, uses a separate function to clean strings, combines title and snippet for each row,
    and returns the first 4 rows as a list.  Handles potential errors gracefully.
    """
    try:
        df = pd.read_csv(filePath, on_bad_lines='warn', encoding='utf-8')

        if 'title' not in df.columns or 'snippet' not in df.columns:
            raise ValueError(f"'title' or 'snippet' column missing in {filePath}")

        # Use the separate cleaning function
        df['title'] = df['title'].apply(_clean_strings)
        df['snippet'] = df['snippet'].apply(_clean_strings)

        df['combined'] = df['title'].fillna('') + ' ' + df['snippet'].fillna('')
        rawText = df['combined'].tolist()[:4]
        return rawText if len(rawText) > 0 else None

    except Exception as e:
        print(f"An unexpected error occurred while processing {filePath}: {e}")
        raise e

def populate_rawText_col(df):
    """
    Populates a new 'rawText' column in the input DataFrame.  If a Google Custom Search API call fails, 
    the corresponding 'rawText' entry is set to "NaN".

    Args:
        df (pd.DataFrame): DataFrame containing an 'id_text' column.

    Returns:
        pd.DataFrame: DataFrame with an added 'rawText' column.
    """
    df['rawText'] = None  # Initialize the column

    for index, row in df.iterrows():
        id_text = row['id_text']
        try:
            file_name = make_API_CALL(id_text)
            raw_text = _get_rawText(file_name)
            df.at[index, 'rawText'] = raw_text if raw_text is not None else ["NaN"] 
        except Exception as e:
            print(f"Error processing {id_text}: {e}")
            df.at[index, 'rawText'] = ["NaN"]

    return df
        
if __name__=='__main__':
    search_query = 'Ngoyi Bukonda northern illinois university'
    make_API_CALL(search_query, "y15")
        