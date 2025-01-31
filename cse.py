import os
import pandas as pd
from dotenv import load_dotenv
import requests

load_dotenv()

BASE_URL = "https://www.googleapis.com/customsearch/v1"

def _build_payload(search_query):
    """TODO: Document + add other relevant params"""
    return {
        'key': os.getenv('CSE_API_KEY'), 
        'cx': os.getenv('SEARCH_ENGINE_ID'),
        'q': search_query
    }

def make_API_CALL(search_query):
    response = requests.get(BASE_URL, params=_build_payload(search_query))
    results = response.json()

    # Save results to a CSV file
    df_results = pd.json_normalize(results.get('items', []))  # Normalize the JSON response
    file_name = '_'.join(search_query)
    df_results.to_csv(f"storage/{file_name}.csv", index=False)
    return f"storage/{search_query}.csv"

def _get_rawText(filePath):
    df = pd.read_csv(filePath)
    rawText = (df['title'] + ' ' + df['snippet']).tolist()[:4]
    return rawText

def populate_rawText_col(df):
    for i, id_text in df['id_text'].items():
        file_name = make_API_CALL(id_text)
        r = _get_rawText(file_name)
        # df.at[i, 'rawText'] = 
        # TODO:
        # unpack snippets into their own cols
        # add col for where we get keyword from
        