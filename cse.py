import os
import pandas as pd
from dotenv import load_dotenv
import requests
import ast
load_dotenv()

BASE_URL = "https://www.googleapis.com/customsearch/v1"

def _build_payload(search_query,date_restrict):
    """TODO: Document + add other relevant params"""
    return {
        'key': os.getenv('CSE_API_KEY'), 
        'cx': os.getenv('SEARCH_ENGINE_ID'),
        'q': search_query,
        'num':5,
        # 'dateRestrict':date_restrict,
        # 'sort': "date:r:20100101:20101231"
    }

def make_API_CALL(search_query,date_restrict):
    response = requests.get(BASE_URL, params=_build_payload(search_query,date_restrict))
    results = response.json()
    # Save results to a CSV file
    df_results = pd.json_normalize(results.get('items', []))
    # print(df_results)  # Normalize the JSON response
    #use 
    file_name = '_'.join(search_query.split(" "))
    df_results.to_csv(f"storage/csv_datasets/{file_name}.csv", index=False)
    return f"storage/csv_datasets/{file_name}.csv"
def replace_escaped_quotes(input_string): 
    # Replace escaped single quotes and double quotes 
    # Replaces \' with a backtick 
    input_string = input_string.replace(r"\xa0", " ") 
    return input_string
    # Replaces \" with a single quote return input_string
def _get_rawText(filePath):
    try:
        df = pd.read_csv(filePath)
        if 'title' and 'snippet' in df.columns:
            df['title']=df['title'].apply(replace_escaped_quotes)
            df['snippet']=df['snippet'].apply(replace_escaped_quotes)
            rawText = (df['title'] + ' ' + df['snippet']).tolist()[:4]
        else:
            rawText = None
        return rawText
    except:
        return None

def populate_rawText_col(df):
    id_text_list = df['id_text'].tolist()
    raw_texts = []
    for i in range(len(df)):
        id_text = id_text_list[i]
        # print(id_text)
        file_name = make_API_CALL(id_text,"y15")
        # print(file_name)
        raw_text = _get_rawText(file_name)
        if raw_text is None:
            print("Error while generating csv from google search for "+id_text)
        raw_texts.append(raw_text)
        # os.remove(file_name)
    df['rawText']=raw_texts
    # df['rawText']=df['rawText'].apply(ast.literal_eval)
    return df
        
        

if __name__=='__main__':
    # search_query = 'David Osullivan university of california-berkeley'
    # make_API_CALL("Frederick Dolan university of california-berkeley","y19")
    search_query = 'Ngoyi Bukonda northern illinois university'
    make_API_CALL(search_query,"y15")
    #print(_get_rawText("storage/David_Osullivan_university_of_california-berkeley.csv"))
        