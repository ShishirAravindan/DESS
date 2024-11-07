"""
Provides functionality to create web drivers for Chrome and Firefox using Selenium, perform Google 
searches, and parse the search results.

Dependencies:
- Selenium: A browser automation framework. Ensure that the appropriate browser driver (ChromeDriver or GeckoDriver) 
is installed and accessible in your PATH.
"""

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.firefox.options import Options as FireFoxOptions
import time
import os
import argparse

LOCAL_PARQUET_PATH = 'storage/uncomplete.parquet'
CHUNK_SIZE = 200

def setup_driver(driver_type: str) -> webdriver:
    """Sets up the web driver based on the specified type."""
    if driver_type == 'chrome':
        return _create_chrome_driver()
    elif driver_type == 'firefox':
        return _create_firefox_driver()
    else:
        raise ValueError("Invalid driver type specified. Use 'chrome' or 'firefox'.")

def _create_chrome_driver():
    """
    Initializes and returns a Chrome web driver with specified options.

    Returns:
        webdriver.Chrome: An instance of the Chrome web driver configured with headless options.
    """
    options = Options()
    options.add_argument("--headless")
    #options.add_experimental_option("detach", True)
    service = Service('/opt/homebrew/bin/chromedriver')
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def _create_firefox_driver():
    """
    Initializes and returns a Firefox web driver with specified options.

    Returns:
        webdriver.Firefox: An instance of the Chrome web driver configured with headless options.
    """
    # Set up Selenium options
    options = FireFoxOptions()
    options.set_preference("dom.popup_maximum", 0)
    options.set_preference("privacy.popups.disable_from_plugins", 3)
    options.add_argument("--headless")
    driver = webdriver.Firefox(options=options)
    return driver

def get_snapshots_from_google(driver: webdriver, search_query:str, snapshots:int):
    """
    Performs a Google search for the given name and university, and retrieves specified snapshots of the search results.

    Args:
        driver (webdriver): The web driver instance to use for the search.
        search_query (str): The name and university to search for (in Google).
        snapshots (int): The number of result snapshots to retrieve.

    Returns:
        list: A list containing the parsed text of the search result snapshots.

    Raises:
        Exception: Raises an exception if there is an error during the search or result retrieval.
    """
    google_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
    driver.get(google_url)

    time.sleep(2)  

    # Find the first snapshot search results
    try:
        #This will contain the raw text from the search results
        feature_vector = []
        results = driver.find_elements(By.XPATH, '//div[@class="sATSHe"]')
        if len(results)<snapshots:
            #Use a different xpath
            results = driver.find_elements(By.XPATH, '//div[@class="MjjYud"]')
        #print(f'Length of the div list is = {len(results)}')
        no_of_wanted_results = snapshots
        completed = 0
        index = 0
        while completed != no_of_wanted_results:
            div_element = results[index]
            try:
                
                div_text = div_element.text
                span_text = div_element.find_element(By.XPATH, './/span').text
                if span_text == 'People also ask':
                    index += 1
                    continue
                elif 'People also ask' in div_text:
                    index += 1
                    continue
                feature_vector.append(parse_text(div_text))
                
                completed += 1
        
            except:
                index += 1
                continue
        
            index += 1
        return feature_vector

    except Exception as e:
        print(f'Failed for prof: {search_query}')
        print(f'Error while scraping', e)
        # driver.quit()

def parse_text(text: str):
    text = text.split('\n')
    return text[-1].strip()

def populate_raw_text(df: pd.DataFrame, driver, snapshots: int) -> pd.Series:
    """Populates the rawText column in the DataFrame."""
    count = 0
    def fetch_raw_text(row):
        nonlocal count
        count += 1
        print(f"\t row #{count}")
       
        return get_snapshots_from_google(driver, row['id_text'], snapshots)
    return df.apply(fetch_raw_text, axis=1)

def search(df: pd.DataFrame, driver_type: str, snapshots: int):
    """
    Populates the DataFrame's rawText columns by scraping search results from Google.

    Args:
        df (pd.DataFrame): DataFrame containing the name and university columns.
        driver_type (str): Type of web driver to use ('chrome' or 'firefox').
        snapshots (int): The number of search results to retrieve for rawText.

    Returns:
        pd.DataFrame: Updated DataFrame with additional columns populated.
    """
    driver = setup_driver(driver_type)
    df['rawText'] = populate_raw_text(df, driver, snapshots)
    driver.quit()

def main(start_index: int):
    if os.path.exists(LOCAL_PARQUET_PATH):
        print("Loading DataFrame from local ...")
        df = pd.read_parquet(LOCAL_PARQUET_PATH)
    else:
        print("FILE NOT FOUND")
        return
    
    # Start the processing-and-caching process
    print(f"PROCESSING: Started from index {start_index}")
    start = time.time()
    for i in range(start_index, len(df), CHUNK_SIZE):
        chunk = df.iloc[i:i + CHUNK_SIZE].copy()
        chunk = search(chunk, 'firefox', 4)
        df.iloc[i:i + CHUNK_SIZE, df.columns.get_loc('rawText')] = chunk['rawText']
        df.to_parquet(LOCAL_PARQUET_PATH, index=False)
        current_time = time.time()
        time_taken = current_time - start
        start = current_time
        print(f"[{time_taken:.2f}] Processed and updated chunk {i // CHUNK_SIZE} of {(df.shape[0]) // CHUNK_SIZE}")


def test_main():
    test_df = pd.DataFrame({
    'rawText': ['', ''],
    'id_text': ['Arnold Rosenbloom University of Toronto', 'Andrew Peterson University of Toronto']
    })
    test_df = populate_raw_text(test_df, _create_firefox_driver(), 4)
    print(test_df)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pass a start index to the script.")
    parser.add_argument("start_index", type=int, help="The index to start processing from")
    args = parser.parse_args()
    main(args.start_index)