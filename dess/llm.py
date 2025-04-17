import pandas as pd
from llms.llm_factory import LLMFactory
import argparse
def infer_departments_with_llm(
    texts: pd.Series, 
    llm_type: str = "gemini", 
    batch_size: int = 10
) -> pd.Series:
    """
    Infer departments from a Series of raw text using the specified LLM.

    Args:
        texts (pd.Series): A pandas Series containing text to infer departments from.
        llm_type (str): The LLM type to use ("gemini" or "gpt").
        batch_size (int): Number of samples to process per batch.

    Returns:
        pd.Series: A Series of inferred department names, indexed to match `texts`.
    """
    llm = LLMFactory.get_llm(llm_type)
    
    if not llm.isOk():
        raise ValueError(f"Failed to initialize {llm_type} LLM")

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts.iloc[i:i + batch_size].tolist()
        try:
            batch_results = llm.infer_departments_batch(batch)
        except Exception as e:
            print(f" Error in batch {i // batch_size + 1}: {e}")
            batch_results = ["ERROR"] * len(batch)
        results.extend(batch_results)

    return pd.Series(results, index=texts.index)

def run_inference(df: pd.DataFrame, llm_type: str, infer_departments:bool=True) -> pd.DataFrame:
    """
    Extract 'rawText' from a DataFrame, infer departments using LLM, and add 'LLMDept' column.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'rawText' column.
        llm_type (str): The type of LLM to use ("gemini" or "gpt").

    Returns:
        pd.Series: The inferred departments as a Series, which is added to the DataFrame as 'LLMDept'.
    """
    if "rawText" not in df.columns:
        raise ValueError("The DataFrame must contain a 'rawText' column.")

    # Extract the 'rawText' column
    raw_texts = df["rawText"]
    if infer_departments:
        # Run department inference in batches
        results = infer_departments_with_llm(raw_texts, llm_type)
        df["LLMDept"] = results
        print(df[["LLMDept"]])
    else:
        #run one by one
        llm = LLMFactory.get_llm(llm_type)
        if not llm.isOk():
            raise ValueError(f"Failed to initialize {llm_type} LLM")
        # Run get_response for each row 
        results=[]
        for i, (idx, text) in enumerate(raw_texts.items()):
            print(f"\nRow {i + 1}")
            print(f"Input: {text}")
            try:
                response = llm.get_response(text)
                print(response)
            except Exception as e:
                response = "ERROR"
                print(f"Error: {e}")
            results.append((idx, response))
            print(f"Response: {response}")
        results_series = pd.Series(dict(results)).reindex(raw_texts.index)
        df["LLMDept"] = results_series
        
    # Return the dataframe containing new column LLMDept
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using an LLM on a Stata (.dta) file")
    parser.add_argument("--llm_type", required=True, help="The type of LLM to use")
    parser.add_argument("--infer_departments", action="store_true", help="If set, infers departments instead of simple prompt completion")
    parser.add_argument("--input_dta_file", required=True, help="Path to the Stata (.dta) file")
    args = parser.parse_args()
    df = pd.read_parquet(args.input_dta_file)
    run_inference(df, 
                  llm_type=args.llm_type, 
                  infer_departments=args.infer_departments
                  )
