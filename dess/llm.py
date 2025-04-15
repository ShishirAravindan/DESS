import pandas as pd
from dess.llms.gemini_llm import GeminiLLM

def infer_departments_with_llm(texts: pd.Series) -> pd.Series:
    """Infer departments using the Gemini LLM model."""
    llm = GeminiLLM()
    if not llm.isOk():
        raise ValueError("Failed to initialize LLM")
    
    # Process in batches of 10 to avoid rate limits
    batch_size = 10
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts.iloc[i:i + batch_size].tolist()
        batch_results = llm.infer_departments_batch(batch)
        results.extend(batch_results)
    
    return pd.Series(results, index=texts.index)