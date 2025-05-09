import pandas as pd
import os
import json
import uuid
import time
from typing import Dict, Any, Optional
from google.cloud import storage
from google import genai
from google.genai import types
from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions
from dotenv import load_dotenv
from .base import BatchInferencePipeline

class GeminiBatchInferencePipeline(BatchInferencePipeline):
    """
    Batch inference pipeline implementation for Google's Gemini models.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the pipeline with model name.
        
        Args:
            model_name (str): Gemini model name to use
        """
        load_dotenv()
        self.model_name = model_name
            
        # Initialize the client with Vertex AI settings
        self.client = genai.Client(http_options=types.HttpOptions(api_version='v1'))
        
    def prepare_batch_file(self, df: pd.DataFrame, output_file: str) -> Dict[str, Any]:
        """
        Creates JSONL batch file in format required by Vertex AI batch predictions.
        
        Args:
            df (pd.DataFrame): DataFrame containing faculty data
            output_file (str): Path to save the JSONL file
            
        Returns:
            Dict[str, Any]: Information about the batch file including mapping between rows and prompts
        """
        # Validate required columns
        required_cols = ['id_text', 'snippet_1', 'snippet_2', 'snippet_3', 'snippet_4']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create a new column for the combined text
        df['rawText'] = df.apply(lambda row: " ".join([row[f'snippet_{i}'] for i in range(1, 5) if pd.notna(row[f'snippet_{i}'])]), axis=1)
        df = df[df['rawText'].notna() & df['rawText'].str.strip()]
        
        # Prepare batch ID and mapping
        batch_id = str(uuid.uuid4())
        row_mapping = []
        
        # Open file for writing
        with open(output_file, 'w') as f:
            for idx, row in df.iterrows():
                # Store mapping information
                row_mapping.append({
                    'idx': int(idx),
                    'id_text': row['id_text']
                })

                # Format according to Vertex AI batch prediction requirements
                instance = {
                    "request": {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [{
                                    "text": f"""Given the following text about a professor or faculty member, extract their department name.
                                    If no department is mentioned, return "MISSING". Only return the department name, nothing else.
                                    
                                    Text: {row['rawText']}
                                    Department:"""
                                }]
                            }
                        ]
                    }
                }
                f.write(json.dumps(instance) + "\n")
        
        # Return batch information
        return {
            "batch_id": batch_id,
            "file_path": output_file,
            "model_name": self.model_name,
            "row_mapping": row_mapping,
            "row_count": len(row_mapping)
        }
        
    def upload_batch_file(self, file_path: str, bucket_name: str, destination_blob_name: Optional[str] = None) -> str:
        """
        Upload a file to Google Cloud Storage.
        
        Args:
            file_path (str): Path to the local file
            bucket_name (str): Name of the GCS bucket
            destination_blob_name (str, optional): Name of the destination blob
            
        Returns:
            str: GCS URI of the uploaded file (gs://bucket/path)
        """
        if destination_blob_name is None:
            destination_blob_name = os.path.basename(file_path)
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        
        blob.upload_from_filename(file_path)
        
        return f"gs://{bucket_name}/{destination_blob_name}"
        
    def create_batch_job(self, source_uri: str, output_uri: str) -> Any:
        """
        Creates batch job on Vertex AI.
        
        Args:
            source_uri (str): URI of the uploaded batch file
            output_uri (str): URI for the output results
            
        Returns:
            Any: Job reference object
        """
        job = self.client.batches.create(
            model=self.model_name,
            src=source_uri,
            config=CreateBatchJobConfig(dest=output_uri),
        )
        
        print(f"Job {job.name} | State: {job.state}")
        
        return job
        
    def get_batch_status(self, job_reference: Any) -> str:
        """
        Gets current status of batch job.
        
        Args:
            job_reference (Any): Job reference object
            
        Returns:
            str: Status of the batch job
        """
        job = self.client.batches.get(name=job_reference.name)
        print(f"Job state: {job.state}")
        return job.state.name
        
    def wait_for_completion(self, job_reference: Any, poll_interval: int = 30) -> Any:
        """
        Waits for batch job to complete.
        
        Args:
            job_reference (Any): Job reference object
            poll_interval (int): Time in seconds between status checks
            
        Returns:
            Any: Updated job reference object
        """
        completed_states = {
            JobState.JOB_STATE_SUCCEEDED,
            JobState.JOB_STATE_FAILED,
            JobState.JOB_STATE_CANCELLED,
            JobState.JOB_STATE_PAUSED,
        }
        
        job = job_reference
        while job.state not in completed_states:
            time.sleep(poll_interval)
            job = self.client.batches.get(name=job.name)
            print(f"Job state: {job.state}")
            
        return job
        
    def retrieve_and_merge_results(self, job_reference: Any, df: pd.DataFrame, mapping: Dict[str, Any]) -> pd.DataFrame:
        """
        Retrieves results from GCS and merges back into original dataframe.
        
        Args:
            job_reference (Any): Job reference object
            df (pd.DataFrame): Original DataFrame
            mapping (Dict[str, Any]): Mapping information from prepare_batch_file
            
        Returns:
            pd.DataFrame: Updated DataFrame with department_llm column populated
        """
        # TODO: Implement results retrieval from GCS
        # This would involve:
        # 1. Getting the output path from job_reference
        # 2. Downloading results from GCS
        # 3. Parsing results and mapping them back to the original dataframe
        
        # For now, this is a placeholder
        df['department_llm'] = None  # Initialize column
        
        # Use mapping to update the department_llm column
        row_mapping = mapping.get("row_mapping", [])
        
        # Return updated dataframe
        return df

def test_main():
    pipeline = GeminiBatchInferencePipeline(model_name="gemini-2.0-flash-001")
    df_test  = pd.read_parquet('/Users/shishiraravindan/Documents/work-RA/dess/storage/test_batch.parquet')
    
    # Create batch file
    batch_info = pipeline.prepare_batch_file(df_test, 'batch_requests.jsonl')
    print("Batch file created")
    
    # Upload to GCS
    source_uri = pipeline.upload_batch_file(
       batch_info['file_path'],
       'dess-llm-jobs',
       'batch_requests.jsonl'
   )
    print("Batch file uploaded to GCS")
   
    # Create and run batch job
    output_uri = 'gs://dess-llm-jobs/output/'
    job = pipeline.create_batch_job(source_uri, output_uri)
    print("Batch job created")
   
    # Wait for completion
    completed_job = pipeline.wait_for_completion(job)
    print("Batch job completed")

    # alt: open result file to unpack
    
    
if __name__=='__main__':
    test_main()