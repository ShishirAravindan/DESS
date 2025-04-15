from dess.llms.llm_base import LLMBase
import google.generativeai as genai
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

class GeminiLLM(LLMBase):
    """
    Gemini LLM model.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel(model_name)
        
    def get_response(self, prompt: str) -> str:
        """Get response for a single prompt."""
        response = self.llm.generate_content(prompt)
        return response.text
    
    def get_batch_responses(self, prompts: List[str]) -> List[str]:
        """Get responses for multiple prompts in batch."""
        responses = self.llm.generate_content(prompts)
        return [response.text for response in responses]
    
    def infer_department(self, text: str) -> str:
        """Infer department from text using a specific prompt."""
        prompt = f"""Given the following text about a professor or faculty member, extract their department name.
        If no department is mentioned, return "MISSING". Only return the department name, nothing else.
        
        Text: {text}
        Department:"""
        
        return self.get_response(prompt)
    
    def infer_departments_batch(self, texts: List[str]) -> List[str]:
        """Infer departments from multiple texts in batch."""
        prompts = [
            f"""Given the following text about a professor or faculty member, extract their department name.
            If no department is mentioned, return "MISSING". Only return the department name, nothing else.
            
            Text: {text}
            Department:"""
            for text in texts
        ]
        
        return self.get_batch_responses(prompts)

        