from abc import ABC

class LLMBase(ABC):
    """
    Abstract base class for all LLM models. The model creation will be in
    contructor of the derived classes. The common code to get the response
    from the model is in this class.
    """
    def isOk(self) -> bool:
        return self.model_name is not None and self.llm is not None
    
    def get_response(self, prompt: str) -> str:
        return self.llm.get_response(prompt)
