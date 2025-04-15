from dess.llms.gemini_llm import GeminiLLM


class LLMFactory:
    """
    This class creates and returns the right LLM object.
    """
    @staticmethod
    def get_llm(llm_type: str):
        if llm_type == "gemini":
            return GeminiLLM(llm_type)
        else:
            raise ValueError(f"Invalid LLM type: {llm_type}")