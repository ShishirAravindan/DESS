# LLM Batch Inference Experiment

> **Status: Archived** - This approach was explored as an alternative to pattern-based department extraction but was not adopted for the final pipeline.

## Overview

This experiment explored using Large Language Models (LLMs) to extract department information from faculty search result snippets. The idea was to leverage LLMs' natural language understanding to improve extraction accuracy beyond what regex patterns could achieve.

## Contents

### Batch Inference Pipeline (`batch_inference/`)

A structured pipeline for processing large batches through LLM inference:

- **Architecture**: Factory pattern supporting multiple providers (Gemini, OpenAI)
- **Components**:
  - `base.py` - Abstract base class defining the pipeline interface
  - `factory.py` - Factory for creating provider-specific pipelines
  - `gemini.py` - Google Gemini/Vertex AI implementation
  - `openai.py` - OpenAI implementation (incomplete)
  - `prompts/` - Prompt templates for department extraction

See `batch_inference/README.md` for detailed API documentation.

### Simple LLM Wrappers (`llms/`)

Earlier iteration with simpler LLM wrapper classes:

- `llm_base.py` - Abstract interface
- `llm_factory.py` - Factory for LLM instantiation
- `gemini_llm.py` - Gemini wrapper
- `gpt_llm.py` - GPT wrapper (uses deprecated OpenAI API)
- `openaibatchprocessor.py` - Standalone OpenAI batch processor

### Entry Points

- `llm.py` - Uses the simple `llms/` wrappers
- `extract_departments_batch.py` - Uses the batch inference pipeline
- `createopenaibatch.py` - Standalone OpenAI batch creation script

## Why It Wasn't Adopted

While LLMs showed promise for understanding context in faculty descriptions, the pattern-based approach in `dess/nlp.py` proved sufficient for the dataset's needs. The main pipeline (Google Custom Search API + pattern matching) achieved acceptable accuracy without the additional complexity, reputational risk and cost of LLM inference.

## If You Want to Explore Further

The batch inference pipeline (`batch_inference/`) is the more complete implementation. To use it:

```bash
# Requires GOOGLE_API_KEY environment variable for Gemini
python extract_departments_batch.py --input_file=data.parquet --provider=gemini-2.0-flash-001
```

Note: The OpenAI implementation is incomplete and the `llms/` wrappers use deprecated APIs.
