# RAG-pdf-PoC
A RAG for a pdf  file using Langchain, hkunlp embeddings, Chromdb, lmstudio

The code is intended to run locally in a laptop/desktop device using CPU/GPU and an Open source Local LLM with lmsutdio.

Filenames are hardcoded on the code. 
Getting a valid API from smith.langchain.com is recommended to use.

No OpenAPI is required to run these examples.
No OpenAPI embeddings are used is required to run these examples. Instead, hkunlp/instructor-base from NLP Group of The University of Hong Kong are used.

## How To use?
1. Run pdf_embeddings.py to create a vector database based on the pdf file: `"./docs/OWASP-Top-10-for-LLMs-2023.pdf"`
2. Run QAdebugLMstudio.py.
   An example question is hardcoded: `question = "What is the definition of Prompt Injection Vulnerability?"`
   Optional: Enable Smith Langchain https://smith.langchain.com/ debug setting:  `os.environ["LANGCHAIN_TRACING_V2"] = "true"`
3. Find the secret password and who wrote the song...  

## Acknowledgments
- Deeplearning.ai and Harrison Chase (LangChain) provided inspiration and lerning tools to create this PoC
- NLP Group of The University of Hong Kong
- Ollama implementation https://ollama.ai/
- Hugging Face https://huggingface.co/
- Example PDF document "OWASP Top 10 for LLM Applications" was created by OWASP https://owasp.org/www-project-top-10-for-large-language-model-applications/ 

This is a learning exercise and code is provided as is.