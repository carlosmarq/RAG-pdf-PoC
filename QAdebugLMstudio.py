#Run pdf_embeddings.py to embed a PDF dopcument into the Chroma database before running this script
#Start LM studio listening at "http://localhost:1234/v1"

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings

persist_directory = 'docs/chroma/'
embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base", model_kwargs={"device": "cpu"})
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

print("Number of vectors in the stored database:")
print(vectordb._collection.count())

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed", temperature=0.0, verbose=True)

from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever() )

#### prompt
from langchain.prompts import PromptTemplate

# Build prompt
template = """Use the only following pieces of context to answer the question at the end. If you don't know the answer, 
just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

question = "What is the definition of Prompt Injection Vulnerability?"
result = qa_chain({"query": question})
result["result"]
result["source_documents"][0]


import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"	# change if debug is required
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = "XXXXXXXXXXXXXX" # replace with your api key

qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce"
)
result = qa_chain_mr({"query": question})
result["result"]

#Print results:
print(result["query"])
print(result["result"])
