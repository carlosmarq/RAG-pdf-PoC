#!pip install sentence_transformers InstructorEmbedding
#https://huggingface.co/hkunlp/instructor-xl
#https://huggingface.co/hkunlp/instructor-base
#https://huggingface.co/hkunlp

from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings

embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base", model_kwargs={"device": "cpu"})

from langchain.document_loaders import PyPDFLoader

# Load PDF
loaders = [
    # Modify the line bellow to indicate the data source PDF file:
    PyPDFLoader("./docs/OWASP-Top-10-for-LLMs-2023.pdf"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

    # Split
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)

splits = text_splitter.split_documents(docs)
print("The document will be divided on:")
print(len(splits))

from langchain.vectorstores import Chroma
persist_directory = 'docs/chroma/'
#!rm -rf ./docs/chroma  # remove old database files if any

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)
print("The number of vectors in the database:")
print(vectordb._collection.count())

vectordb.persist()

#semantic similarity
'''
question = "who are the authors of the document"
docs = vectordb.similarity_search(question,k=3)
len(docs)
docs[0].page_content
'''