from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings,HuggingFaceBgeEmbeddings
     
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
hf_bge_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en")

# Now ingest the Data into the Chroma Database
from langchain.vectorstores import Chroma
import chromadb
import os

os.getcwd()
# '/content'

CURRENT_DIR = os.path.dirname(os.path.abspath("."))
CURRENT_DIR    
# '/'


DB_DIR = os.path.join(CURRENT_DIR, "/content/db")
DB_DIR    
# '/content/db'     


client_settings = chromadb.config.Settings(
    is_persistent=True,
    persist_directory=DB_DIR,
    anonymized_telemetry=False,
)
     

harrypotter_vectorstore = Chroma.from_documents(text_harrypotter,
                                       hf_bge_embeddings,
                                       client_settings=client_settings,
                                       collection_name="harrypotter",
                                       collection_metadata={"hnsw":"cosine"},
                                       persist_directory="/store/harrypotter")
     

got_vectorstore = Chroma.from_documents(text_got,
                                       hf_bge_embeddings,
                                       client_settings=client_settings,
                                       collection_name="got",
                                       collection_metadata={"hnsw":"cosine"},
                                       persist_directory="/store/got")
     
