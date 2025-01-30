import nest_asyncio
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)
from pathlib import Path

nest_asyncio.apply()

_ = load_dotenv()

Settings.embed_model = OpenAIEmbedding(
    model_name="text-embedding-3-large"
)

PERSIST_DIR = "./vector_db"

def get_query_engine():
    """Initialize and return the query engine"""
    if not Path(PERSIST_DIR).exists():
        raise ValueError("No index found. Please ingest documents first.")
        
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    return index.as_query_engine(similarity_top_k=3)

def query_kb(query: str) -> str:
    """Query the knowledge base for the answer to the question."""
    query_engine = get_query_engine()
    response = query_engine.query(query)
    return str(response)