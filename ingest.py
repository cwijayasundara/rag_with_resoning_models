import nest_asyncio
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI

nest_asyncio.apply()

_ = load_dotenv()

Settings.embed_model = OpenAIEmbedding(
    model_name="text-embedding-3-large"
)

def ingest_pdf(file_path: str, persist_dir: str):

    print("pushing the document to the vector index")
    documents = LlamaParse(result_type="markdown").load_data(file_path)
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=persist_dir)
    index = VectorStoreIndex.from_documents(documents)
    return "success"