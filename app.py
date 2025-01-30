import streamlit as st
import nest_asyncio
from pathlib import Path
import os
from ingest import ingest_pdf
from retriever import query_kb
from llama_index.llms.openai import OpenAI
from llama_index.llms.deepseek import DeepSeek
from llama_index.llms.groq import Groq
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings

nest_asyncio.apply()

def sanitize_filename(filename):
    """
    Sanitize the filename to prevent path traversal attacks and remove unwanted characters.
    """
    filename = os.path.basename(filename)
    filename = "".join(c for c in filename if c.isalnum() or c in (" ", ".", "_", "-")).rstrip()
    return filename

def save_uploaded_file(uploaded_file, file_path):
    """Save the uploaded file and return the save path"""
    upload_dir = Path.cwd()/"uploaded_docs" / file_path
    upload_dir.mkdir(parents=True, exist_ok=True)
    original_filename = Path(uploaded_file.name).name
    sanitized_filename = sanitize_filename(original_filename)
    save_path = str(upload_dir / sanitized_filename)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

PERSIST_DIR = "./vector_db"

st.title("Welcome to RAG app")
st.write("RAG system to manage your documents!")

api_key = os.getenv("DEEPSEEK_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
                      
with st.sidebar:
    st.image("images/plus.png", width=600)

    st.write("Uselect the LLM you want to use")
    llm_choice = st.radio(
        "**Select LLM**",
        options=(
            "o1",
            "o1-mini",
            "gemini-2.0-flash-thinking-exp-01-21",
            "deepseek-reasoner",
            "deepseek-reasoner-groq",
            "deepseek-chat"
        )
    )

    st.write(f"You selected {llm_choice}")
    if llm_choice == "o1":
        llm = OpenAI(model="gpt-4o", temperature=1.0)                   
    elif llm_choice == "o1-mini":
        llm = OpenAI(model="gpt-4o-mini", temperature=1.0)
    elif llm_choice == "gemini-2.0-flash-thinking-exp-01-21":
        llm = Gemini(model="models/gemini-2.0-flash-thinking-exp-01-21")
    elif llm_choice == "deepseek-reasoner":
        llm = DeepSeek(model="deepseek-reasoner", api_key=api_key)
    elif llm_choice == "deepseek-reasoner-groq":
        llm = Groq(model="deepseek-r1-distill-llama-70b", api_key=groq_api_key)
    elif llm_choice == "deepseek-chat":
        llm = DeepSeek(model="deepseek-chat", api_key=api_key)

    Settings.llm = llm

    st.write("Select the operation you want to perform")

    add_radio = st.radio(
        "**Select Operation**",
        options=(
            "upload your documents",
            "Chat with your knowledge base",
        )
    )

if add_radio == "upload your documents":
    uploaded_file = st.file_uploader("Choose file", type=["pdf"])
    file_uploader_button = st.button("Upload", icon="⬆️")
    if uploaded_file and file_uploader_button:
        save_path = save_uploaded_file(uploaded_file, "files")
        ingest_pdf(save_path, PERSIST_DIR)
        st.write("your document is successfully saved!")

elif add_radio == "Chat with your knowledge base":
    query = st.text_input("Enter your question")
    if st.button("Answer", icon="💬"):
        if query:
            answer = query_kb(query)
            st.write(answer)
        else:
            st.warning("Please enter a query first.")





