import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import fitz  # PyMuPDF for PDFs
import json

# LangChain / OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from openai import OpenAI

# -------------------
# Setup
# -------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Please set your OPENAI_API_KEY in .env")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
EMBED_MODEL = "all-MiniLM-L6-v2"
PERSIST_DIR = "chroma_db"

# -------------------
# Simple File Reader
# -------------------
def read_file(uploaded_file):
    path = Path(uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    text = ""
    if uploaded_file.name.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    elif uploaded_file.name.endswith(".pdf"):
        with fitz.open(path) as doc:
            for page in doc:
                text += page.get_text()
    os.remove(path)
    return text

# -------------------
# Classification
# -------------------
def classify_doc(text, filename):
    prompt = f"""
    You are an expert document classifier. Classify this document as one of:
    contract, invoice, earnings report, or unknown.
    Then extract simple metadata.

    Return JSON with keys: type, metadata.
    Document text:
    {text[:1500]}
    """
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = resp.choices[0].message.content.strip()
    try:
        return json.loads(content.replace("'", '"'))
    except:
        return {"type": "unknown", "metadata": {"raw_output": content}}

# -------------------
# RAG Q&A
# -------------------
def build_vector_store(texts):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma.from_texts(texts, embeddings, persist_directory=PERSIST_DIR)

def get_rag_chain(vector_store):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="Doc Classifier + Q&A", layout="wide")
st.title("📄 Document Classifier + Analyzer")

uploaded = st.file_uploader("Upload a TXT or PDF", type=["txt", "pdf"])

if uploaded:
    text = read_file(uploaded)

    # --- Classification ---
    st.subheader("📑 Classification")
    result = classify_doc(text, uploaded.name)
    st.json(result)

    # --- RAG Q&A ---
    st.subheader("🤖 Ask Questions")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    docs = splitter.split_text(text)
    vector_store = build_vector_store(docs)
    qa = get_rag_chain(vector_store)

    query = st.text_input("Ask something about this document")
    if query:
        answer = qa.run(query)
        st.success(answer)
