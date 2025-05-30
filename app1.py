import streamlit as st
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import os

# Model & PDF setup
MODEL_PATH = "/content/drive/MyDrive/GEN AI/llama-2-7b-chat.ggmlv3.q8_0.bin"
PDF_DIR = "/content/drive/MyDrive/PDF_Files"
VDB_PATH = "db/faiss"

@st.cache_resource
def load_vector_store():
    if not os.path.exists(VDB_PATH):
        # Load PDFs
        loader = DirectoryLoader(PDF_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()

        # Chunk text
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # Embeddings
        embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        # Vector store
        vectordb = FAISS.from_documents(chunks, embedder)
        vectordb.save_local(VDB_PATH)
    else:
        embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        vectordb = FAISS.load_local(VDB_PATH, embedder, allow_dangerous_deserialization=True)


    return vectordb

@st.cache_resource
def load_llm():
    return CTransformers(
        model=MODEL_PATH,
        model_type="llama",
        config={
            "max_new_tokens": 512,
            "temperature": 0.5,
            "context_length": 2048,
        }
    )

st.set_page_config(page_title="Resume Q&A RAG", layout="centered")
st.title("📄 Resume-Based RAG System")

query = st.text_input("Ask a question based on your resume documents:")

if query:
    with st.spinner("Setting up model & vector store..."):
        vectordb = load_vector_store()
        llm = load_llm()

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(),
            chain_type="stuff"
        )

    with st.spinner("Generating answer..."):
        result = qa_chain.run(query)

    st.subheader("📌 Answer")
    st.success(result)
