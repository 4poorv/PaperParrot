from os import getenv, path

import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

# Load .env variables
load_dotenv()


def save_file_to_uploads_dir(uploaded_file):
    """Write the mutable uploaded file object to disk"""
    uploaded_file_path = path.join(getenv('UPLOAD_DIRECTORY'), uploaded_file.name)
    try:
        with open(uploaded_file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
    except Exception as e:
        st.error(f"An error occurred while saving the file: {str(e)}")
        return None
    return uploaded_file_path


def get_pdf_vector_store(pdf_path):
    """
    Load a PDF into a vector database and then into a langchain toolkit
    :param pdf_path: path to the PDF
    :return: a vector database and a langchain toolkit
    """
    # Load documents into vector database aka ChromaDB
    pages = PyPDFLoader(pdf_path).load_and_split()

    # Convert the PDF pages into a document store
    return Chroma.from_documents(pages, collection_name='annualreport')


def initialize_layout():
    """Initialize application layout."""
    st.sidebar.markdown("## PaperParrot")
    st.sidebar.markdown("A POC for processing PDF using LLM")
    st.title('PaperParrot')
