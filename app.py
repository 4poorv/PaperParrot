from os import getenv, path

import streamlit as st
from dotenv import load_dotenv
from langchain.agents.agent_toolkits import VectorStoreInfo, VectorStoreToolkit, create_vectorstore_agent
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

# Load environment variables from .env file
load_dotenv()

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True, api_key=getenv('OPENAI_API_KEY'))

st.title('Investment Advisor')

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# If a file has been uploaded
if uploaded_file is not None:
    # Save the uploaded file
    file_path = path.join(getenv('UPLOAD_DIRECTORY'), uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getvalue())

    # Load documents into vector database aka ChromaDB
    pages = PyPDFLoader(file_path).load_and_split()
    store = Chroma.from_documents(pages, collection_name='annualreport')

    # Convert the document store into a langchain toolkit
    toolkit = VectorStoreToolkit(vectorstore_info=VectorStoreInfo(
            name="annual_report",
            description="a banking annual report as a pdf",
            vectorstore=store
    ))

    # Add the toolkit to an end-to-end LC
    agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)

    if prompt := st.text_input('Input your prompt here'):
        st.write(agent_executor.run(prompt))  # Then pass the prompt to the LLM and write it out to the screen

        # With a streamlit expander
        with st.expander('Document Similarity Search'):
            # Find the relevant pages and Write out the first
            st.write(store.similarity_search_with_score(prompt)[0][0].page_content)
