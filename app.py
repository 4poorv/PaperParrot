from os import getenv, path

import streamlit as st
from dotenv import load_dotenv
from langchain.agents.agent_toolkits import VectorStoreInfo, VectorStoreToolkit, create_vectorstore_agent
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

# Load environment variables from .env file
load_dotenv()

input_shared_token = st.text_input("Enter shared token", type="password")
if input_shared_token.strip() != getenv('SHARING_TOKEN'):
    st.stop()

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True, api_key=getenv('OPENAI_API_KEY'))

st.title('PaperParrot')

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# If a file has been uploaded
if uploaded_file is not None:
    # Save the uploaded file
    file_path = path.join(getenv('UPLOAD_DIRECTORY'), uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getvalue())

    st.success('Congratulations! PDF successfully uploaded.')

    st.info("Hold on tight! We're loading the PDF and training ChatGPT")
    progress = st.progress(0)

    # Load documents into vector database aka ChromaDB
    pages = PyPDFLoader(file_path).load_and_split()

    # Update progress bar
    progress.progress(50)

    store = Chroma.from_documents(pages, collection_name='annualreport')

    # Convert the document store into a langchain toolkit
    toolkit = VectorStoreToolkit(vectorstore_info=VectorStoreInfo(
            name="annual_report",
            description="a banking annual report as a pdf",
            vectorstore=store
    ))

    # Add the toolkit to an end-to-end LC
    agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)

    # Update progress bar to 100%
    progress.progress(100)
    st.success('PDF is loaded and our model is trained. Get ready to ask your side-splitting questions.')

    # Print a summary of the PDF (e.g. the first page)
    # st.info(f"PDF Summary:\n\n{pages[0]}")

    if prompt := st.text_input('Fire away with your follow-up questions!'):
        st.write(agent_executor.run(prompt))  # Then pass the prompt to the LLM and write it out to the screen

        # With a streamlit expander
        with st.expander('Related Sections for Above Information'):
            # Find the relevant pages and Write out the first
            st.write(store.similarity_search_with_score(prompt)[0][0].page_content)
