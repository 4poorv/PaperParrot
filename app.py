import streamlit as st
from dotenv import load_dotenv
from langchain import LangchainAgent
from langchain.chat_models import openai


class DocumentProcessor:
    def __init__(self, uploads_dir):
        self.uploads_dir = uploads_dir

    def process_file(self, file_name):
        with open(f"{self.uploads_dir}/{file_name}", "rb") as f:
            file_content = f.read()

        # Preprocess the file if necessary
        # For example, extract text from images

        # Load the file into a vector store
        vector_store = LangchainVectorStore()
        vector_store.load_from_file(file_content)

        return vector_store


class Agent:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm_agent = LangchainAgent(self.vector_store)

    def predict(self, question):
        return self.llm_agent.predict(question)


class App:
    def __init__(self):
        self.document_processor = DocumentProcessor(load_dotenv(".env").get("UPLOADS_DIR"))

        # Load OpenAI API token
        openai.api_key = load_dotenv(".env").get("OPENAI_API_KEY")

    def run(self):
        # Initialize Streamlit interface
        st.title("PDF File Processing Application")

        # Display file upload interface
        file_name = st.file_uploader("Upload a PDF file", type="pdf")

        if file_name is not None:
            # Process the file
            vector_store = self.document_processor.process_file(file_name)

            # Create an agent for the processed file
            agent = Agent(vector_store)

            # Display an interaction interface for the agent
            st.write("Here is an interaction interface for the agent:")
            st.text_input("Enter a question:")
            st.write("Answer:", agent.predict(st.text_input("Enter a question:")))

        # Display a sidebar with all uploaded files
        uploaded_files = st.sidebar.selectbox("Select a file", list(self.document_processor.uploads_dir.glob("*.pdf")))

        if uploaded_files is not None:
            # Get the vector store for the selected file
            vector_store = self.document_processor.process_file(uploaded_files)

            # Create an agent for the selected file
            agent = Agent(vector_store)

            # Display an interaction interface for the agent
            st.sidebar.write("Here is an interaction interface for the agent:")
            st.sidebar.text_input("Enter a question:")
            st.sidebar.write("Answer:", agent.predict(st.sidebar.text_input("Enter a question:")))


if __name__ == "__main__":
    app = App()
    app.run()
