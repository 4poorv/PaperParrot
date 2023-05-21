from os import getenv, path

import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI

from dataloader import DataLoader


class App:
    """A Streamlit application for PaperParrot."""

    def __init__(self):
        """Initialize the application."""
        load_dotenv()
        self.uploaded_file = None
        self.uploaded_file_path = None
        self.store = None
        self.agent_executor = None

        self.shared_token = st.text_input("Enter shared token", type="password")
        if self.shared_token.strip() != getenv('SHARING_TOKEN'):
            st.stop()
        # Add sidebar with description
        st.sidebar.markdown("## PaperParrot")
        st.sidebar.markdown("A POC for processing PDF using LLM")
        st.title('PaperParrot')

        self.llm = OpenAI(temperature=0.1, verbose=True, api_key=getenv('OPENAI_API_KEY'))

        self.uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if self.uploaded_file is not None:
            self.save_file_to_uploads_dir()
            self.train_model()
            if prompt := st.text_input('Fire away with your follow-up questions!'):
                self.handle_prompt(prompt)

    def train_model(self):
        """Train the model with the uploaded PDF."""
        st.info("Hold on tight! We're loading the PDF and training ChatGPT")
        progress = st.progress(0)

        vector_db_manager = DataLoader(self.llm)
        self.store, self.agent_executor = vector_db_manager.process_pdf(self.uploaded_file_path)

        # Update progress bar to 100%
        progress.progress(100)
        st.success('PDF is loaded and our model is trained. Get ready to ask your side-splitting questions.')

    def save_file_to_uploads_dir(self):
        """Write the mutable uploaded file object to disk"""
        self.uploaded_file_path = path.join(getenv('UPLOAD_DIRECTORY'), self.uploaded_file.name)
        with open(self.uploaded_file_path, 'wb') as f:
            f.write(self.uploaded_file.getvalue())

    def handle_prompt(self, prompt):
        """Handle the user prompt."""
        st.write(self.agent_executor.run(prompt))  # Then pass the prompt to the LLM and write it out to the screen

        # With a streamlit expander
        with st.expander('Related Sections for Above Information'):
            # Find the relevant pages and Write out the first
            st.write(self.store.similarity_search_with_score(prompt)[0][0].page_content)


if __name__ == "__main__":
    App()
