import os
from os import getenv

import streamlit as st
from langchain.llms import OpenAI

import utils as utils
from pdfprocessor import PDFlangchainAgent

# Get maximum number of PDFs
MAX_PDFS = int(os.getenv('MAX_PDFS', 10))


class App:
    """A Streamlit application for PaperParrot."""

    def __init__(self):
        """Initialize the application."""
        self.shared_token = None
        self.pdf_vstore_agent = None
        self.pdf_agents = None
        self.active_file_name = None
        self.llm = OpenAI(temperature=0.1, verbose=True, api_key=getenv('OPENAI_API_KEY'))

        # Validate shared token
        self.validate_token()

        if st.session_state['token_validated']:
            # Initialize application layout
            utils.initialize_layout()

            # Handle PDF uploads and model training
            self.handle_uploads()

            # Handle active PDF selection
            self.handle_active_pdf()

    def handle_uploads(self):
        """Handle PDF uploads and model training."""
        if not st.session_state.get('token_validated', False):
            return

        self.pdf_agents = st.session_state.get('pdf_agents', {})
        if len(self.pdf_agents) >= MAX_PDFS:
            st.sidebar.warning(f"Reached maximum number of PDFs ({MAX_PDFS}). Please remove one to continue.")
        else:
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
            if uploaded_file is not None:
                if uploaded_file_path := utils.save_file_to_uploads_dir(uploaded_file):
                    self.pdf_vstore_agent = PDFlangchainAgent(self.llm).get_vectorstore_agent(uploaded_file_path)
                    self.pdf_agents[uploaded_file.name] = self.pdf_vstore_agent
                    # Only rerun on the second upload onwards
                    # if len(self.pdf_agents) == 1:
                    #     st.experimental_rerun()
            # Update session_state
            st.session_state['pdf_agents'] = self.pdf_agents

    def handle_prompt(self, prompt):
        """Handle the user prompt."""
        try:
            response = self.pdf_vstore_agent.run(prompt)
            st.write(response)  # Then pass the prompt to the LLM and write it out to the screen

            # With a streamlit expander
            with st.expander('Related Sections for Above Information'):
                # Find the relevant pages and Write out the first
                related_section = self.pdf_vstore_agent.vector_db_store.similarity_search_with_score(prompt)[0][0].page_content
                st.write(related_section)
        except Exception as e:
            st.error(f"An error occurred while processing the prompt: {str(e)}")

    def validate_token(self):
        """Validate shared token."""

        # Initialize 'token_validated' if not already in session state
        if 'token_validated' not in st.session_state:
            st.session_state['token_validated'] = False

        input_slot = st.empty()

        if not st.session_state['token_validated']:
            self.shared_token = input_slot.text_input("Enter shared token", type="password")
            if self.shared_token:
                if self.shared_token.strip() != getenv('SHARING_TOKEN'):
                    st.error("Invalid token. Please try again.")
                    st.stop()
                else:
                    st.session_state['token_validated'] = True
                    input_slot.empty()  # This line effectively removes the input field

    def handle_active_pdf(self):
        """Handle active PDF selection and question prompt."""
        if not self.pdf_agents:
            return

        self.active_file_name = st.sidebar.selectbox("Select PDF to use", list(self.pdf_agents.keys()))
        if self.active_file_name:
            self.pdf_vstore_agent = self.pdf_agents[self.active_file_name]
            st.sidebar.markdown(f"**Active PDF:** {self.active_file_name}")

            if st.sidebar.button("Remove selected PDF"):
                self.pdf_agents.pop(self.active_file_name, None)
                st.sidebar.success(f"Removed {self.active_file_name}")
            # Update session_state
            st.session_state['pdf_agents'] = self.pdf_agents

            if prompt := st.text_input('Fire away with your follow-up questions!'):
                self.handle_prompt(prompt)

    def train_model(self, uploaded_file_path):
        """Train the model with the uploaded PDF."""
        st.info("Loading the PDF and training ChatGPT...")
        progress = st.progress(0)
        try:
            pdf_lc_agent = PDFlangchainAgent(self.llm)
            self.pdf_vstore_agent = pdf_lc_agent.get_vectorstore_agent(uploaded_file_path)
        except Exception as e:
            st.error(f"An error occurred during model training: {str(e)}")
        # Update progress bar to 100%
        progress.progress(100)
        st.success('PDF is loaded and our model is trained. Get ready to ask your side-splitting questions.')


if __name__ == "__main__":
    App()
