from os import getenv, path

import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI

from dataloader import DataLoader


class App:
    def __init__(self):
        load_dotenv()

        self.shared_token = st.text_input("Enter shared token", type="password")

        if self.shared_token.strip() != getenv('SHARING_TOKEN'):
            st.stop()

        self.llm = OpenAI(temperature=0.1, verbose=True, api_key=getenv('OPENAI_API_KEY'))

        ## todo add sidebar

        st.title('PaperParrot')

        if self.uploaded_file is not None:
            self.handle_file_upload()
            self.train_model()

    def train_model(self):
        """
        Train the model on the uploaded PDF
        """
        st.info("Hold on tight! We're loading the PDF and training ChatGPT")
        progress = st.progress(0)

        vector_db_manager = DataLoader(self.llm)
        self.store, self.agent_executor = vector_db_manager.process_pdf(self.uploaded_file_path)

        # Update progress bar to 100%
        progress.progress(100)
        st.success('PDF is loaded and our model is trained. Get ready to ask your side-splitting questions.')

        if prompt := st.text_input('Fire away with your follow-up questions!'):
            self.handle_prompt(prompt)

    def handle_file_upload(self):
        """
        Handle the file upload
        """
        self.uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        # Save the uploaded file
        self.uploaded_file_path = path.join(getenv('UPLOAD_DIRECTORY'), self.uploaded_file.name)
        with open(self.uploaded_file_path, 'wb') as f:
            f.write(self.uploaded_file.getvalue())

        st.success('Congratulations! PDF successfully uploaded.')

    def handle_prompt(self, prompt):
        """
        Handle the prompt
        """
        st.write(self.agent_executor.run(prompt))  # Then pass the prompt to the LLM and write it out to the screen

        # With a streamlit expander
        with st.expander('Related Sections for Above Information'):
            # Find the relevant pages and Write out the first
            st.write(self.store.similarity_search_with_score(prompt)[0][0].page_content)


if __name__ == "__main__":
    App()
