from langchain.agents.agent_toolkits import VectorStoreInfo, VectorStoreToolkit, create_vectorstore_agent

import utils as utils


class PDFlangchainAgent:
    """
    Load a PDF into a vector database and then into a langchain agent
    """

    def __init__(self, llm):
        self.llm = llm
        self.vector_db_store = None

    def get_vectorstore_agent(self, pdf_path):
        """
        Load a PDF into a vector database and then into a langchain toolkit
        """
        # Load documents into vector database aka ChromaDB
        self.vector_db_store = utils.get_pdf_vector_store(pdf_path)

        # Convert the document store into a langchain toolkit
        toolkit = VectorStoreToolkit(vectorstore_info=VectorStoreInfo(
                name="pdf",
                description="A general PDF vectorstore",
                vectorstore=self.vector_db_store
        ))

        # Add the toolkit to an end-to-end LC
        return create_vectorstore_agent(llm=self.llm, toolkit=toolkit, verbose=True)
