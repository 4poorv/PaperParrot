from langchain.agents.agent_toolkits import VectorStoreInfo, VectorStoreToolkit, create_vectorstore_agent

import utils as utils


class AgentFactory:
    """
    A factory class for creating Langchain agents.
    """

    def __init__(self, llm):
        """
        Initialize the factory with a Langchain language model.

        Args:
            llm: A Langchain language model.
        """
        self.llm = llm

    def create_agent(self, pdf_path):
        """
        Create a Langchain agent for a given PDF file.

        Args:
            pdf_path: The path to the PDF file.

        Returns:
            A Langchain agent.
        """
        # Load documents into vector database aka ChromaDB
        vector_db_store = utils.get_pdf_vector_store(pdf_path)

        # Convert the document store into a langchain toolkit
        toolkit = VectorStoreToolkit(vectorstore_info=VectorStoreInfo(
                name="pdf",
                description="A general PDF vectorstore",
                vectorstore=vector_db_store
        ))

        # Add the toolkit to an end-to-end LC
        return create_vectorstore_agent(llm=self.llm, toolkit=toolkit, verbose=True)
