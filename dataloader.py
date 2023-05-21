from langchain.agents.agent_toolkits import VectorStoreInfo, VectorStoreToolkit, create_vectorstore_agent
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma


class DataLoader:
    """
    Load a PDF into a vector database and then into a langchain toolkit
    """

    def __init__(self, llm):
        self.llm = llm

    def process_pdf(self, pdf_path):
        """
        Load a PDF into a vector database and then into a langchain toolkit
        :param pdf_path: path to the PDF
        :return: a vector database and a langchain toolkit
        """
        # Load documents into vector database aka ChromaDB
        pages = PyPDFLoader(pdf_path).load_and_split()

        store = Chroma.from_documents(pages, collection_name='annualreport')

        # Convert the document store into a langchain toolkit
        toolkit = VectorStoreToolkit(vectorstore_info=VectorStoreInfo(
                name="annual_report",
                description="a banking annual report as a pdf",
                vectorstore=store
        ))

        # Add the toolkit to an end-to-end LC
        agent_executor = create_vectorstore_agent(llm=self.llm, toolkit=toolkit, verbose=True)

        return store, agent_executor
