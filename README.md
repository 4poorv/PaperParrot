# Streamlit Application for PDF Processing using Langchain

This application allows users to process PDF files using the Langchain library. The application takes a PDF file as
input, generates a language model agent for it, and allows the user to interact with the agent using a question-prompt
interface.

## Installation

To install the application, you will need:

* Python 3.10+
* Poetry

Once you have installed Python and Poetry, you can install the application by running the following command:

```
poetry install
```

## Usage

To use the application, run the following command:

```
streamlit run app.py
```

The application will open in your web browser. You can then upload a PDF file and interact with the agent using the
question-prompt interface.

## Enhancements

The following are some enhancements that could be made to the application:

* Support for multiple file types
* File preprocessing
* Advanced text extraction
* User accounts and authentication
* Scalability
* Text analysis visualization

## License

The application is licensed under the MIT License.

## Design

### System Overview

The application is split into three primary components:

1. **App Class**: The primary class responsible for setting up the user interface, managing user interactions, and
   maintaining the application state.
2. **PDFAgentGenerator Class**: A helper class responsible for handling tasks related to individual PDF files, including
   file saving, vector store loading, and agent creation.
3. **Utils Module**: A set of utility functions for performing common tasks such as saving uploaded files, loading PDFs
   into a vector store, and initializing the application layout.

The application uses Streamlit's session states feature to persist data across page reloads. Two session states are
used: `token_validated`, a boolean flag indicating whether the shared token has been validated, and `pdf_handlers`, a
dictionary of `PDFAgentGenerator` instances, keyed by the names of the uploaded PDF files.

### Component Design

#### App Class

The `App` class manages user interactions and the overall application state. It initializes the Streamlit interface,
validates the shared token, provides a file upload interface, and creates instances of `PDFAgentGenerator` for each
uploaded PDF file. These instances are stored in the `pdf_handlers` session state.

#### PDFAgentGenerator Class

Each instance of `PDFAgentGenerator` corresponds to one uploaded PDF file. The class is responsible for:

1. Saving the uploaded file to the server's 'uploads' directory.
2. Loading the file into a vector store using the `get_pdf_vector_store` utility function.
3. Creating a Langchain agent using the loaded vector store.

#### Utils Module

The `Utils` module contains three utility functions:

1. `save_file_to_uploads_dir`: Saves an uploaded file to the server's 'uploads' directory.
2. `get_pdf_vector_store`: Loads a PDF file into a vector store using the `PyPDFLoader` and `Chroma` classes from
   the `langchain` library.
3. `initialize_layout`: Sets up the initial layout of the Streamlit application.

## Enhancements Suggested

The following are some enhancements that could be made to the application:

* Support for multiple file types
* File preprocessing
* Advanced text extraction
* User accounts and authentication
* Scalability
* Text analysis visualization

## Pseudo Logic

The following is a pseudo logic for the application:

1. The user uploads a PDF file.
2. The application saves the file to the server's 'uploads' directory.
3. The application loads the file into a vector store.
4. The application creates a Langchain agent using the loaded vector store.
5. The application displays a question-prompt interface to the user.
6. The user enters a question.
7. The application asks the agent the question.
8. The agent responds to the question.
9. The application displays the agent's response to the user.
10. The user can repeat steps 6-9 as needed.

I hope this is helpful!