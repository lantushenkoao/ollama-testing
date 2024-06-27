from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf_data(file_path):
    # Create a PyMuPDFLoader object with file_path
    loader = PyMuPDFLoader(file_path=file_path)

    # load the PDF file
    docs = loader.load()

    # return the loaded document
    return docs


# Responsible for splitting the documents into several chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    # Initialize the RecursiveCharacterTextSplitter with
    # chunk_size and chunk_overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Split the documents into chunks
    chunks = text_splitter.split_documents(documents=documents)

    # return the document chunks
    return chunks
