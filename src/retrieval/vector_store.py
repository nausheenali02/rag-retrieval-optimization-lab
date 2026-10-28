
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


def create_vector_store(
    chunks: list[Document],
    persist_directory: str = "chroma_db",
) -> Chroma:
    """Create a fresh Chroma vector store from document chunks."""

    database_path = Path(persist_directory)

    # Remove the previous local index so chunks are not duplicated.
    if database_path.exists():
        import shutil
        shutil.rmtree(database_path)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="rag_retrieval_baseline",
        persist_directory=persist_directory,
    )

    return vector_store
