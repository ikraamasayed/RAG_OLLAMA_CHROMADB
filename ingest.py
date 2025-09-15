import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm
from config import DOCS_PATH, DB_PATH

# Paths

# Ensure directory exists
DB_PATH.mkdir(parents=True, exist_ok=True)

def load_documents():
    loaders = [
        DirectoryLoader(DOCS_PATH, glob="**/*.txt",
                        loader_cls=lambda path: TextLoader(path, encoding="utf-8"),
                        show_progress=True),
        DirectoryLoader(DOCS_PATH, glob="**/*.pdf",
                        loader_cls=PyPDFLoader,
                        show_progress=True),
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs


def create_vector_db():
    print("Loading documents...")
    documents = load_documents()

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks to embed: {len(chunks)}")
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

    print("Storing in ChromaDB with progress...")
    vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # Insert in batches with progress bar
    batch_size = 200
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding & storing"):
        batch = chunks[i:i + batch_size]
        vectordb.add_documents(batch)

    print(f"✅ Database updated at {DB_PATH}")

if __name__ == "__main__":
    create_vector_db()
