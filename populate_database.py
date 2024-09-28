import argparse
import os
import shutil
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
# from langchain_community.vectorstores.chroma import Chroma
from langchain_chroma import Chroma


CHROMA_PATH = "chroma"
DATA_PATH = "data"
DATA_PATH = 'data/books'
DATA_PATH = 'sm_data'


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def load_single_pdf(file_path: str):
    """
    Load a single PDF file.
    :param file_path: The path to the PDF file.
    :return: A list of documents loaded from the PDF.
    """
    document_loader = PyPDFLoader(file_path)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        # db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


def remove_document_from_chroma(document_name: str):
    """
    Remove all chunks from the Chroma vector store that belong to a specific document.

    :param document_name: The name of the document to remove (e.g., 'example.pdf').
    """
    # Load the existing Chroma database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Fetch all items in the database.
    # Get metadata to filter based on source
    existing_items = db.get(include=["metadatas"])

    # Find chunks that belong to the specified document.
    ids_to_remove = []
    for i, metadata in enumerate(existing_items["metadatas"]):
        if document_name in metadata.get("source"):
            # Add the corresponding chunk ID to the removal list.
            ids_to_remove.append(existing_items["ids"][i])

    if ids_to_remove:
        print(
            f"🗑 Removing {len(ids_to_remove)} chunks from document: {document_name}")
        # Remove these chunks from the Chroma vector store.
        db.delete(ids=ids_to_remove)
        print("✅ Document removed successfully.")
    else:
        print(f"⚠️ No chunks found for document: {document_name}")


if __name__ == "__main__":
    # main()
    remove_document_from_chroma(
        '1704.03155v2.pdf')
