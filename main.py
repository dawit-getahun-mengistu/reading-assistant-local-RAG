from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader

from fastapi import FastAPI, File, Request, UploadFile
import uvicorn
import os


app = FastAPI()
chached_llm = Ollama(model='gemma:2b')
UPLOAD_DIR = 'data/'
CHROMA_DB = 'db'

embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)


@app.get("/")
async def root():
    return {"message": "RAG chat bot"}


@app.post("/ai")
async def aiPost(request: Request):
    print("Post /ai called")

    json_content = await request.json()
    query = json_content.get("query")
    print(f"query: {query}")

    response = chached_llm.invoke(query)
    print(f"response: {response}")
    return {"answer": response}


@app.post("/upload-file")
async def uploadFile(file: UploadFile = File(...)):
    file_name = file.filename
    save_path = os.path.join(UPLOAD_DIR, file_name)

    # Save file
    with open(save_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    print(f"filename: {file_name}")

    loader = PDFPlumberLoader(save_path)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=CHROMA_DB)
    # vector_store.persist()

    response = {
        "status": "successfully uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }

    return response


def start_app():
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    if not os.path.exists(CHROMA_DB):
        os.makedirs(CHROMA_DB)

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":

    start_app()
