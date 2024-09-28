from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

from fastapi import FastAPI, File, Request, UploadFile
import uvicorn
import os

from get_embedding_function import get_embedding_function
from populate_database import load_single_pdf, load_documents, split_documents, add_to_chroma, calculate_chunk_ids, remove_document_from_chroma


app = FastAPI()
chached_llm = Ollama(model='mistral')
UPLOAD_DIR = 'data/'
CHROMA_DB = 'chroma'

embedding = get_embedding_function()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template(
    """
    <s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information, say so. [/INST] </s>
    [INST] {input}
        Context: {context}
        Answer:
    [/INST]                                          
    """)


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


@app.post('/ask_pdf')
async def askPDFPost(request: Request):
    print("Post /ask_pdf called")
    json_content = await request.json()
    query = json_content.get("query")
    print(f"query: {query}")

    print("Loading vector store")
    vector_store = Chroma(persist_directory=CHROMA_DB,
                          embedding_function=embedding)

    print("Creating Chain")
    retriever = vector_store.as_retriever(
        search_type='similarity_score_threshold',
        search_kwargs={
            "k": 20,
            "score_threshold": 0.5
        }
    )

    document_chain = create_stuff_documents_chain(chached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    result = chain.invoke({"input": query})

    print(f"response: {result['answer']}")
    return {"answer": result['answer']}


@app.post("/upload-file")
async def uploadFile(file: UploadFile = File(...)):
    file_name = file.filename
    save_path = os.path.join(UPLOAD_DIR, file_name)

    # Save file
    with open(save_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    print(f"filename: {file_name}")

    # loader = PDFPlumberLoader(save_path)
    # docs = loader.load_and_split()
    # print(f"docs len={len(docs)}")

    # chunks = text_splitter.split_documents(docs)
    # print(f"chunks len={len(chunks)}")

    # vector_store = Chroma.from_documents(
    #     documents=chunks, embedding=embedding, persist_directory=CHROMA_DB)
    # vector_store.persist()
    docs = load_single_pdf(save_path)
    chunks = split_documents(docs)
    add_to_chroma(chunks)

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
