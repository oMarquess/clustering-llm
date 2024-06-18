from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, Response, UploadFile
from typing import Annotated, Any, List, Optional, Union
from fastapi.datastructures import UploadFile as FastAPIUploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from langchain.document_loaders.mongodb import MongodbLoader
from langchain_community.chat_models import BedrockChat
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.document_loaders import DirectoryLoader, S3DirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import UnstructuredExcelLoader, UnstructuredCSVLoader, UnstructuredFileLoader
import os
from dotenv import load_dotenv
from langchain import hub
from langchain.chat_models import ChatOpenAI
import logging
import tempfile

cluster_prompt = hub.pull('sredeemer/cluster')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
AWS_DEFAULT_REGION = os.getenv("AWS_REGION")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# initialize LLM
#llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, openai_api_key = '') 
llm = BedrockChat(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={"temperature": 0.2, "max_tokens": 200000 },
    region_name = AWS_DEFAULT_REGION
)


router = APIRouter(
     prefix="/files",
    responses={404: {"description": "Not found"}},
    #dependencies=[Depends(JWTBearer())],
    tags=["files"],
)

# @router.get("/users/", tags=["users"])
# async def read_user():
#     return [{"Hello world": 23}]


#         #TODO: Make LLM map headings to standard headings
#         # Extract content of relevant headings
#         # Re-format content into one doc well structured
#         # Embed structured content
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




@router.post("/cluster")
async def generate_clustering(files: List[UploadFile] = File(...)):
    results = []
    file_count = 0

    for file in files:
        content = await file.read()
        suffix = ".csv" if file.filename.endswith('.csv') else (".xlsx" if file.filename.endswith('.xlsx') else None)
        
        if suffix:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(content)
                temp_file.flush()

                # Load the documents based on file type
                try:
                    loader_class = UnstructuredCSVLoader if suffix == ".csv" else UnstructuredExcelLoader
                    loader = loader_class(temp_file.name, unstructured_kwargs={"encoding": "latin1", "delimiter": ","})
                    docs = loader.load()

                    # Create and persist document embeddings
                    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    docs = text_splitter.split_documents(docs)
                    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                    db = Chroma.from_documents(docs, embedding_function, ids=None, collection_name="testdoc", persist_directory="./chroma_db")
                    logging.info("Completed embedding process")
                    retriever = db.as_retriever()

                    # Chain the processes: Retrieval -> Formatting -> Clustering -> Parsing response
                    rag_chain = (
                        {"context": retriever, "question": RunnablePassthrough()} | cluster_prompt | llm | StrOutputParser()
                    )
                    clusters = rag_chain.invoke("Tell me the clusters you can find")
                    results.append(clusters)

                    #delete the db collection
                    db.delete_collection()

                except Exception as e:
                    logging.error(f"Failed to process file {file.filename}: {e}")
                    continue  # Skip processing this file
        else:
            logging.error(f"Unsupported file format for file {file.filename}")
            continue  # Skip processing this file

        file_count += 1

    logging.info(f"Total number of files processed: {file_count}, Names of files: {[file.filename for file in files]}")
    return JSONResponse(content={"status": "success", "data": results, "file_count": file_count})
