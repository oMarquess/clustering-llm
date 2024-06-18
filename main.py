from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel
from typing import List, Optional
from uuid import UUID, uuid4
from starlette.formparsers import MultiPartParser
from dotenv import load_dotenv
import os


app = FastAPI()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
from routers.cluster import router

app.include_router(router)

#faster
# @app.post("/upload/")
# async def file_endpoint(uploaded_file: UploadFile):
#     content = await uploaded_file.read()
#     print(content)





if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
