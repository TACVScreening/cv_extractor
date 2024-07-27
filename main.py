from fastapi import FastAPI, File, UploadFile
from pdf2image import convert_from_bytes
from fastapi.middleware.cors import CORSMiddleware
import contentExtractor
import entitiesExtractor
import cosinesim
import time

from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/getInformation/")
async def getInformation(file: UploadFile = File(...)):
    filename = file.filename

    # Read the file data into bytes
    file_bytes = await file.read()
    
    # Convert PDF to images using bytes
    images = convert_from_bytes(file_bytes)

    # Read text from images using Tesseract OCR
    start_time = time.time()
    text = contentExtractor.extract_text_from_file(images)
    end_time = time.time()

    execution_time = end_time - start_time

    information = entitiesExtractor.Gpt_infer(text)

    cosine_similarity = cosinesim.calculate_cosinesim(text, information["entities"])

    return {
        "filename": filename,
        "execution_time": execution_time,
        "total_token": information["total_token"],
        "entities": information["entities"],
        "cosine_similarity": cosine_similarity
    }