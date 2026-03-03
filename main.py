from fastapi import FastAPI, File, UploadFile
import requests
import os

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your frontend URL for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_KEY = os.getenv("HF_API_KEY")
MODEL_URL = "https://api-inference.huggingface.co/models/umm-maybe/AI-image-detector"

@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    response = requests.post(
        MODEL_URL,
        headers=headers,
        data=image_bytes
    )

    raw = response.json()

    return {
        "aiGenerated": raw.get("ai_generated", False), 
        "confidence": raw.get("confidence", 0)
    }
