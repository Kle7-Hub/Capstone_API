from fastapi import FastAPI, File, UploadFile
import requests
import os

app = FastAPI()

# Load your HF API key from Render environment variables
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

    return response.json()
