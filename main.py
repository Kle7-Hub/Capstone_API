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
    response = requests.post(MODEL_URL, headers=headers, data=image_bytes)

    raw = response.json()

    # Case 1: HF returns a list of predictions
    if isinstance(raw, list) and len(raw) > 0:
        # Find AI and Human scores if they exist
        ai_score = None
        human_score = None

        for item in raw:
            label = item.get("label", "").lower()
            score = item.get("score", 0)

            if label == "ai":
                ai_score = score
            if label == "human":
                human_score = score

        # If both scores exist, choose the higher one
        if ai_score is not None and human_score is not None:
            return {
                "aiGenerated": ai_score > human_score,
                "confidence": max(ai_score, human_score)
            }

        # If only one exists, use it
        if ai_score is not None:
            return {"aiGenerated": True, "confidence": ai_score}
        if human_score is not None:
            return {"aiGenerated": False, "confidence": human_score}

    # Case 2: HF returns a dict with different keys
    if isinstance(raw, dict):
        return {
            "aiGenerated": raw.get("ai_generated") or raw.get("is_ai") or False,
            "confidence": raw.get("confidence") or raw.get("confidence_score") or 0
        }

    # Fallback
    return {"aiGenerated": False, "confidence": 0}
