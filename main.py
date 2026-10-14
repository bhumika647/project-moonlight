from contextlib import asynccontextmanager
from pathlib import Path
import pickle
import re

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from keras.models import load_model
from keras.utils import pad_sequences
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / "BiGRU.keras"
tokenizer_path = BASE_DIR / "tokenizer.pkl"
max_sequence_length = 50

emotions = ["anger", "fear", "joy", "love", "sadness", "surprise"]


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to be analyzed",
    )


class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence: float
    all_probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


dl_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    dl_model["BiGRU"] = load_model(model_path)

    with open(tokenizer_path, "rb") as file:
        dl_model["Tokenizer"] = pickle.load(file)

    print("Model and tokenizer loaded successfully.")

    yield

    dl_model.clear()


app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", model_loaded=bool(dl_model))


@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):
    model = dl_model.get("BiGRU")
    tokenizer = dl_model.get("Tokenizer")

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded")

    cleaned_text = preprocess_text(text_input.text)

    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(
        sequence,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post",
    )

    probabilities = model.predict(padded, verbose=0)[0]
    top_index = int(np.argmax(probabilities))

    all_probabilities = {
        label: float(prob) for label, prob in zip(emotions, probabilities)
    }

    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=emotions[top_index],
        confidence=float(probabilities[top_index]),
        all_probabilities=all_probabilities,
    )