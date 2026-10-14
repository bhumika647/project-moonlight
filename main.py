from contextlib import asynccontextmanager
import re
import numpy as np
import pickle

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field
from keras.models import load_model
from tensorflow import keras
from keras.utils import pad_sequences

from tensorflow.keras.preprocessing.sequence import pad_sequences

from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / "BiGRU.keras"

tokenizer_path = BASE_DIR / "tokenizer.pickle"
max_sequence_length = 50


emotions = [
    "anger",
    "fear",
    "joy",
    "love",
    "sadness",
    "surprise"
]

EMOTION_EMOJIS = {
    "sadness": "😔",
    "joy": "😊",
    "love": "❤️",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😮",
}


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r""" """ , " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to be analyzed",
        json_schema_extra={
            "example": {
                "text": "I am so happy today!"
            }
        }
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
    model_path = "BiGRU.keras"
    tokenizer_path = "tokenizer.pkl"

    dl_model["BiGRU"] = load_model(model_path)

    with open(tokenizer_path, "rb") as file:
        dl_model["Tokenizer"] = pickle.load(file)

    print("Model and tokenizer loaded successfully.")

    yield

    dl_model.clear()

    


app = FastAPI(
    lifespan=lifespan
)


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Static files
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse("static/index.html")


@app.get("/health", response_model=HealthResponse)
def health_check():

    return HealthResponse(
        status="ok",
        model_loaded=bool(dl_model)
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):

    # Get model and tokenizer
    BiGRU_model = dl_model.get("BiGRU")
    tokenizer = dl_model.get("Tokenizer")

    if BiGRU_model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model or tokenizer not loaded"
        )

    # Preprocess input
    cleaned_text = preprocess_text(text_input.text)

    # Convert text to sequence
    tokenized_text = tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    # Padding
    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post"
    )

    # Prediction
    probabilities = BiGRU_model.predict(
        padded_sequence,
        verbose=0
    )[0]

    # Get highest probability
    top_emotion_index = int(
        np.argmax(probabilities)
    )

    # All probabilities
    all_probabilities = {
        label: float(prob)
        for prob, label in zip(emotions, probabilities)
    }

    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=emotions[top_emotion_index],
        confidence=float(
            probabilities[top_emotion_index]
        ),
        all_probabilities=all_probabilities
    )
