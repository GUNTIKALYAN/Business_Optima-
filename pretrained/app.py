from fastapi import FastAPI
import uvicorn
from schema import TextRequest, PredictionResponse

import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="Smart Text Classifier")

MODEL_DIR = "intent_classifier_distilbert"

# Load model & tokenizer once 
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

with open(f"{MODEL_DIR}/label_mapping.json") as f:
    id2label = json.load(f)
    id2label = {int(k): v for k, v in id2label.items()}  # ensure int keys


@app.get("/")
def health():
    return {"message": "Welcome to smart text prediction type /docs to predict"}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: TextRequest):
    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    pred_id = torch.argmax(outputs.logits, dim=1).item()
    intent = id2label[pred_id]

    return {"intent": intent}


if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)
