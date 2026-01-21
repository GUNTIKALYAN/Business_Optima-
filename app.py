from fastapi import FastAPI
import uvicorn
import joblib
import re
from schemas import TextRequest, PredictionResponse

model = joblib.load("./smart_text_classification.joblib")
tfidf = joblib.load("./tfidf_vectorizer.joblib")


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\{.*?\}", "", text)
    text = text.replace("\xa0", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

app = FastAPI(title="Smart Text Classification")

@app.get("/")
def greet():
    return {"message" : "Welcome to smart text prediction type /docs to predict"}


@app.post("/predict",response_model=PredictionResponse)
def text_prediction(request:TextRequest):
    cleaned = clean_text(request.text)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    return {"category" : prediction}

if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)
