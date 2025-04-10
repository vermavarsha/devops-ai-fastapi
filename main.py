# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

class ReviewInput(BaseModel):
    review: str

@app.post("/predict")
def predict_sentiment(data: ReviewInput):
    review_text = [data.review]
    vectorized_text = vectorizer.transform(review_text)
    prediction = model.predict(vectorized_text)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    return {"sentiment": sentiment}
