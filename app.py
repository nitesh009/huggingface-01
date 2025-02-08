from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

# Load the trained model
classifier = pipeline("sentiment-analysis")

@app.get("/predict/")
def predict(text: str):
    return classifier(text)


#  Run the API with: python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload