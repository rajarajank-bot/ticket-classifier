from contextlib import asynccontextmanager
from typing import List
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models once at startup
    models["category"] = joblib.load("models/baseline_category.joblib")
    models["priority"] = joblib.load("models/baseline_priority.joblib")
    yield
    models.clear()

app = FastAPI(title="Customer Support Ticket Classifier", lifespan=lifespan)

class TicketRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Support ticket description")

class BatchTicketRequest(BaseModel):
    tickets: List[TicketRequest]

class TicketResponse(BaseModel):
    category: str
    priority: str
    confidence: float

def predict_single(text: str) -> TicketResponse:
    clean_text = text.lower().strip()
    if not clean_text:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="Ticket text cannot be empty or whitespace")

    cat_model = models["category"]
    prio_model = models["priority"]

    cat_pred = cat_model.predict([clean_text])[0]
    cat_prob = float(np.max(cat_model.predict_proba([clean_text])))

    prio_pred = prio_model.predict([clean_text])[0]
    prio_prob = float(np.max(prio_model.predict_proba([clean_text])))

    avg_conf = round((cat_prob + prio_prob) / 2, 2)
    return TicketResponse(category=cat_pred, priority=prio_pred, confidence=avg_conf)

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": len(models) == 2}

@app.post("/predict", response_model=TicketResponse)
def predict(request: TicketRequest):
    return predict_single(request.text)

@app.post("/predict-batch", response_model=List[TicketResponse])
def predict_batch(request: BatchTicketRequest):
    return [predict_single(item.text) for item in request.tickets]