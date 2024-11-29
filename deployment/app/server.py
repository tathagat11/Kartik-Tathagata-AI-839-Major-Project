import json
import logging
import uuid
import time
from datetime import datetime
from typing import Dict, List, Optional, Union

import mlflow
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import RobertaTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

file_handler = logging.FileHandler("model_usage.log")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

app = FastAPI(title="Review Rating Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    text: str

class BatchInputData(BaseModel):
    data: List[InputData]

class PredictionResponse(BaseModel):
    rating: int
    confidence: float

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

# Load the model and tokenizer at startup
try:
    print("Loading MLflow model...")
    loaded_model = mlflow.pytorch.load_model(
        "/app/model",
        map_location=torch.device('cpu')  # Explicitly load on CPU
    )
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    device = torch.device("cpu")  # Always use CPU for inference
    loaded_model.to(device)
    loaded_model.eval()
    print("Model loaded successfully on CPU")
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    raise

def process_text(text: str) -> PredictionResponse:
    """Process a single text input and return prediction."""
    try:
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)

        # Get prediction
        with torch.no_grad():
            outputs = loaded_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][prediction].item()

        # Add 1 to prediction since model outputs 0-4 but we want 1-5
        return PredictionResponse(
            rating=prediction + 1,
            confidence=confidence
        )
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add structured logging
@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: InputData):
    """Endpoint for single prediction"""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        prediction = process_text(input_data.text)
        duration = time.time() - start_time
        
        logger.info("Prediction completed", extra={
            'request_id': request_id,
            'text_length': len(input_data.text),
            'duration_ms': round(duration * 1000, 2),
            'prediction': prediction.rating,
            'confidence': prediction.confidence
        })
        return prediction
        
    except Exception as e:
        logger.error("Prediction failed", extra={
            'request_id': request_id,
            'error': str(e)
        })
        raise

@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(input_data: BatchInputData):
    """Endpoint for batch predictions"""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    batch_size = len(input_data.data)
    
    try:
        predictions = [process_text(item.text) for item in input_data.data]
        duration = time.time() - start_time
        
        logger.info("Batch prediction completed", extra={
            'request_id': request_id,
            'batch_size': batch_size,
            'avg_duration_ms': round((duration * 1000) / batch_size, 2),
            'inputs': [item.text[:100] + '...' if len(item.text) > 100 else item.text for item in input_data.data],
            'predictions': [{'rating': p.rating, 'confidence': p.confidence} for p in predictions],
        })
        return BatchPredictionResponse(predictions=predictions)
        
    except Exception as e:
        logger.error("Batch prediction failed", extra={
            'request_id': request_id,
            'batch_size': batch_size,
            'error': str(e)
        })
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": loaded_model is not None}

@app.get("/logs")
async def get_logs(
    limit: int = Query(100, description="Number of log entries to return"),
    level: Optional[str] = Query(None, description="Log level filter"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
) -> List[Dict]:
    """Retrieve filtered logs"""
    try:
        with open("model_usage.log", "r") as f:
            logs = f.readlines()[-limit:]
            
        filtered_logs = []
        for log in logs:
            log_entry = parse_log_line(log)
            
            if level and log_entry["level"] != level:
                continue
                
            if start_date:
                log_date = datetime.strptime(log_entry["timestamp"], "%Y-%m-%d %H:%M:%S,%f")
                if log_date < datetime.strptime(start_date, "%Y-%m-%d"):
                    continue
                    
            if end_date:
                log_date = datetime.strptime(log_entry["timestamp"], "%Y-%m-%d %H:%M:%S,%f")
                if log_date > datetime.strptime(end_date, "%Y-%m-%d"):
                    continue
                    
            filtered_logs.append(log_entry)
            
        return filtered_logs
    except Exception as e:
        logger.error(f"Error retrieving logs: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving logs")

def parse_log_line(line: str) -> Dict:
    """Parse a log line into a structured format"""
    try:
        parts = line.split(" - ")
        return {
            "timestamp": parts[0],
            "logger": parts[1],
            "level": parts[2],
            "message": parts[3].strip()
        }
    except Exception:
        return {"raw_log": line.strip()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)