from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import google.generativeai as genai
from supabase import create_client, Client
import os
from typing import Optional
import re
import json
from datetime import datetime

app = FastAPI(title="Fake News Detector API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration - ganti dengan nilai Anda
SUPABASE_URL = "https://blnehwafvmysyyudcglg.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJsbmVod2Fmdm15c3l5dWRjZ2xnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg5NzgwNDEsImV4cCI6MjA3NDU1NDA0MX0.MJPfcA0iBSw5iHNrCmpdjjVGIUv5O4a4R9L9nCyw4Sw"
GEMINI_API_KEY = "AIzaSyBEG-U0N3H-39A8rpZyHh4zhqykz2sMZI4"

# Initialize components
try:
    # Load DistilBERT model
    model_path = "/fake_news_detector_distilbert"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    tokenizer = None
    model = None

try:
    # Configure Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-pro')
    print("✅ Gemini configured successfully")
except Exception as e:
    print(f"❌ Error configuring Gemini: {e}")
    gemini_model = None

try:
    # Initialize Supabase
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Supabase connected successfully")
except Exception as e:
    print(f"❌ Error connecting to Supabase: {e}")
    supabase = None

class NewsRequest(BaseModel):
    text: str
    user_id: Optional[str] = None

class ChatRequest(BaseModel):
    prediction_id: str
    message: str
    user_id: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    gemini_analysis: str
    prediction_id: str

class ChatResponse(BaseModel):
    response: str
    chat_id: str

@app.get("/")
async def root():
    return {"message": "Fake News Detector API is running!"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_news(request: NewsRequest):
    try:
        if not model or not tokenizer:
            raise HTTPException(status_code=500, detail="Model not loaded properly")
        
        # Preprocess text
        text = clean_text(request.text)
        
        # Tokenize and predict
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probabilities = torch.softmax(outputs.logits, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)
        
        result = "REAL" if prediction.item() == 1 else "FAKE"
        confidence_score = confidence.item()
        
        # Get Gemini analysis
        gemini_analysis = get_gemini_analysis(text, result, confidence_score)
        
        # Save to database
        prediction_id = "demo-id"  # Default untuk demo
        
        if supabase:
            prediction_data = {
                "user_id": request.user_id or "demo-user",
                "news_text": request.text,
                "prediction_result": result,
                "confidence": float(confidence_score),
                "gemini_analysis": gemini_analysis
            }
            
            response = supabase.table("prediction_history").insert(prediction_data).execute()
            if response.data:
                prediction_id = response.data[0]["id"]
        
        return PredictionResponse(
            prediction=result,
            confidence=float(confidence_score),
            gemini_analysis=gemini_analysis,
            prediction_id=prediction_id
        )
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_gemini(request: ChatRequest):
    try:
        if not gemini_model:
            raise HTTPException(status_code=500, detail="Gemini not configured")
        
        # For demo purposes, create a simple response
        context = f"""
        User asked: {request.message}
        About a news prediction analysis.
        """
        
        response = gemini_model.generate_content(context)
        
        chat_id = "demo-chat-id"
        if supabase:
            chat_data = {
                "prediction_id": request.prediction_id,
                "user_message": request.message,
                "gemini_response": response.text
            }
            
            chat_response = supabase.table("chat_history").insert(chat_data).execute()
            if chat_response.data:
                chat_id = chat_response.data[0]["id"]
        
        return ChatResponse(response=response.text, chat_id=chat_id)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{user_id}")
async def get_prediction_history(user_id: str):
    try:
        # For demo, return empty array
        if not supabase:
            return []
            
        response = supabase.table("prediction_history")\
            .select("*, chat_history(*)")\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .execute()
        
        return response.data if response.data else []
    except Exception as e:
        print(f"Error getting history: {e}")
        return []

@app.delete("/history/{prediction_id}")
async def delete_prediction(prediction_id: str, user_id: str):
    try:
        if supabase:
            supabase.table("prediction_history")\
                .delete()\
                .eq("id", prediction_id)\
                .execute()
        
        return {"message": "Prediction deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def clean_text(text):
    """Clean and preprocess text"""
    if not text:
        return ""
    
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_gemini_analysis(text, prediction, confidence):
    """Get detailed analysis from Gemini"""
    if not gemini_model:
        return "Gemini analysis unavailable in demo mode"
    
    prompt = f"""
    Analyze this news text and explain why it was classified as {prediction} with {confidence:.2f} confidence.
    
    News Text: {text[:1000]}...
    
    Provide analysis in the following format:
    
    1. **Classification**: {prediction} ({confidence:.2%} confidence)
    2. **Key Indicators**: 
    3. **Linguistic Analysis**:
    4. **Structural Analysis**:
    5. **Recommendations**:
    
    Keep the analysis concise but informative.
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Analysis unavailable: {str(e)}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)