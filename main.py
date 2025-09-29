from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import json, os
import re
import torch
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ========= Config via ENV =========
MODEL_PATH = os.getenv("MODEL_PATH", "fake_news_detector_distilbert")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBEG-U0N3H-39A8rpZyHh4zhqykz2sMZI4")
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://blnehwafvmysyyudcglg.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJsbmVod2Fmdm15c3l5dWRjZ2xnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg5NzgwNDEsImV4cCI6MjA3NDU1NDA0MX0.MJPfcA0iBSw5iHNrCmpdjjVGIUv5O4a4R9L9nCyw4Sw")

# ========= Optional: Supabase client =========
supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase connected")
    except Exception as e:
        print(f"⚠️ Supabase disabled: {e}")
        supabase = None

# ========= Optional: Gemini client =========
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        print("✅ Gemini ready")
    except Exception as e:
        print(f"⚠️ Gemini disabled: {e}")
        gemini_model = None

# ========= Load DistilBERT =========
tokenizer, model = None, None
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✅ Model loaded: {MODEL_PATH} on {device}")
except Exception as e:
    print(f"❌ Error loading model '{MODEL_PATH}': {e}")

# ========= FastAPI app =========
app = FastAPI(title="Fake News Detector API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= Schemas =========
class PredictIn(BaseModel):
    text: str

class PredictOut(BaseModel):
    label: str
    score: float
    model_version: str
    timestamp: str

class AnalyzeIn(BaseModel):
    text: str
    prediction: Dict[str, Any]

class Factor(BaseModel):
    name: str
    weight: float
    explanation: Optional[str] = None

class Chart(BaseModel):
    labels: List[str]
    weights: List[float]

class AnalyzeOut(BaseModel):
    summary: str
    factors: List[Factor]
    chart: Chart
    tips: Optional[List[str]] = None

class ChatIn(BaseModel):
    session_id: Optional[str] = None
    message: str
    context: Optional[Dict[str, Any]] = None

class ChatOut(BaseModel):
    reply: str

# ========= Utils =========
def clean_text(text: str) -> str:
    if not text:
        return ""
    t = str(text)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def softmax_logits_to_label(logits) -> Dict[str, Any]:
    probs = torch.softmax(logits, dim=1)
    conf, pred_idx = torch.max(probs, dim=1)
    id2label = getattr(model.config, "id2label", None)
    if id2label:
        raw_label = id2label[pred_idx.item()].upper()
        label = "REAL" if "REAL" in raw_label else "FAKE"
    else:
        label = "REAL" if pred_idx.item() == 1 else "FAKE"
    return {"label": label, "score": float(conf.item())}

def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def strip_code_fences(s: str) -> str:
    if not s:
        return ""
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
    return m.group(1).strip() if m else s.strip()

def extract_json_from_text(s: str):
    if not s:
        return None
    s = strip_code_fences(s)
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def normalize_weights(factors):
    cleaned = []
    for f in factors:
        name = str(f.get("name","")).strip() or "Factor"
        try:
            w = float(f.get("weight", 0.0))
        except Exception:
            w = 0.0
        if w < 0: w = 0.0
        if w > 1.0: w = w/100 if w > 1.5 else 1.0
        cleaned.append({"name": name, "weight": w, "explanation": f.get("explanation")})
    s = sum(x["weight"] for x in cleaned)
    if s > 1.0001:
        cleaned = [{**x, "weight": (x["weight"]/s) if s else 0.0} for x in cleaned]
    return cleaned[:8]

# ========= Routes =========
@app.get("/health")
def health():
    return {"ok": True, "model_loaded": model is not None, "gemini": gemini_model is not None}

@app.get("/api")
def root():
    return {"message": "Fake News Detector API running. See /api/* endpoints."}

@app.post("/api/predict", response_model=PredictOut)
def api_predict(payload: PredictIn):
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model not loaded")

    text = clean_text(payload.text)
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")

    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)

    y = softmax_logits_to_label(out.logits)
    resp = PredictOut(
        label=y["label"],
        score=y["score"],
        model_version=getattr(model.config, "name_or_path", "distilbert-finetuned"),
        timestamp=now_iso(),
    )

    # Optional Supabase logging
    if supabase:
        try:
            supabase.table("predictions").insert({
                "text": payload.text,
                "label": resp.label,
                "score": resp.score,
                "model_version": resp.model_version,
            }).execute()
        except Exception as e:
            print(f"Supabase log predict failed: {e}")

    return resp

@app.post("/api/analyze", response_model=AnalyzeOut)
def api_analyze(payload: AnalyzeIn):
    try:
        text = clean_text(payload.text or "")
        pred = payload.prediction or {}
        pred_label = str(pred.get("label", "UNKNOWN")).upper()
        pred_score = float(pred.get("score") or 0.0)

        parsed = None
        summary_text = ""

        if gemini_model:
            try:
                prompt = f"""
Anda adalah analis misinformasi yang membantu. 
1) Pertama tulis penjelasan singkat dan conversational (2-5 kalimat) dalam bahasa Indonesia tentang MENGAPA teks diklasifikasikan sebagai {pred_label} (confidence {pred_score:.2f}). 
2) Kemudian output blok JSON kompak HANYA untuk faktor (tanpa prosa di sekitarnya).

FORMAT KETAT:
<summary>
...penjelasan bahasa biasa Anda di sini...
</summary>
<json>
{{"factors":[{{"name":"...","weight":0.42,"explanation":"..."}}, ...]}}
</json>

TEKS (dipotong hingga 1200 karakter):
{text[:1200]}
"""
                res = gemini_model.generate_content(prompt)
                content = (res.text or "").strip()

                # Extract summary
                m = re.search(r"<summary>\s*([\s\S]*?)\s*</summary>", content, re.IGNORECASE)
                summary_text = (m.group(1).strip() if m else "").strip()

                # Extract JSON
                m2 = re.search(r"<json>\s*([\s\S]*?)\s*</json>", content, re.IGNORECASE)
                json_blob = m2.group(1).strip() if m2 else content
                parsed = extract_json_from_text(json_blob)

            except Exception as e:
                print(f"[Analyze] Gemini error: {e}")
                parsed, summary_text = None, ""

        if not parsed:
            # Fallback heuristic
            cues = [
                {"name":"Banyak huruf kapital/seruan", "weight":0.25, "explanation":"Gaya hiperbolik/sensasional."},
                {"name":"Sumber tidak jelas", "weight":0.30, "explanation":"Tidak ada sumber kredibel yang dikutip."},
                {"name":"Klaim sulit diverifikasi", "weight":0.25, "explanation":"Kurang detail yang dapat dicek."},
                {"name":"Inkonsistensi struktur", "weight":0.20, "explanation":"Masalah logika/timeline."},
            ]
            parsed = {"factors": cues}
            if not summary_text:
                summary_text = f"Klasifikasi {pred_label} (conf {pred_score:.2f}). Ini ringkasan heuristik saat LLM tidak tersedia."

        # Clean & normalize factors
        factors_clean = normalize_weights(parsed.get("factors", []))
        factors = [Factor(**f) for f in factors_clean]
        labels = [f.name for f in factors]
        weights = [float(f.weight) for f in factors]

        if not summary_text:
            summary_text = f"Teks diklasifikasi sebagai {pred_label} (confidence {pred_score:.2f}). Faktor kunci terlihat pada pola bahasa, keterverifikasian klaim, dan kredibilitas sumber."

        return AnalyzeOut(
            summary=summary_text,
            factors=factors,
            chart=Chart(labels=labels, weights=weights),
            tips=["Cek sumber primer", "Bandingkan ke media arus utama", "Waspadai framing sensasional"],
        )

    except Exception as e:
        print(f"[Analyze] Fatal fallback: {e}")
        return AnalyzeOut(
            summary="Analisis fallback: sistem bermasalah, tampilkan heuristik minimal.",
            factors=[Factor(name="Fallback", weight=0.5, explanation="Handler error, gunakan verifikasi manual.")],
            chart=Chart(labels=["Fallback"], weights=[0.5]),
            tips=["Coba ulang beberapa saat lagi"],
        )

@app.post("/api/chat", response_model=ChatOut)
def api_chat(payload: ChatIn):
    user_msg = (payload.message or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Empty message")

    ctx = payload.context or {}
    pred = ctx.get("prediction", {})
    analysis = ctx.get("analysis", {})
    base_summary = analysis.get("summary", "") if analysis else ""
    text = ctx.get("text", "")

    if gemini_model:
        try:
            prompt = f"""
Anda asisten fact-checking. Jawab dengan ringkas & natural (Bahasa Indonesia).
Konteks klasifikasi: {pred}
Ringkasan analisis: {base_summary}
Teks asli: {text[:500]}

Pertanyaan pengguna:
{user_msg}

Jawab dengan jujur. Jika tidak yakin, berikan langkah verifikasi praktis & saran sumber kredibel.
"""
            res = gemini_model.generate_content(prompt)
            reply = (res.text or "").strip() or "Maaf, belum ada jawaban."
        except Exception as e:
            print(f"[Chat] Gemini error: {e}")
            reply = "Aku lagi tidak bisa akses LLM. Intinya: " + (base_summary[:200] or "Coba ulang beberapa saat lagi.")
    else:
        reply = "LLM belum aktif di server. Tapi dari analisis: " + (base_summary[:200] or "Belum ada ringkasan.")

    # Optional log
    if supabase and payload.session_id:
        try:
            supabase.table("chats").insert({
                "session_id": payload.session_id,
                "role": "assistant",
                "message": reply
            }).execute()
        except Exception as e:
            print(f"Supabase log chat failed: {e}")

    return ChatOut(reply=reply)

# Serve frontend static files
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    print(f"✅ Serving frontend from {FRONTEND_DIR}")
else:
    print(f"⚠️ Frontend dir not found: {FRONTEND_DIR}. Serving API-only")

# ========= Entrypoint =========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="82.197.71.171", port=4000, reload=False)
