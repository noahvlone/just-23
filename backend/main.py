from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import json, os
import re
import torch
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ========= Config via ENV (JANGAN hardcode kunci di repo publik) =========
MODEL_PATH = os.getenv("MODEL_PATH", "fake_news_detector_distilbert")  # local folder or HF id
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "ISI_API_KAMU")
SUPABASE_URL = os.getenv("SUPABASE_URL", "ISI_URL_KAMU")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "ISI_KEY_KAMU")

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
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.5-pro")
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

# CORS: frontend bisa di-serve same-origin. Untuk aman, allow all.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # kalau mau spesifik, isi origin frontend lo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= Schemas (match frontend) =========
class PredictIn(BaseModel):
    text: str

class PredictOut(BaseModel):
    label: str                # "FAKE" | "REAL"
    score: float              # 0..1
    model_version: str        # e.g. "distilbert-finetuned"
    timestamp: str            # ISO time

class AnalyzeIn(BaseModel):
    text: str
    prediction: Dict[str, Any]  # { label, score }

class Factor(BaseModel):
    name: str
    weight: float               # 0..1
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
    # lebih soft: jangan buang semua tanda baca, biar konteks tetap kaya
    t = re.sub(r"\s+", " ", t).strip()
    return t

def softmax_logits_to_label(logits) -> Dict[str, Any]:
    probs = torch.softmax(logits, dim=1)
    conf, pred_idx = torch.max(probs, dim=1)
    # Asumsi: id2label = {0: 'FAKE', 1: 'REAL'} (atau kebalikan)
    id2label = getattr(model.config, "id2label", None)
    if id2label:
        raw_label = id2label[pred_idx.item()].upper()
        label = "REAL" if "REAL" in raw_label else "FAKE"
    else:
        # fallback: index 1 = REAL, 0 = FAKE
        label = "REAL" if pred_idx.item() == 1 else "FAKE"
    return {"label": label, "score": float(conf.item())}

def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def strip_code_fences(s: str) -> str:
    if not s:
        return ""
    # hapus ```json ... ``` atau ``` ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
    return m.group(1).strip() if m else s.strip()

def extract_json_from_text(s: str):
    """Coba ambil JSON object pertama dari string (walau ada text lain)."""
    if not s:
        return None
    s = strip_code_fences(s)
    # cari blok {...}
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def normalize_weights(factors):
    """Pastikan weight float [0..1]; kalau >1 atau sum>1, normalisasi."""
    cleaned = []
    for f in factors:
        name = str(f.get("name","")).strip() or "Factor"
        try:
            w = float(f.get("weight", 0.0))
        except Exception:
            w = 0.0
        # clamp
        if w < 0: w = 0.0
        if w > 1.0: w = w/100 if w > 1.5 else 1.0
        cleaned.append({"name": name, "weight": w, "explanation": f.get("explanation")})
    s = sum(x["weight"] for x in cleaned)
    if s > 1.0001:
        cleaned = [{**x, "weight": (x["weight"]/s) if s else 0.0} for x in cleaned]
    return cleaned[:8]  # batasi 8 faktor

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

    # Backend logging ke Supabase (opsional) – frontend juga sudah simpan
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
        pred = (payload.prediction or {})
        pred_label = str(pred.get("label", "UNKNOWN")).upper()
        pred_score = float(pred.get("score") or 0.0)

        parsed = None
        summary_text = ""

        if gemini_model:
            try:
                prompt = f"""
You are a helpful misinformation analyst. 
1) First write a short, conversational explanation (2–5 sentences) in Indonesian about WHY the text was classified as {pred_label} (confidence {pred_score:.2f}). 
2) Then output a compact JSON block ONLY for factors (no prose around it).

FORMAT STRICT:
<summary>
...your plain-language explanation here...
</summary>
<json>
{{"factors":[{{"name":"...","weight":0.42,"explanation":"..."}}, ...]}}
</json>

TEXT (truncated to 1200 chars):
{text[:1200]}
"""
                res = gemini_model.generate_content(prompt)
                content = (res.text or "").strip()

                # Ambil <summary>...</summary>
                m = re.search(r"<summary>\s*([\s\S]*?)\s*</summary>", content, re.IGNORECASE)
                summary_text = (m.group(1).strip() if m else "").strip()

                # Ambil <json>...</json> lalu parse JSON-nya
                m2 = re.search(r"<json>\s*([\s\S]*?)\s*</json>", content, re.IGNORECASE)
                json_blob = m2.group(1).strip() if m2 else content
                parsed = extract_json_from_text(json_blob)

            except Exception as e:
                print(f"[Analyze] Gemini error: {e}")
                parsed, summary_text = None, ""

        if not parsed:
            # fallback heuristik
            cues = [
                {"name":"Banyak huruf kapital/seruan", "weight":0.25, "explanation":"Hyperbolic/sensational style."},
                {"name":"Sumber tidak jelas",          "weight":0.30, "explanation":"No credible source cited."},
                {"name":"Klaim sulit diverifikasi",    "weight":0.25, "explanation":"Lack of checkable details."},
                {"name":"Inkonsistensi struktur",      "weight":0.20, "explanation":"Logic/timeline issues."},
            ]
            parsed = {"factors": cues}
            if not summary_text:
                summary_text = f"Klasifikasi {pred_label} (conf {pred_score:.2f}). Ini ringkasan heuristik saat LLM tidak tersedia."

        # Bersihkan & normalisasi faktor
        factors_clean = normalize_weights(parsed.get("factors", []))
        factors = [Factor(**f) for f in factors_clean]
        labels = [f.name for f in factors]
        weights = [float(f.weight) for f in factors]

        # Kalau summary kosong, isi default yang enak dibaca
        if not summary_text:
            summary_text = f"Teks diklasifikasi sebagai {pred_label} (confidence {pred_score:.2f}). Faktor kunci terlihat pada pola bahasa, keterverifikasian klaim, dan kredibilitas sumber."

        return AnalyzeOut(
            summary=summary_text,                    # <-- teks ngobrol
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
    base_summary = (ctx.get("analysis", {}) or {})/get("summary", "")

    if gemini_model:
        try:
            prompt = f"""
Kamu asisten fact-checking. Jawab ringkas & natural (Bahasa Indonesia).
Konteks klasifikasi: {pred}
Ringkasan analisis: {base_summary}

Pertanyaan pengguna:
{user_msg}

Jawab jujur. Jika tidak yakin, beri langkah verifikasi praktis & saran sumber kredibel.
"""
            res = gemini_model.generate_content(prompt)
            reply = (res.text or "").strip() or "Maaf, belum ada jawaban."
        except Exception as e:
            print(f"[Chat] Gemini error: {e}")
            reply = "Aku lagi nggak bisa akses LLM. Intinya: " + (base_summary[:200] or "Coba ulang beberapa saat lagi.")
    else:
        reply = "LLM belum aktif di server. Tapi dari analisis: " + (base_summary[:200] or "Belum ada ringkasan.")

    # Optional log
    if supabase:
        try:
            supabase.table("chats").insert({
                "session_id": payload.session_id,
                "role": "assistant",
                "message": reply
            }).execute()
        except Exception as e:
            print(f"Supabase log chat failed: {e}")

    return ChatOut(reply=reply)


from fastapi.staticfiles import StaticFiles
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    print(f"⚠️ Frontend dir not found: {FRONTEND_DIR}. Serving API-only at /api/*")

# ========= Entrypoint =========
if __name__ == "__main__":
    import uvicorn
    # listen di server 82.197.71.171:4000
    uvicorn.run("main:app", host="0.0.0.0", port=4000, reload=False)

