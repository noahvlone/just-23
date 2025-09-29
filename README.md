# 🧠 Fake News Detector

Sistem deteksi berita palsu yang menggabungkan DistilBERT untuk prediksi cepat dan Google Gemini untuk analisis mendalam, dengan Supabase untuk penyimpanan data.

## ✨ Fitur

- **⚡ Prediksi Cepat**: Model DistilBERT untuk klasifikasi FAKE/REAL
- **🧠 Analisis Mendalam**: AI Gemini untuk analisis faktor dan penjelasan
- **💬 Chat Interaktif**: Tanya jawab tentang analisis
- **📊 Visualisasi Data**: Chart.js untuk visualisasi faktor deteksi
- **💾 Penyimpanan Data**: Integrasi Supabase untuk riwayat sesi
- **🎨 UI Modern**: Desain responsif dengan dukungan tema gelap/terang
- **📱 Mobile Friendly**: Berjalan lancar di desktop dan mobile

## 🏗️ Arsitektur

```
Frontend (HTML/CSS/JS) → FastAPI Backend → AI Models → Supabase Database
     │                        │               │              │
     ├── Chart.js             ├── DistilBERT  ├── Gemini     ├── Sessions
     ├── Supabase Client      ├── CORS        └── Analysis   ├── Predictions
     └── Responsive UI        └── Static Serving            └── Chats
```

## 🚀 Panduan Instalasi Cepat

### Prerequisites

- Python 3.8+
- Akun Supabase
- Google AI Studio API key

### 1. Setup Backend

```bash
# Buat virtual environment
python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn transformers torch google-generativeai supabase python-multipart python-dotenv
```

### 2. Konfigurasi Environment

Buat file `.env`:

```env
MODEL_PATH=fake_news_detector_distilbert
GEMINI_API_KEY=your_gemini_api_key_here
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_anon_key_here
```

### 3. Setup Database

Jalankan SQL ini di Supabase SQL editor:

```sql
-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Sessions table
CREATE TABLE sessions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    title TEXT,
    source_text TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW())
);

-- Predictions table
CREATE TABLE predictions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    text TEXT,
    label TEXT CHECK (label IN ('FAKE', 'REAL')),
    score FLOAT,
    model_version TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW())
);

-- Analyses table
CREATE TABLE analyses (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    summary TEXT,
    factors JSONB,
    chart JSONB,
    tips TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW())
);

-- Chats table
CREATE TABLE chats (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT CHECK (role IN ('user', 'assistant')),
    message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW())
);

-- Create indexes
CREATE INDEX idx_sessions_created_at ON sessions(created_at);
CREATE INDEX idx_predictions_session_id ON predictions(session_id);
CREATE INDEX idx_analyses_session_id ON analyses(session_id);
CREATE INDEX idx_chats_session_id ON chats(session_id);
```

### 4. Setup Frontend

```bash
# Buat folder frontend
mkdir -p frontend

# Simpan index.html di folder frontend
cp index.html frontend/
```

### 5. Jalankan Aplikasi

```bash
# Start server backend
python main.py
```

Aplikasi akan tersedia di:
- Frontend: http://localhost:4000
- API Docs: http://localhost:4000/docs

## 📁 Struktur Proyek

```
fake-news-detector/
├── main.py                 # FastAPI backend
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables
├── README.md              # File ini
└── frontend/
    └── index.html         # Aplikasi frontend
```

## 🛠️ API Endpoints

### Health Check
- `GET /health` - Status service

### Prediction
- `POST /api/predict` - Klasifikasi berita sebagai FAKE/REAL
```json
{
  "text": "Konten berita di sini..."
}
```

### Analysis
- `POST /api/analyze` - Dapatkan analisis detail dengan faktor
```json
{
  "text": "Konten berita...",
  "prediction": {
    "label": "FAKE",
    "score": 0.85
  }
}
```

### Chat
- `POST /api/chat` - Tanya jawab interaktif tentang analisis
```json
{
  "message": "Mengapa ini diklasifikasikan sebagai fake?",
  "context": {
    "prediction": {...},
    "analysis": {...}
  }
}
```

## 🎯 Panduan Penggunaan

1. **Input Teks Berita**: Paste atau ketik konten berita di text area
2. **Dapatkan Prediksi**: Klik "Prediksi" untuk klasifikasi DistilBERT
3. **Analisis Mendalam**: Klik "Analisis" untuk analisis faktor oleh Gemini
4. **Tanya Pertanyaan**: Gunakan chat untuk mendapatkan insight lebih
5. **Review Riwayat**: Akses sesi sebelumnya dari sidebar

## 🔧 Konfigurasi

### Model Configuration
- **DistilBERT Model**: Set `MODEL_PATH` ke model fine-tuned Anda
- **Gemini Model**: Menggunakan `gemini-2.0-flash` untuk respons cepat

### Supabase Configuration
- Update `SUPABASE_URL` dan `SUPABASE_KEY` di environment
- Pastikan kebijakan RLS dikonfigurasi jika diperlukan

## 🐛 Troubleshooting

### Masalah Umum

1. **Model Gagal Load**
   - Cek `MODEL_PATH`是否存在
   - Verifikasi instalasi transformers

2. **Error Gemini API**
   - Validasi API key di Google AI Studio
   - Cek batas quota

3. **Koneksi Supabase Gagal**
   - Verifikasi tabel database ada
   - Cek kredensial koneksi

4. **Error CORS**
   - Pastikan backend mengizinkan origin frontend
   - Cek console browser untuk error

### Debug Mode

Aktifkan debug logging dengan environment variable:
```bash
export LOG_LEVEL=DEBUG
```

## 🚀 Deployment

### Production Deployment

1. **Backend**: Deploy ke services seperti:
   - AWS EC2/Lambda
   - Google Cloud Run
   - Heroku
   - DigitalOcean App Platform

2. **Frontend**: Serve via:
   - Nginx
   - Vercel
   - Netlify

3. **Database**: Gunakan instance Supabase production

### Environment Variables untuk Production

```env
MODEL_PATH=your/production/model
GEMINI_API_KEY=your_production_gemini_key
SUPABASE_URL=your_production_supabase_url
SUPABASE_KEY=your_production_supabase_key
LOG_LEVEL=INFO
```

## 📝 Lisensi

Proyek ini menggunakan lisensi MIT - lihat file [LICENSE](LICENSE) untuk detail.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co) untuk library Transformers
- [Google AI](https://ai.google.dev) untuk model Gemini
- [Supabase](https://supabase.com) untuk layanan backend
- [FastAPI](https://fastapi.tiangolo.com) untuk web framework
- [Chart.js](https://chartjs.org) untuk visualisasi data

---

**Catatan**: Proyek ini untuk tujuan edukasi dan penelitian. Selalu verifikasi informasi melalui berbagai sumber kredibel.
```
