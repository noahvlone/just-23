# 📰 Fake News Detector (DistilBERT + Gemini)

Proyek ini adalah sistem deteksi berita palsu berbasis **Deep Learning** (DistilBERT) yang dikombinasikan dengan **Large Language Model (LLM) Gemini** untuk memberikan analisis lanjutan dan alasan klasifikasi.  
Aplikasi sudah terintegrasi dengan **Supabase** untuk menyimpan riwayat prediksi, analisis, dan chat.  
Frontend dibangun dengan **HTML/CSS/JavaScript**, Backend dengan **FastAPI**, dan deployment via **Docker**.

---

## ✨ Fitur
- **Deteksi Berita Palsu** menggunakan DistilBERT yang di-finetune.
- **Analisis Lanjutan dengan LLM (Gemini)** → memberikan ringkasan, alasan klasifikasi, dan insight tambahan.
- **Tanya LLM** → user bisa ngobrol/chat dengan AI soal hasil prediksi.
- **Riwayat (History)** → semua sesi, prediksi, analisis, dan chat disimpan di **Supabase (PostgreSQL)**.
- **Visualisasi** → confidence score, faktor LLM, chart bobot, dan riwayat analisis.
- **Frontend Interaktif** → dark/light mode, dashboard responsif, history panel.
- **Deployable** → dengan **Docker** jadi bisa jalan di server tanpa buka terminal terus.

---

## 🛠️ Tech Stack
- **Backend**: FastAPI + Uvicorn
- **Model**: HuggingFace DistilBERT (local fine-tuned model)
- **LLM**: Google Gemini API
- **Database**: Supabase (PostgreSQL)
- **Frontend**: HTML + CSS + Vanilla JS + Chart.js
- **Deployment**: Docker

---

## 📂 Struktur Project


├── backend/                 
├── frontend/              
├── fake_news_detector_distilbert/ 
├── requirements.txt         
├── Dockerfile               
├── README.md



---

## ⚡ Cara Jalanin di Lokal

### 1. Clone repo
```bash
git clone https://github.com/your-username/fnd-app.git
cd fnd-app
````

### 2. Install dependency

```bash
pip install -r requirements.txt
```

### 3. Setup `.env`

Buat file `.env`:

```env
GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-1.5-flash
SUPABASE_URL=https://yourproject.supabase.co
SUPABASE_KEY=your_supabase_anon_key
```

### 4. Jalankan backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 4000
```

### 5. Akses frontend

Buka `http://localhost:4000` → otomatis serve `frontend/index.html`.

---

## 🐳 Deployment dengan Docker

### Build image

```bash
docker build -t fnd-app:latest .
```

### Run container (port 4000)

```bash
docker run -d --name fnd \
  --restart unless-stopped \
  --env-file .env \
  -p 4000:4000 \
  fnd-app:latest \
  uvicorn backend.main:app --host 0.0.0.0 --port 4000
```

Akses via:
👉 `http://your-server-ip:4000`

---

## 📊 Supabase Schema

```sql
create table sessions (
  id uuid primary key default gen_random_uuid(),
  created_at timestamp with time zone default now(),
  title text,
  source_text text
);

create table predictions (
  id uuid primary key default gen_random_uuid(),
  created_at timestamp with time zone default now(),
  session_id uuid references sessions(id) on delete cascade,
  text text,
  label text,
  score float8,
  model_version text
);

create table analyses (
  id uuid primary key default gen_random_uuid(),
  created_at timestamp with time zone default now(),
  session_id uuid references sessions(id) on delete cascade,
  summary text,
  factors jsonb,
  chart jsonb,
  tips jsonb
);

create table chats (
  id uuid primary key default gen_random_uuid(),
  created_at timestamp with time zone default now(),
  session_id uuid references sessions(id) on delete cascade,
  role text check (role in ('user','assistant')),
  message text
);
```

---

## 📸 Screenshots

* Input berita → Prediksi DistilBERT
* Analisis lanjutan Gemini
* Chat interaktif dengan LLM
* Riwayat & visualisasi faktor klasifikasi

---

## 🚀 Roadmap

* [x] Integrasi DistilBERT
* [x] Analisis dengan Gemini
* [x] Supabase untuk riwayat
* [x] Chat interaktif LLM
* [x] Docker deployment
* [ ] Tambah opsi multi-model (RoBERTa, BERT)
* [ ] Integrasi autentikasi user
* [ ] Deploy HTTPS dengan reverse proxy (Caddy/Nginx)

---

## 👨‍💻 Kontributor

* **Farhan Ramadhan** (Lead Dev / Data Scientist)
* Open for collaboration! 🚀

---

