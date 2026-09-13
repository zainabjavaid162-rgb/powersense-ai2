# ⚡ PowerSense AI

An intelligent electricity-bill assistant with **real RAG (Retrieval-Augmented Generation)**:
upload your electricity bill (or any related PDF), and PowerSense AI indexes it, answers
questions grounded in it, and generates a **personalized, AI-written savings plan** — powered
by [Groq](https://groq.com)'s fast LLM inference.

Two interfaces are included, sharing the same core idea:

| File            | Framework | Best for                                      |
|------------------|-----------|------------------------------------------------|
| `app.py`         | Streamlit | Full dashboard: KPIs, charts, bill analyzer, anomaly detection, savings simulator |
| `gradio_app.py`  | Gradio    | Lightweight, chat-first interface — quick to deploy on Hugging Face Spaces |

---

## ✨ Features

- **📄 Real PDF ingestion** — upload your electricity bill (or a tariff notice, FAQ, etc.).
  Text is extracted with `pypdf`, chunked, and indexed with TF-IDF (`scikit-learn`) — a real
  retrieval index, not a hardcoded lookup.
- **🤖 RAG-grounded AI Assistant** — ask questions in plain language. Retrieved chunks (from
  the built-in knowledge base **and** anything you've uploaded) are passed to Groq's LLM as
  context, so answers are grounded in real content. Without an API key, you still get honest
  retrieval-only answers assembled directly from the indexed text.
- **💡 AI-generated savings suggestions** — a personalized plan (not a generic list) built from
  your actual bill data, a calculated baseline, and RAG context from your own documents.
- **🧾 Bill Analyzer** *(Streamlit)* — extracts consumer ID, units, readings, due date, and
  total from an uploaded PDF via regex; missing fields fall back to clearly-labeled demo data.
- **📊 Consumption analytics, anomaly detection, and a savings simulator** *(Streamlit)*.
- **🎨 Professional UI** — custom theming on both interfaces (indigo/emerald palette, clean
  cards, KPI tiles).
- **Honest about what's real** — see the header comment in `app.py` for exactly what's live
  computation vs. simplified/simulated demo data (e.g. the 12-month consumption history is
  synthetic; there's no OCR for scanned/image bills yet).

---

## 🗂 Project structure

```
powersense-ai/
├── app.py                        # Streamlit app (main dashboard)
├── gradio_app.py                 # Gradio app (chat-first alternative UI)
├── requirements.txt
├── .env.example                  # Template for local env vars
├── .streamlit/
│   ├── config.toml               # Theme (already configured, no action needed)
│   └── secrets.toml.example      # Template for Streamlit Cloud secrets
├── .gitignore
└── README.md
```

---

## 🚀 Quickstart (local)

```bash
# 1. Clone your repo
git clone https://github.com/<your-username>/powersense-ai.git
cd powersense-ai

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Groq API key (free key: https://console.groq.com/keys)
cp .env.example .env
# then edit .env and paste your key, OR export it directly:
export GROQ_API_KEY=your_groq_api_key_here     # Windows: set GROQ_API_KEY=...

# 5a. Run the Streamlit app
streamlit run app.py

# 5b. Or run the Gradio app
python gradio_app.py
```

You don't strictly need a Groq key to run either app — both work in **retrieval-only mode**
without one, and you can also paste a key directly into the app's Settings panel/Accordion at
runtime (it's kept in session memory only, never written to disk).

---

## ☁️ Deploying

### Streamlit Community Cloud (`app.py`)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick your repo,
   branch, and set the main file to `app.py`.
3. In **Advanced settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
   (This is exactly the content of `.streamlit/secrets.toml.example`.)
4. Deploy. The app reads the key automatically — no code changes needed.

### Hugging Face Spaces (`gradio_app.py`)

1. Create a new Space → SDK: **Gradio**.
2. Push this repo's contents (or just `gradio_app.py` + `requirements.txt`) to the Space repo.
   Rename `gradio_app.py` to `app.py` inside the Space, or set the Space's app file to
   `gradio_app.py` in the Space settings.
3. In **Settings → Repository secrets**, add `GROQ_API_KEY` with your key.
4. The Space builds and launches automatically.

### GitHub (source control for both)

```bash
git init
git add .
git commit -m "Initial commit: PowerSense AI (Streamlit + Gradio, Groq-powered RAG)"
git branch -M main
git remote add origin https://github.com/<your-username>/powersense-ai.git
git push -u origin main
```

`.gitignore` already excludes `.env` and `.streamlit/secrets.toml` so your real key never
gets committed.

---

## 🔑 Getting a Groq API key

1. Go to [console.groq.com/keys](https://console.groq.com/keys) and sign up (free tier
   available).
2. Create a new API key (starts with `gsk_...`).
3. Use it via `.env`, Streamlit secrets, an exported environment variable, or paste it
   directly into either app's Settings UI.

Default model: `llama-3.3-70b-versatile`. Other options you can type into the Model field:
`llama-3.1-8b-instant` (faster/cheaper), `gemma2-9b-it`. See the full list at
[console.groq.com/docs/models](https://console.groq.com/docs/models).

---

## 🧠 How the RAG pipeline works

1. **Chunking** — uploaded PDF text and built-in knowledge-base articles are split into small,
   sentence-grouped chunks.
2. **Indexing** — chunks are vectorized with a TF-IDF vectorizer (unigrams + bigrams,
   English stop words removed).
3. **Retrieval** — a user's question is vectorized the same way, and the top-*k* chunks by
   cosine similarity above a minimum score are retrieved (below the threshold → legitimately
   zero sources, no forced answer).
4. **Generation** — retrieved chunks + your bill data are sent to Groq's chat completion API
   as context, with a system prompt instructing the model to answer only from that context.
5. **Fallback** — without a Groq key (or if the API call fails), the top retrieved chunk(s)
   are shown directly, so the app is still useful and never silently fabricates an LLM answer.

---

## ⚠️ Known limitations

- No OCR — scanned/image-only PDF bills aren't parsed, only text-based PDFs.
- The 12-month consumption history (Streamlit dashboard) is synthetic demo data, labeled as
  such in the UI.
- The tariff model is a single flat rate per unit, not a full multi-slab schedule.
- TF-IDF retrieval is lexical (keyword-based), not a neural embedding search — good for a
  lightweight, dependency-light demo, but less semantically flexible than an embeddings-based
  RAG pipeline.

---

## 📄 License

Use, modify, and deploy freely for your own project.
