# ⚡ PowerSense AI

An intelligent electricity-bill assistant with **real RAG (Retrieval-Augmented Generation)**:
upload your electricity bill (or any related PDF), and PowerSense AI indexes it, answers
questions grounded in it, and generates a **personalized, AI-written savings plan** — powered
by [Groq](https://groq.com)'s fast LLM inference. Built with **Streamlit**.

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
- **🧾 Bill Analyzer** — extracts consumer ID, units, readings, due date, and total from an
  uploaded PDF via regex; missing fields fall back to clearly-labeled demo data.
- **📊 Consumption analytics, anomaly detection, and a savings simulator.**
- **🎨 Professional UI** — custom theming (indigo/emerald palette, clean cards, KPI tiles).
- **Honest about what's real** — see the header comment in `app.py` for exactly what's live
  computation vs. simplified/simulated demo data (e.g. the 12-month consumption history is
  synthetic; there's no OCR for scanned/image bills yet).

---

## 🗂 Project structure

```
powersense-ai/
├── app.py                        # Streamlit app
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

# 5. Run the app
streamlit run app.py
```

You don't strictly need a Groq key to run the app — it works in **retrieval-only mode**
without one, and you can also paste a key directly into the app's Settings page at runtime
(it's kept in session memory only, never written to disk).

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick your repo,
   branch, and set the main file to `app.py`.
3. **Set the Python version explicitly.** In the deploy dialog (or later, in your app's
   **Settings → General**), pick **Python 3.11 or 3.12** rather than leaving it on the
   default. Streamlit Cloud's default Python (currently 3.14) is too new for several
   scientific-Python packages to have pre-built wheels yet, which can cause builds to fail
   or hang trying to compile from source. `runtime.txt` is **not** honored for this — the
   version must be picked in the UI.
4. In **Advanced settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
   (This is exactly the content of `.streamlit/secrets.toml.example`.)
5. Deploy. The app reads the key automatically — no code changes needed.

### GitHub (source control)

```bash
git init
git add .
git commit -m "Initial commit: PowerSense AI (Streamlit, Groq-powered RAG)"
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
   directly into the app's Settings page.

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
- The 12-month consumption history is synthetic demo data, labeled as such in the UI.
- The tariff model is a single flat rate per unit, not a full multi-slab schedule.
- TF-IDF retrieval is lexical (keyword-based), not a neural embedding search — good for a
  lightweight, dependency-light demo, but less semantically flexible than an embeddings-based
  RAG pipeline.

---

## 📄 License

Use, modify, and deploy freely for your own project.
