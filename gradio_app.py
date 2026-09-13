"""
PowerSense AI — Gradio Interface
---------------------------------
A standalone Gradio front-end for PowerSense AI, alongside the Streamlit
app (app.py). Same core idea, different UI:

  - Upload an electricity bill (or any relevant PDF, e.g. a tariff notice)
    and it is chunked + indexed with real TF-IDF retrieval (scikit-learn).
  - Ask the AI Assistant questions — answers are RAG-grounded: retrieved
    chunks are passed to Groq as context. Without a Groq key, you still
    get retrieval-only answers built directly from the indexed text.
  - Get a personalized, AI-generated savings plan based on your entered
    usage/bill numbers plus whatever's been indexed (built-in knowledge
    base + any PDFs you've uploaded).

Run with:
    pip install -r requirements.txt
    export GROQ_API_KEY=your_key_here   # optional — can also be entered in the UI
    python gradio_app.py
"""

import io
import os
import re

import gradio as gr

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

DEFAULT_MODEL = "llama-3.3-70b-versatile"

# ----------------------------------------------------------------------
# BUILT-IN KNOWLEDGE BASE (same content family as the Streamlit app)
# ----------------------------------------------------------------------

KNOWLEDGE_BASE = [
    {
        "title": "Electricity Tariff Slabs Explained",
        "text": (
            "Most residential electricity tariffs are structured in slabs, meaning the "
            "price per unit (kWh) increases as your total monthly consumption crosses "
            "certain thresholds, rather than every unit being billed at a single flat "
            "rate. This is why doubling your usage from one month to the next often more "
            "than doubles your bill — part of the extra consumption falls into a "
            "higher-priced slab."
        ),
    },
    {
        "title": "Fuel Price Adjustment (FPA)",
        "text": (
            "The Fuel Price Adjustment (FPA) is a variable surcharge added to your bill "
            "to account for month-to-month swings in generation fuel costs, separate "
            "from the base tariff rate. A month with more expensive thermal generation "
            "typically produces a higher FPA charge."
        ),
    },
    {
        "title": "Common Reasons for Sudden Bill Increases",
        "text": (
            "A sudden jump in your bill usually traces back to genuinely higher "
            "consumption (longer AC/heater runtime, a new appliance, more people at "
            "home), tariff-slab crossing (a modest unit increase pushes part of your "
            "usage into a higher-priced slab), a moving FPA surcharge, or occasionally "
            "a meter/billing error."
        ),
    },
    {
        "title": "Energy-Saving Tips for Air Conditioners",
        "text": (
            "Air conditioners are typically the single largest contributor to a "
            "household's summer electricity bill. Setting the thermostat to 24-26°C "
            "instead of 18-20°C, running on a timer, sealing doors/windows while it "
            "runs, cleaning filters regularly, and shading sun-facing windows all "
            "meaningfully cut consumption. Shifting AC-heavy hours outside peak pricing "
            "windows (often early evening) can also reduce cost."
        ),
    },
    {
        "title": "Reducing Standby and Lighting Load",
        "text": (
            "Switching remaining incandescent or CFL bulbs to LEDs cuts lighting energy "
            "use substantially for the same brightness. Unplugging chargers, routers, "
            "and appliances left on standby, and using power strips to cut multiple "
            "devices at once, removes a small but constant background draw that adds up "
            "over a full month."
        ),
    },
    {
        "title": "Water Heater and Refrigerator Efficiency",
        "text": (
            "Water heaters and refrigerators run for long hours and are worth checking "
            "for efficiency: lowering a water heater's thermostat a few degrees, "
            "insulating hot water pipes, and keeping a refrigerator's coils clean and "
            "its door seal tight all reduce their electricity draw without changing how "
            "you use them day to day."
        ),
    },
    {
        "title": "How to Read Your Electricity Meter",
        "text": (
            "For a standard digital meter, subtract the earlier cumulative kWh reading "
            "from the later reading to calculate your own usage between two dates; it "
            "should be close to the 'units consumed' figure on your bill. A large "
            "mismatch is worth reporting to the utility."
        ),
    },
]

# ----------------------------------------------------------------------
# PDF + CHUNKING + TF-IDF RAG
# ----------------------------------------------------------------------


def extract_text_from_pdf(file_path):
    if not PYPDF_AVAILABLE or not file_path:
        return ""
    try:
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def _split_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _chunk_text(text, sentences_per_chunk=3):
    sents = _split_sentences(text)
    chunks = []
    for i in range(0, len(sents), sentences_per_chunk):
        chunk = " ".join(sents[i:i + sentences_per_chunk])
        if chunk:
            chunks.append(chunk)
    return chunks


def build_index(custom_chunks):
    chunks = [{"title": a["title"], "text": a["text"]} for a in KNOWLEDGE_BASE] + custom_chunks
    if not chunks or not SKLEARN_AVAILABLE:
        return None, None, chunks
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix, chunks


def rag_search(query, index, k=4, min_score=0.0):
    vectorizer, matrix, chunks = index
    if vectorizer is None or matrix is None or not chunks or not query.strip():
        return []
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix).flatten()
    ranked = sims.argsort()[::-1][:k]
    results = []
    for i in ranked:
        if sims[i] >= min_score:
            item = dict(chunks[i])
            item["score"] = round(float(sims[i]), 3)
            results.append(item)
    return results


# ----------------------------------------------------------------------
# BILL FIELD EXTRACTION (best-effort regex, same idea as the Streamlit app)
# ----------------------------------------------------------------------

def parse_bill_text(text):
    patterns = {
        "units_consumed": r"(?:units\s*consumed|total\s*units)[:\s]*([\d,]+)",
        "total_payable": r"(?:total\s*(?:payable|amount|bill)|amount\s*due)[:\s]*(?:rs\.?)?\s*([\d,]+)",
    }
    found = {}
    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                found[key] = int(m.group(1).replace(",", ""))
            except ValueError:
                pass
    return found


# ----------------------------------------------------------------------
# GROQ CLIENT + LLM CALLS
# ----------------------------------------------------------------------

def get_groq_client(api_key):
    key = (api_key or "").strip() or os.environ.get("GROQ_API_KEY", "")
    if not key or not GROQ_AVAILABLE:
        return None
    try:
        return Groq(api_key=key)
    except Exception:
        return None


def call_groq(client, model, system_prompt, user_prompt, max_tokens=500):
    resp = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


# ----------------------------------------------------------------------
# APP LOGIC (bound to Gradio state, no globals mutated across users)
# ----------------------------------------------------------------------

def handle_pdf_upload(file, custom_chunks, log):
    if file is None:
        return custom_chunks, log, "No file uploaded yet."
    if not PYPDF_AVAILABLE:
        return custom_chunks, log, "⚠️ `pypdf` isn't installed — cannot read PDFs."
    raw_text = extract_text_from_pdf(file.name if hasattr(file, "name") else file)
    if not raw_text.strip():
        return custom_chunks, log, "⚠️ Couldn't extract text (it may be a scanned image — OCR isn't implemented)."
    fname = os.path.basename(file.name if hasattr(file, "name") else str(file))
    new_chunks = [{"title": fname, "text": c} for c in _chunk_text(raw_text)]
    custom_chunks = (custom_chunks or []) + new_chunks
    fields = parse_bill_text(raw_text)
    field_note = f" Detected fields: {fields}." if fields else " No structured bill fields detected — you can enter them manually below."
    status = f"✅ Indexed **{len(new_chunks)} chunks** from **{fname}**.{field_note}"
    return custom_chunks, log, status


def chat_respond(message, history, custom_chunks, api_key, model, units, bill_amount):
    index = build_index(custom_chunks or [])
    retrieved = rag_search(message, index, k=3, min_score=0.03)
    bill_facts = f"Units this month: {units or 'unknown'} kWh. Bill amount: Rs. {bill_amount or 'unknown'}."

    client = get_groq_client(api_key)
    if client is not None:
        context_block = "\n\n".join(f"Source: {r['title']}\n{r['text']}" for r in retrieved) \
            or "No matching knowledge base passages retrieved."
        try:
            text = call_groq(
                client, model,
                system_prompt=(
                    "You are PowerSense AI, an assistant inside a household electricity "
                    "dashboard. Answer using ONLY the bill data and knowledge-base context "
                    "given. Be concise (3-5 sentences). If the context doesn't cover the "
                    "question, say so plainly instead of guessing."
                ),
                user_prompt=f"Bill data:\n{bill_facts}\n\nContext:\n{context_block}\n\nQuestion: {message}",
                max_tokens=400,
            )
            if text:
                if retrieved:
                    text += "\n\n📚 Sources: " + ", ".join(sorted({r['title'] for r in retrieved}))
                return text
        except Exception as e:
            return f"⚠️ Groq API error: {e}"

    if retrieved:
        lead = retrieved[0]
        text = f"From **{lead['title']}**: {lead['text']}"
        if len(retrieved) > 1:
            text += f"\n\nAlso relevant — **{retrieved[1]['title']}**: {retrieved[1]['text']}"
        text += f"\n\n_Your numbers: {bill_facts}_"
        return text
    return ("I couldn't find anything indexed relevant to that. Try asking about tariff "
            "slabs, the FPA, sudden bill increases, or AC energy saving — or upload a PDF "
            "for more grounded answers.\n\n💡 Add a Groq API key above for fully generated answers.")


def generate_savings_suggestions(custom_chunks, api_key, model, units, bill_amount, rate):
    try:
        units = float(units) if units else 300.0
    except ValueError:
        units = 300.0
    try:
        bill_amount = float(bill_amount) if bill_amount else units * (rate or 42.0)
    except ValueError:
        bill_amount = units * (rate or 42.0)
    rate = rate or 42.0

    # Baseline rule-based plan (always available, no API key needed)
    plan = [
        ("Shift AC usage outside peak hours (7–11 PM)", 0.095),
        ("Reduce AC runtime by 1 hour/day", 0.07),
        ("Switch remaining bulbs to LED lighting", 0.03),
        ("Service appliances older than 8 years", 0.045),
        ("Unplug standby devices / use power strips", 0.015),
    ]
    lines = []
    total_saving = 0
    for title, pct in plan:
        u = max(1, round(units * pct))
        saving = int(u * rate)
        total_saving += saving
        lines.append(f"- **{title}** — ~{u} units/month, est. **Rs. {saving:,}/month**")
    baseline_text = "\n".join(lines) + f"\n\n**Combined estimated saving: Rs. {total_saving:,}/month (~Rs. {total_saving*12:,}/year)**"

    client = get_groq_client(api_key)
    if client is None:
        return baseline_text, "⚪ No Groq key active — showing the calculated plan only. Add a key above for an AI-written, PDF-grounded plan."

    index = build_index(custom_chunks or [])
    retrieved = rag_search(
        "reduce electricity bill save energy tips appliances peak hours", index, k=4, min_score=0.0
    )
    context_block = "\n\n".join(f"Source: {r['title']}\n{r['text']}" for r in retrieved) \
        or "No specific knowledge-base passages retrieved."
    rule_lines = "\n".join(f"- {t}: ~{round(units*p)} units, est. saving factored at Rs. {rate}/unit" for t, p in plan)
    bill_summary = f"Units this month: {units} kWh. Bill amount: Rs. {bill_amount:,.0f}. Tariff: Rs. {rate}/kWh."

    try:
        text = call_groq(
            client, model,
            system_prompt=(
                "You are PowerSense AI's savings advisor. Using ONLY the bill data, the "
                "baseline calculated recommendations, and the knowledge-base/PDF context "
                "given, write a short personalized savings plan (4-6 bullet points, each "
                "one line, each with a rough Rs./month impact where sensible). Ground "
                "every claim in the data given — don't invent appliances or numbers the "
                "user hasn't mentioned. End with one sentence naming the single "
                "highest-impact action."
            ),
            user_prompt=(
                f"Bill data:\n{bill_summary}\n\nBaseline recommendations:\n{rule_lines}\n\n"
                f"Knowledge base / uploaded PDF context:\n{context_block}\n\n"
                "Write the personalized savings plan."
            ),
            max_tokens=500,
        )
        if text:
            sources = ", ".join(sorted({r['title'] for r in retrieved})) if retrieved else "Knowledge Center"
            return text, f"🧠 AI-generated plan via Groq · grounded in: {sources}"
    except Exception as e:
        return baseline_text, f"⚠️ Groq API error, showing calculated plan instead: {e}"

    return baseline_text, "⚪ AI generation returned nothing — showing the calculated plan."


def save_settings(api_key, model):
    status = "✅ Settings saved for this session."
    if not GROQ_AVAILABLE:
        status = "⚠️ `groq` package isn't installed — retrieval-only mode will be used regardless of key."
    return status


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

CUSTOM_CSS = """
#ps-header {text-align:center; padding: 18px 0 6px 0;}
#ps-title {font-size: 2.1rem; font-weight: 800; margin-bottom: 0;
           background: linear-gradient(90deg, #6366f1, #10b981);
           -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
#ps-subtitle {color: #8a8f98; font-size: 1rem; margin-top: -4px;}
.gradio-container {max-width: 1100px !important; margin: auto !important;}
footer {display: none !important;}
"""

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="emerald",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
)

with gr.Blocks(theme=THEME, css=CUSTOM_CSS, title="PowerSense AI") as demo:
    custom_chunks_state = gr.State([])
    log_state = gr.State([])

    gr.HTML(
        '<div id="ps-header">'
        '<div id="ps-title">⚡ PowerSense AI</div>'
        '<div id="ps-subtitle">Intelligent electricity management — RAG-grounded, powered by Groq</div>'
        '</div>'
    )

    with gr.Accordion("⚙️ AI Settings (Groq)", open=False):
        with gr.Row():
            api_key_box = gr.Textbox(
                label="Groq API key", type="password", scale=2,
                placeholder="gsk_...  (or set the GROQ_API_KEY environment variable)",
            )
            model_box = gr.Textbox(label="Model", value=DEFAULT_MODEL, scale=1)
        settings_status = gr.Markdown("")
        save_btn = gr.Button("Save settings", size="sm")
        save_btn.click(save_settings, inputs=[api_key_box, model_box], outputs=settings_status)
        if not GROQ_AVAILABLE:
            gr.Markdown("⚠️ `groq` isn't installed — run `pip install groq`.")
        if not SKLEARN_AVAILABLE:
            gr.Markdown("⚠️ `scikit-learn` isn't installed — RAG retrieval is disabled.")

    with gr.Tabs():
        # -------------------- TAB 1: UPLOAD & BILL --------------------
        with gr.Tab("📄 Upload Bill / Documents"):
            gr.Markdown(
                "Upload your electricity bill (PDF) — or any related document like a "
                "tariff notice — to ground the AI Assistant and Savings Suggestions in "
                "your own real data."
            )
            pdf_upload = gr.File(label="Upload a PDF", file_types=[".pdf"])
            upload_status = gr.Markdown("No document indexed yet.")

            gr.Markdown("##### Your usage (auto-detected fields can be overridden here)")
            with gr.Row():
                units_box = gr.Number(label="Units consumed this month (kWh)", value=300)
                bill_box = gr.Number(label="Total bill amount (Rs.)", value=None)
                rate_box = gr.Number(label="Tariff rate (Rs./unit)", value=42.0)

            pdf_upload.change(
                handle_pdf_upload,
                inputs=[pdf_upload, custom_chunks_state, log_state],
                outputs=[custom_chunks_state, log_state, upload_status],
            )

        # -------------------- TAB 2: AI ASSISTANT --------------------
        with gr.Tab("🤖 Ask AI"):
            gr.Markdown(
                "Ask about your bill, tariffs, sudden increases, or how to save — answers "
                "are retrieved from the Knowledge Center plus anything you've uploaded, "
                "and (with a Groq key) generated by a live model."
            )
            gr.Markdown(
                "_Try: \"Why is my bill high this month?\" · \"How can I reduce my "
                "electricity bill?\" · \"What is the FPA charge?\" · \"Is my consumption "
                "unusual?\"_"
            )
            chatbot = gr.ChatInterface(
                fn=chat_respond,
                additional_inputs=[custom_chunks_state, api_key_box, model_box, units_box, bill_box],
                type="messages",
            )

        # -------------------- TAB 3: SAVINGS SUGGESTIONS --------------------
        with gr.Tab("💡 Savings Suggestions"):
            gr.Markdown(
                "Get a personalized savings plan. Works without any API key (calculated "
                "plan); add a Groq key in Settings above for an AI-written plan grounded "
                "in your bill and any uploaded documents."
            )
            generate_btn = gr.Button("✨ Generate Savings Suggestions", variant="primary")
            suggestions_out = gr.Markdown()
            suggestions_meta = gr.Markdown()

            generate_btn.click(
                generate_savings_suggestions,
                inputs=[custom_chunks_state, api_key_box, model_box, units_box, bill_box, rate_box],
                outputs=[suggestions_out, suggestions_meta],
            )

    gr.Markdown(
        "<div style='text-align:center; color:#8a8f98; font-size:0.8rem; padding-top:12px;'>"
        "PowerSense AI · Streamlit + Gradio + Groq · RAG grounded in your own documents"
        "</div>"
    )

if __name__ == "__main__":
    demo.launch()
