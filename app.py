
import io
import os
import re
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

# Optional dependencies
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

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


st.set_page_config(
    page_title="PowerSense AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
#MainMenu, footer {visibility:hidden;}
.ps-title{font-size:2rem;font-weight:750;margin-bottom:2px}
.ps-subtitle{color:#7b8190;margin-bottom:1.2rem}
.kpi{border:1px solid rgba(120,120,120,.18);border-radius:15px;padding:1rem;background:rgba(120,120,120,.045);height:100%}
.kpi-label{font-size:.8rem;color:#7b8190}
.kpi-value{font-size:1.55rem;font-weight:750;margin-top:4px}
.source{display:inline-block;border-radius:999px;padding:3px 9px;margin:2px;background:rgba(99,102,241,.1);font-size:.75rem}
.notice{border-left:4px solid #6366f1;padding:.8rem 1rem;border-radius:8px;background:rgba(99,102,241,.06)}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# SESSION STATE — no synthetic electricity data is ever created.
# ---------------------------------------------------------------------
defaults = {
    "bill_data": None,
    "bill_raw_text": "",
    "bill_source": "",
    "bill_history": [],
    "chat_history": [],
    "custom_kb_chunks": [],
    "uploaded_kb_files": [],
    "groq_api_key": "",
    "llm_model": "llama-3.3-70b-versatile",
    "nav_page": "🏠 Dashboard",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------
# KNOWLEDGE BASE
# ---------------------------------------------------------------------
KNOWLEDGE_BASE = [
    {
        "title": "Understanding Electricity Tariff Slabs",
        "category": "Billing",
        "content": (
            "Residential electricity tariffs can use consumption slabs, so the "
            "price structure may change when usage crosses a threshold. The exact "
            "rates depend on the utility, connection category and applicable tariff."
        ),
    },
    {
        "title": "Fuel Price Adjustment (FPA)",
        "category": "Billing",
        "content": (
            "Fuel Price Adjustment is a variable billing component that can increase "
            "or decrease independently of household consumption. The amount printed "
            "on the electricity bill should be treated as the authoritative value."
        ),
    },
    {
        "title": "Checking Meter Readings",
        "category": "Troubleshooting",
        "content": (
            "Compare the previous and current meter readings printed on the bill with "
            "the physical meter where possible. If the readings do not match, contact "
            "the relevant electricity provider."
        ),
    },
    {
        "title": "Common Reasons for a Higher Bill",
        "category": "Troubleshooting",
        "content": (
            "A higher bill can result from higher units consumed, tariff changes, "
            "taxes or adjustments, seasonal appliance use, or billing issues. "
            "The bill itself should be checked before assigning a specific cause."
        ),
    },
    {
        "title": "Energy Saving Basics",
        "category": "Conservation",
        "content": (
            "Energy-saving actions include reducing unnecessary appliance runtime, "
            "maintaining cooling equipment, improving insulation and switching off "
            "equipment when it is not needed. These are general recommendations, "
            "not measurements of a particular household's appliance consumption."
        ),
    },
]


def split_sentences(text):
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x.strip()]


def chunk_text(text, n=3):
    s = split_sentences(text)
    return [" ".join(s[i:i+n]) for i in range(0, len(s), n) if s[i:i+n]]


def build_base_chunks():
    out = []
    for article in KNOWLEDGE_BASE:
        for c in chunk_text(article["content"]):
            out.append({
                "title": article["title"],
                "category": article["category"],
                "text": c,
            })
    return out


def get_rag_index():
    chunks = build_base_chunks() + st.session_state.custom_kb_chunks
    if not SKLEARN_AVAILABLE or not chunks:
        return None, None, chunks

    texts = [x["text"] for x in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix, chunks


def rag_search(query, k=4, min_score=0.05):
    vectorizer, matrix, chunks = get_rag_index()
    if vectorizer is None or matrix is None:
        return []

    q = vectorizer.transform([query])
    scores = cosine_similarity(q, matrix).flatten()
    ranked = scores.argsort()[::-1][:k]

    result = []
    for i in ranked:
        if scores[i] >= min_score:
            item = dict(chunks[i])
            item["score"] = round(float(scores[i]), 3)
            result.append(item)
    return result


# ---------------------------------------------------------------------
# REAL BILL EXTRACTION
# ---------------------------------------------------------------------
def clean_number(value):
    if value is None:
        return None
    value = str(value).replace(",", "").replace("Rs.", "").replace("Rs", "")
    m = re.search(r"-?\d+(?:\.\d+)?", value)
    if not m:
        return None
    number = float(m.group())
    return int(number) if number.is_integer() else number


def first_match(text, patterns):
    for pattern in patterns:
        m = re.search(pattern, text, re.I | re.M)
        if m:
            return m.group(1).strip()
    return None


def parse_bill_text(text):
    """
    Extract only values actually present in the bill text.
    Missing fields remain None. No demo values are generated.
    """
    text = re.sub(r"[ \t]+", " ", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)

    data = {}

    data["consumer_id"] = first_match(text, [
        r"(?:consumer\s*(?:id|no|number)|reference\s*(?:no|number))\s*[:#\-]?\s*([A-Za-z0-9\-]{5,30})",
    ])

    data["meter_no"] = first_match(text, [
        r"(?:meter\s*(?:no|number|#))\s*[:#\-]?\s*([A-Za-z0-9\-]{3,30})",
    ])

    data["units_consumed"] = clean_number(first_match(text, [
        r"(?:units?\s*(?:consumed|used)|consumption|units)\s*[:\-]?\s*([\d,]+(?:\.\d+)?)",
    ]))

    data["previous_reading"] = clean_number(first_match(text, [
        r"(?:previous|prev)\s*(?:meter\s*)?reading\s*[:\-]?\s*([\d,]+(?:\.\d+)?)",
    ]))

    data["current_reading"] = clean_number(first_match(text, [
        r"(?:current|present|present\s*meter)\s*(?:meter\s*)?reading\s*[:\-]?\s*([\d,]+(?:\.\d+)?)",
    ]))

    data["total_payable"] = clean_number(first_match(text, [
        r"(?:total\s*(?:amount\s*)?(?:payable|due)|amount\s*(?:payable|due)|net\s*payable)\s*[:\-]?\s*(?:rs\.?|pkr)?\s*([\d,]+(?:\.\d+)?)",
    ]))

    data["due_date"] = first_match(text, [
        r"(?:due\s*date|payable\s*by)\s*[:\-]?\s*([0-3]?\d[\/\-\s][A-Za-z0-9]{2,10}[\/\-\s]\d{2,4})",
        r"(?:due\s*date|payable\s*by)\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
    ])

    data["billing_month"] = first_match(text, [
        r"(?:billing\s*month|bill\s*month|month)\s*[:\-]?\s*([A-Za-z]+\s+\d{4})",
    ])

    # Optional printed bill components — only populated if found.
    component_patterns = {
        "electricity_charges": [
            r"(?:electricity\s*charges?|energy\s*charges?|cost\s*of\s*electricity)\s*[:\-]?\s*(?:rs\.?|pkr)?\s*([\d,]+(?:\.\d+)?)"
        ],
        "taxes": [
            r"(?:gst|tax(?:es)?|general\s*sales\s*tax)\s*[:\-]?\s*(?:rs\.?|pkr)?\s*([\d,]+(?:\.\d+)?)"
        ],
        "fpa_adjustment": [
            r"(?:fpa|fuel\s*price\s*adjustment)\s*[:\-]?\s*(?:rs\.?|pkr)?\s*(-?[\d,]+(?:\.\d+)?)"
        ],
        "electricity_duty": [
            r"(?:electricity\s*duty)\s*[:\-]?\s*(?:rs\.?|pkr)?\s*([\d,]+(?:\.\d+)?)"
        ],
        "nj_surcharge": [
            r"(?:n\.?\s*j\.?\s*surcharge|nj\s*surcharge)\s*[:\-]?\s*(?:rs\.?|pkr)?\s*([\d,]+(?:\.\d+)?)"
        ],
    }

    for key, patterns in component_patterns.items():
        value = clean_number(first_match(text, patterns))
        if value is not None:
            data[key] = value

    # If the bill contains both readings, units can be calculated from those
    # actual readings — but only when an explicit units value was not found.
    if data.get("units_consumed") is None:
        prev = data.get("previous_reading")
        curr = data.get("current_reading")
        if prev is not None and curr is not None and curr >= prev:
            data["units_consumed"] = curr - prev

    return {k: v for k, v in data.items() if v not in (None, "")}


def extract_pdf_text(uploaded_file):
    if not PYMUPDF_AVAILABLE:
        return ""

    try:
        doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
        text = "\n".join(page.get_text("text") for page in doc)
        doc.close()
        return text.strip()
    except Exception:
        return ""


def ocr_image_bytes(image_bytes):
    if not OCR_AVAILABLE:
        return ""

    try:
        image = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(image).strip()
    except Exception:
        return ""


def extract_bill_file(uploaded_file):
    """
    Returns raw text and extraction method.
    Never creates substitute/fake bill values.
    """
    if uploaded_file is None:
        return "", "none"

    file_type = (getattr(uploaded_file, "type", "") or "").lower()
    raw = uploaded_file.getvalue()

    if "pdf" in file_type or uploaded_file.name.lower().endswith(".pdf"):
        text = extract_pdf_text(uploaded_file)
        if text.strip():
            return text, "PDF text extraction"

        # Try OCR for scanned PDFs by rendering pages.
        if PYMUPDF_AVAILABLE and OCR_AVAILABLE:
            try:
                doc = fitz.open(stream=raw, filetype="pdf")
                pages = []
                for page in doc:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    pages.append(ocr_image_bytes(pix.tobytes("png")))
                doc.close()
                text = "\n".join(x for x in pages if x).strip()
                if text:
                    return text, "OCR"
            except Exception:
                pass

        return "", "unreadable PDF"

    if "image" in file_type or uploaded_file.name.lower().endswith((".png", ".jpg", ".jpeg")):
        text = ocr_image_bytes(raw)
        return text, "OCR" if text else "unreadable image"

    return "", "unsupported file type"


def process_bill(uploaded_file):
    raw_text, method = extract_bill_file(uploaded_file)

    if not raw_text.strip():
        return None, "", method

    parsed = parse_bill_text(raw_text)

    if not parsed:
        return None, raw_text, method

    parsed["_source_file"] = uploaded_file.name
    parsed["_extraction_method"] = method
    parsed["_fields_found"] = list(parsed.keys())

    return parsed, raw_text, method


# ---------------------------------------------------------------------
# GROQ — grounded strictly in actual extracted data
# ---------------------------------------------------------------------
def get_groq_client():
    key = (
        st.session_state.get("groq_api_key")
        or os.environ.get("GROQ_API_KEY", "")
    )
    if not key or not GROQ_AVAILABLE:
        return None
    try:
        return Groq(api_key=key)
    except Exception:
        return None


def bill_context():
    bill = st.session_state.bill_data
    if not bill:
        return "NO BILL HAS BEEN UPLOADED."

    allowed = {k: v for k, v in bill.items() if not k.startswith("_")}
    return "\n".join(f"{k}: {v}" for k, v in allowed.items())


def ai_chat_response(question):
    bill = st.session_state.bill_data
    retrieved = rag_search(question, k=4)

    if not bill:
        return (
            "Please upload an electricity bill first. I can answer general "
            "electricity questions, but I will not invent your bill's numbers.",
            [r["title"] for r in retrieved],
            "retrieval",
        )

    context = "\n\n".join(
        f"Source: {r['title']}\n{r['text']}" for r in retrieved
    )

    client = get_groq_client()
    if client:
        try:
            response = client.chat.completions.create(
                model=st.session_state.llm_model,
                max_tokens=500,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are PowerSense AI. Use ONLY the supplied extracted "
                            "electricity-bill facts and knowledge-base text. "
                            "Never invent, estimate, assume, or fabricate a bill number. "
                            "If a requested field is absent, say 'Not available on the "
                            "uploaded bill.' Do not turn general knowledge into a claim "
                            "about this household. Clearly distinguish bill facts from "
                            "general recommendations."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"EXTRACTED BILL FACTS:\n{bill_context()}\n\n"
                            f"KNOWLEDGE BASE:\n{context or 'No matching knowledge retrieved.'}\n\n"
                            f"QUESTION: {question}"
                        ),
                    },
                ],
            )
            answer = (response.choices[0].message.content or "").strip()
            if answer:
                return answer, [r["title"] for r in retrieved], "llm"
        except Exception as exc:
            st.session_state["_llm_error"] = str(exc)

    # Retrieval-only fallback.
    if retrieved:
        return (
            "Here is the relevant information from the Knowledge Center:\n\n"
            + "\n\n".join(f"**{r['title']}** — {r['text']}" for r in retrieved),
            [r["title"] for r in retrieved],
            "retrieval",
        )

    return (
        "I could not find reliable information for that question in the "
        "Knowledge Center. I will not guess.",
        [],
        "retrieval",
    )


# ---------------------------------------------------------------------
# BILL-ONLY CALCULATIONS
# ---------------------------------------------------------------------
def bill_health(bill):
    """
    Health is calculated only when the same bill provides enough evidence.
    No historical comparison is invented.
    """
    units = bill.get("units_consumed")
    prev = bill.get("previous_reading")
    curr = bill.get("current_reading")

    if units is None:
        return "Unavailable", "Not enough bill data to assess consumption."

    if prev is not None and curr is not None:
        calculated = curr - prev
        if calculated >= 0 and calculated == units:
            return "Consistent", "Units consumed match current reading − previous reading."
        if calculated >= 0:
            return "Check", (
                f"The bill reports {units} units, while the two readings imply "
                f"{calculated} units. Verify the readings on the bill."
            )

    return "No comparison", "The bill contains units but not enough readings for a meter cross-check."


def recommendations(bill):
    """
    General recommendations only. No fake savings amounts are generated.
    """
    recs = [
        "Review high-consumption appliances such as ACs, heaters and water pumps.",
        "Compare the printed current and previous meter readings with the physical meter.",
        "Check FPA, taxes and other printed adjustments separately from energy charges.",
        "If usage is unexpectedly high, compare this bill with earlier real bills.",
        "Use appliance maintenance and reduced runtime as general energy-saving measures.",
    ]
    return recs


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ PowerSense AI")
    st.caption("Real-bill electricity intelligence")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠 Dashboard",
            "🧾 Bill Analyzer",
            "📊 Consumption",
            "🚨 Problem Detection",
            "💡 Recommendations",
            "💰 Savings Simulator",
            "🤖 AI Assistant",
            "📚 Knowledge Center",
            "⚙️ Settings",
        ],
        label_visibility="collapsed",
        key="nav_page",
    )

    st.divider()
    st.markdown("##### Data integrity")
    if st.session_state.bill_data:
        st.success("REAL BILL LOADED")
        st.caption(
            f"Source: {st.session_state.bill_data.get('_source_file', 'uploaded bill')}"
        )
    else:
        st.warning("NO BILL LOADED")
        st.caption("No electricity figures are shown until a real bill is uploaded.")

    st.divider()
    st.caption("PowerSense never creates synthetic meter history or fake bill totals.")


# ---------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------
if page == "🏠 Dashboard":
    st.markdown(
        '<div class="ps-title">PowerSense AI</div>'
        '<div class="ps-subtitle">Electricity intelligence grounded in your actual bill</div>',
        unsafe_allow_html=True,
    )

    bill = st.session_state.bill_data

    if not bill:
        st.markdown(
            '<div class="notice"><b>Upload a real electricity bill to begin.</b><br>'
            'Until a bill is uploaded, PowerSense will not display invented units, '
            'amounts, readings, history or savings.</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("🧾 Upload / Analyze Bill", type="primary"):
            st.session_state.nav_page = "🧾 Bill Analyzer"
            st.rerun()

    else:
        total = bill.get("total_payable")
        units = bill.get("units_consumed")
        due = bill.get("due_date")
        health, health_detail = bill_health(bill)

        cols = st.columns(4)
        values = [
            ("💰 Total Payable", f"Rs. {total:,}" if total is not None else "Not available"),
            ("⚡ Units Consumed", f"{units} kWh" if units is not None else "Not available"),
            ("📅 Due Date", due or "Not available"),
            ("🔎 Bill Check", health),
        ]
        for col, (label, value) in zip(cols, values):
            with col:
                st.markdown(
                    f'<div class="kpi"><div class="kpi-label">{label}</div>'
                    f'<div class="kpi-value">{value}</div></div>',
                    unsafe_allow_html=True,
                )

        st.write("")
        st.info(health_detail)

        st.markdown("#### What PowerSense knows from this bill")
        rows = []
        labels = {
            "consumer_id": "Consumer / Reference ID",
            "meter_no": "Meter Number",
            "billing_month": "Billing Month",
            "previous_reading": "Previous Reading",
            "current_reading": "Current Reading",
            "units_consumed": "Units Consumed",
            "electricity_charges": "Electricity Charges",
            "taxes": "Taxes / GST",
            "fpa_adjustment": "FPA",
            "electricity_duty": "Electricity Duty",
            "nj_surcharge": "N.J. Surcharge",
            "total_payable": "Total Payable",
            "due_date": "Due Date",
        }
        for key, label in labels.items():
            if key in bill:
                rows.append({"Field": label, "Value": bill[key]})

        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        st.caption(
            f"Extraction method: {bill.get('_extraction_method', 'unknown')} · "
            f"Source: {bill.get('_source_file', 'uploaded bill')}"
        )


# ---------------------------------------------------------------------
# BILL ANALYZER
# ---------------------------------------------------------------------
elif page == "🧾 Bill Analyzer":
    st.markdown(
        '<div class="ps-title">🧾 Bill Analyzer</div>'
        '<div class="ps-subtitle">Extract only information actually printed on your bill</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload electricity bill",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Text PDF, scanned PDF, PNG or JPG.",
    )

    if uploaded:
        with st.spinner("Reading and validating the bill..."):
            data, raw_text, method = process_bill(uploaded)

        if data:
            st.session_state.bill_data = data
            st.session_state.bill_raw_text = raw_text
            st.session_state.bill_source = uploaded.name

            # Add the actual bill text to RAG.
            if raw_text.strip():
                chunks = chunk_text(raw_text)
                st.session_state.custom_kb_chunks = [
                    x for x in st.session_state.custom_kb_chunks
                    if x.get("_source_file") != uploaded.name
                ]
                st.session_state.custom_kb_chunks.extend(
                    {
                        "title": f"Uploaded Bill — {uploaded.name}",
                        "category": "User Bill",
                        "text": c,
                        "_source_file": uploaded.name,
                    }
                    for c in chunks
                )

            st.success(
                f"Bill processed successfully using {method}. "
                f"{len(data.get('_fields_found', []))} real fields extracted."
            )
        else:
            st.session_state.bill_data = None
            st.error(
                "No reliable bill fields could be extracted. "
                "No fake/demo data was inserted."
            )
            if method == "unreadable PDF":
                if not OCR_AVAILABLE:
                    st.warning(
                        "This may be a scanned PDF. Install OCR dependencies "
                        "listed in requirements.txt and the Tesseract system package."
                    )
            elif method == "unreadable image":
                st.warning("OCR could not read this image. Try a clearer scan.")

    bill = st.session_state.bill_data

    if bill:
        st.markdown("#### Extracted bill data")
        rows = []
        labels = {
            "consumer_id": "Consumer / Reference ID",
            "meter_no": "Meter Number",
            "billing_month": "Billing Month",
            "previous_reading": "Previous Reading",
            "current_reading": "Current Reading",
            "units_consumed": "Units Consumed (kWh)",
            "electricity_charges": "Electricity Charges (Rs.)",
            "taxes": "Taxes / GST (Rs.)",
            "fpa_adjustment": "FPA (Rs.)",
            "electricity_duty": "Electricity Duty (Rs.)",
            "nj_surcharge": "N.J. Surcharge (Rs.)",
            "total_payable": "Total Payable (Rs.)",
            "due_date": "Due Date",
        }
        for key, label in labels.items():
            rows.append({
                "Field": label,
                "Value": bill.get(key, "Not available on bill")
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        st.markdown("#### Bill consistency check")
        health, detail = bill_health(bill)
        if health == "Consistent":
            st.success(f"{health}: {detail}")
        elif health == "Check":
            st.warning(f"{health}: {detail}")
        else:
            st.info(f"{health}: {detail}")

        st.markdown("#### Bill source")
        st.caption(
            f"File: {bill.get('_source_file')} · "
            f"Extraction: {bill.get('_extraction_method')}"
        )

        with st.expander("View extracted text"):
            st.text(st.session_state.bill_raw_text[:20000])


# ---------------------------------------------------------------------
# CONSUMPTION — ONLY REAL BILLS / NO MODELED HOURLY CURVE
# ---------------------------------------------------------------------
elif page == "📊 Consumption":
    st.markdown(
        '<div class="ps-title">📊 Consumption Analytics</div>'
        '<div class="ps-subtitle">Real bill consumption only</div>',
        unsafe_allow_html=True,
    )

    bill = st.session_state.bill_data

    if not bill:
        st.info(
            "Upload a real electricity bill first. One bill can provide its own "
            "consumption, but it cannot create a 12-month history or hourly smart-meter data."
        )
    else:
        units = bill.get("units_consumed")
        if units is not None:
            st.metric("Actual units on uploaded bill", f"{units} kWh")
        else:
            st.warning("Units consumed are not available on the uploaded bill.")

        prev = bill.get("previous_reading")
        curr = bill.get("current_reading")
        if prev is not None and curr is not None:
            st.metric("Reading difference", f"{curr - prev} kWh")
            if units is not None and curr - prev != units:
                st.warning(
                    "The printed units do not match the difference between the two "
                    "printed readings. Please verify the bill."
                )

        st.markdown("#### Historical consumption")
        if len(st.session_state.bill_history) >= 2:
            hist = pd.DataFrame(st.session_state.bill_history)
            if {"billing_month", "units_consumed"}.issubset(hist.columns):
                hist = hist.dropna(subset=["units_consumed"])
                if not hist.empty:
                    fig = px.line(hist, x="billing_month", y="units_consumed", markers=True)
                    fig.update_layout(yaxis_title="Real units (kWh)", xaxis_title=None)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "No synthetic history is shown. Upload previous real bills to build "
                "a real consumption history."
            )

        st.markdown("#### Hourly usage")
        st.info(
            "Hourly usage is not available because an ordinary electricity bill does "
            "not contain 24-hour interval meter measurements. Upload real smart-meter "
            "interval data if you want hourly analytics."
        )


# ---------------------------------------------------------------------
# PROBLEM DETECTION — EVIDENCE-BASED ONLY
# ---------------------------------------------------------------------
elif page == "🚨 Problem Detection":
    st.markdown(
        '<div class="ps-title">🚨 Problem Detection</div>'
        '<div class="ps-subtitle">Flags issues only when the bill contains evidence</div>',
        unsafe_allow_html=True,
    )

    bill = st.session_state.bill_data
    if not bill:
        st.info("Upload a real bill first.")
    else:
        health, detail = bill_health(bill)
        st.subheader(f"Bill check: {health}")
        st.write(detail)

        if bill.get("total_payable") is None:
            st.warning("Total payable is not available on the bill.")
        if bill.get("units_consumed") is None:
            st.warning("Units consumed are not available on the bill.")
        if bill.get("previous_reading") is None or bill.get("current_reading") is None:
            st.info("A meter-reading comparison cannot be performed from this bill.")

        st.markdown("#### Evidence from the bill")
        evidence = []
        for key in [
            "units_consumed", "previous_reading", "current_reading",
            "electricity_charges", "taxes", "fpa_adjustment",
            "electricity_duty", "nj_surcharge", "total_payable"
        ]:
            if key in bill:
                evidence.append({"Field": key, "Actual value": bill[key]})

        if evidence:
            st.dataframe(pd.DataFrame(evidence), hide_index=True, use_container_width=True)

        st.markdown("#### General possible causes")
        for r in rag_search("electricity bill unusual consumption billing issue", k=3, min_score=0.02):
            st.markdown(f"- {r['text']}")
            st.markdown(f'<span class="source">📚 {r["title"]}</span>', unsafe_allow_html=True)


# ---------------------------------------------------------------------
# RECOMMENDATIONS — NO FABRICATED SAVINGS
# ---------------------------------------------------------------------
elif page == "💡 Recommendations":
    st.markdown(
        '<div class="ps-title">💡 Recommendations</div>'
        '<div class="ps-subtitle">Practical advice without inventing appliance usage or savings</div>',
        unsafe_allow_html=True,
    )

    bill = st.session_state.bill_data
    if not bill:
        st.info("Upload a real bill first.")
    else:
        st.markdown("#### Based on your actual bill")
        if bill.get("units_consumed") is not None:
            st.write(f"Your bill reports **{bill['units_consumed']} kWh**.")
        if bill.get("total_payable") is not None:
            st.write(f"Your printed total payable is **Rs. {bill['total_payable']:,}**.")

        for r in recommendations(bill):
            with st.container(border=True):
                st.write("💡", r)

        st.info(
            "PowerSense does not display a fake 'Rs. saved per month' figure. "
            "A savings amount requires measured usage, a validated tariff and/or "
            "real historical bills."
        )

        client = get_groq_client()
        if client:
            if st.button("✨ Generate AI plan from this bill", type="primary"):
                retrieved = rag_search(
                    "energy saving electricity bill appliances consumption",
                    k=4,
                    min_score=0.0,
                )
                context = "\n".join(r["text"] for r in retrieved)
                try:
                    response = client.chat.completions.create(
                        model=st.session_state.llm_model,
                        max_tokens=450,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Create a practical electricity-saving plan. "
                                    "Use only the supplied bill facts. Never invent "
                                    "appliances, usage hours or rupee savings. "
                                    "If savings cannot be calculated from the data, "
                                    "say that explicitly."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"BILL:\n{bill_context()}\n\n"
                                    f"KNOWLEDGE:\n{context}"
                                ),
                            },
                        ],
                    )
                    st.markdown(response.choices[0].message.content)
                except Exception as exc:
                    st.error(f"AI plan could not be generated: {exc}")
        else:
            st.caption("Add a Groq API key in Settings for the generated plan.")


# ---------------------------------------------------------------------
# SAVINGS SIMULATOR — SCENARIO ONLY, NEVER PRESENTED AS REAL BILL DATA
# ---------------------------------------------------------------------
elif page == "💰 Savings Simulator":
    st.markdown(
        '<div class="ps-title">💰 Savings Scenario Simulator</div>'
        '<div class="ps-subtitle">A hypothetical calculator — not a measurement of your household</div>',
        unsafe_allow_html=True,
    )

    bill = st.session_state.bill_data
    if not bill or bill.get("total_payable") is None:
        st.info(
            "Upload a bill containing a total payable amount first. "
            "The simulator will then use that actual printed amount as its baseline."
        )
    else:
        current_bill = float(bill["total_payable"])
        reduction = st.slider(
            "Hypothetical reduction in the bill (%)",
            min_value=0,
            max_value=50,
            value=10,
        )
        projected = current_bill * (1 - reduction / 100)

        c1, c2, c3 = st.columns(3)
        c1.metric("Actual printed bill", f"Rs. {current_bill:,.0f}")
        c2.metric("Scenario reduction", f"{reduction}%")
        c3.metric("Hypothetical bill", f"Rs. {projected:,.0f}")

        st.caption(
            "This is a user-controlled scenario, not a prediction. "
            "PowerSense does not claim that the household will actually save this amount."
        )

        fig = px.bar(
            pd.DataFrame({
                "Scenario": ["Actual bill", "Hypothetical"],
                "Rs.": [current_bill, projected],
            }),
            x="Scenario",
            y="Rs.",
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# AI ASSISTANT
# ---------------------------------------------------------------------
elif page == "🤖 AI Assistant":
    st.markdown(
        '<div class="ps-title">🤖 Ask PowerSense</div>'
        '<div class="ps-subtitle">Answers grounded in your uploaded bill and the Knowledge Center</div>',
        unsafe_allow_html=True,
    )

    quick = [
        "What is my total payable?",
        "How many units did I consume?",
        "Do my meter readings match my units?",
        "What is FPA on my bill?",
    ]

    cols = st.columns(len(quick))
    clicked = None
    for col, q in zip(cols, quick):
        if col.button(q, use_container_width=True):
            clicked = q

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for source in msg.get("sources", []):
                st.markdown(
                    f'<span class="source">📚 {source}</span>',
                    unsafe_allow_html=True,
                )

    user_input = st.chat_input("Ask about your actual bill...")
    prompt = clicked or user_input

    if prompt:
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt,
        })
        answer, sources, mode = ai_chat_response(prompt)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "mode": mode,
        })
        st.rerun()


# ---------------------------------------------------------------------
# KNOWLEDGE CENTER
# ---------------------------------------------------------------------
elif page == "📚 Knowledge Center":
    st.markdown(
        '<div class="ps-title">📚 Knowledge Center</div>'
        '<div class="ps-subtitle">RAG over electricity guidance and your uploaded bill text</div>',
        unsafe_allow_html=True,
    )

    if not SKLEARN_AVAILABLE:
        st.warning("Install scikit-learn to enable TF-IDF retrieval.")

    kb_upload = st.file_uploader(
        "Add a knowledge PDF",
        type=["pdf"],
        key="kb_upload",
    )

    if kb_upload and kb_upload.name not in st.session_state.uploaded_kb_files:
        raw = extract_pdf_text(kb_upload)
        if raw:
            for c in chunk_text(raw):
                st.session_state.custom_kb_chunks.append({
                    "title": kb_upload.name,
                    "category": "Uploaded",
                    "text": c,
                })
            st.session_state.uploaded_kb_files.append(kb_upload.name)
            st.success(f"Indexed {kb_upload.name}")
        else:
            st.error("Could not extract text from this PDF.")

    query = st.text_input("Test RAG retrieval")
    if query:
        results = rag_search(query, k=5, min_score=0.0)
        for r in results:
            with st.container(border=True):
                st.markdown(f"**{r['title']}** · similarity {r['score']}")
                st.write(r["text"])

    st.markdown("#### Built-in knowledge")
    for article in KNOWLEDGE_BASE:
        with st.expander(article["title"]):
            st.write(article["content"])


# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------
elif page == "⚙️ Settings":
    st.markdown(
        '<div class="ps-title">⚙️ Settings</div>'
        '<div class="ps-subtitle">AI configuration only — bill numbers are never manually fabricated</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Groq")
    key = st.text_input(
        "Groq API key",
        value=st.session_state.groq_api_key,
        type="password",
    )
    model = st.text_input(
        "Groq model",
        value=st.session_state.llm_model,
    )

    if st.button("Save AI settings", type="primary"):
        st.session_state.groq_api_key = key
        st.session_state.llm_model = model.strip() or "llama-3.3-70b-versatile"
        st.success("AI settings saved for this session.")

    st.divider()
    st.markdown("#### Data integrity policy")
    st.success(
        "Real bill values only. Missing fields are shown as 'Not available'. "
        "No synthetic 12-month history, fake meter readings, fake bill totals, "
        "fake taxes, fake FPA or fake hourly load curves are generated."
    )

    st.markdown("#### Current extraction status")
    if st.session_state.bill_data:
        st.write(
            f"Loaded: **{st.session_state.bill_data.get('_source_file')}** · "
            f"Method: **{st.session_state.bill_data.get('_extraction_method')}**"
        )
    else:
        st.write("No bill loaded.")
