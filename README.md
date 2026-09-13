# ⚡ PowerSense AI — Pakistan Electricity Bill Intelligence & Verification System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://powersense-ai2-ge5yixnuauquakhbuaaw5n.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

**PowerSense AI** is a state-of-the-art electricity bill transparency, mathematical verification, and regulatory intelligence platform built specifically for Pakistani electricity consumers across all distribution companies (**FESCO, IESCO, LESCO, GEPCO, MEPCO, PESCO, HESCO, SEPCO, QESCO, and K-Electric**).

---

## 🌟 Key Upgrades in This Version

1. **PDF is the Source of Truth — 100% Real Bill History**:
   - Previous versions generated synthetic 12-month series or showed "Not available" when LLM regex failed.
   - The upgraded engine extracts the **genuine 12-month consumption, billed amount, and payment history** actually printed on Pakistani bills (e.g. Aug 2025 – Jul 2026 + current Aug 2026 month).
   - **Zero fabricated or synthetic numbers**.

2. **Dedicated "BILL CHECK — PDF Verification" Engine**:
   - Automated mathematical audit matching NEPRA consumer regulations:
     - ✅ **Meter Reading Check**: `Present Reading − Previous Reading = Units Consumed` (Exact match).
     - ✅ **Current Bill Check**: `Net Electricity Charges + Taxes = Current Bill` (Exact match).
     - ✅ **Grand Total Check**: `Current Bill + Total FPA = Payable Within Due Date` (Exact match).
     - ✅ **Subsidy Check**: `Gross Charges − Protected Subsidy = Net Charges` (Verified).
     - ✅ **Payment Check**: Confirms payment date and settlement against outstanding dues (0 arrears).
     - ✅ **Consumption Anomaly Analysis**: Compares current usage against previous month, 12-month average, and peak month to detect true anomalies rather than normal seasonal domestic cooling patterns.
     - 🟡 **Regulatory Component Analysis (FPA)**: Accurately explains Fuel Price Adjustment as a statutory NEPRA generation mix variation rather than an overbilling error.

3. **10/10 Modern Interactive Visual Interface**:
   - Plotly-powered dynamic charts:
     - 12-Month Consumption Trend (kWh) with historical average line and 200 kWh Protected Slab Threshold.
     - Monthly Billed Amount vs Consumer Payment History.
     - True Bill Net Flow Waterfall (Gross Energy → Subsidy → Net Energy → Taxes → FPA → Grand Total) with zero duplicate double counting.
     - Cost-Per-Unit (PKR/kWh) metric trend over time.

4. **Transparent Interval Data Disclosure**:
   - Transparently clarifies that monthly induction/electronic meters record cumulative monthly kWh, and hourly/24-hour load profiles are not manufactured without smart AMI meters.

5. **Hypothetical Savings Simulator**:
   - Explicitly separated from actual bill data.
   - Allows users to model hypothetical appliance usage reductions (e.g., Inverter AC runtime reduction, LED retrofits).
   - Monitors the 200 kWh threshold to safeguard Protected Consumer status.

6. **Grounded RAG & Groq LLM Assistant**:
   - Cross-references extracted bill figures with official NEPRA Consumer Service Manual (CSM) and tariff schedules in `knowledge/`.
   - Supports English, Urdu (اردو), and Roman Urdu.

7. **Formal Complaint Assistant**:
   - Identifies legitimate billing disputes and drafts formal, factual letters citing meter numbers, dates, and reference IDs.
   - Provides verified direct links to the official [NEPRA Consumer Complaint Portal](https://www.nepra.org.pk/Complaint.php).

---

## 📁 Repository Structure

```text
├── app.py                 # Main Streamlit 10/10 application
├── rag_utils.py           # Bill parser, verification engine & RAG pipeline
├── requirements.txt       # Python dependencies (Streamlit, Plotly, PyPDF, Groq, etc.)
├── packages.txt           # Debian apt packages for Streamlit Cloud (Tesseract OCR)
├── apt_packages.txt       # System package aliases
├── knowledge/             # Official NEPRA & DISCO reference PDFs
│   ├── 01_NEPRA_Consumer_Service_Manual_2025_Summary.pdf
│   ├── 02_FESCO_2026_Tariff_Reference.pdf
│   ├── 03_NEPRA_Complaint_Handling_Rules_2015_Summary.pdf
│   └── 04_PowerSense_Bill_Verification_Guide.pdf
└── README.md              # Project documentation
```

---

## 🚀 Deployment on Streamlit Cloud

1. Upload all 4 core files (`app.py`, `rag_utils.py`, `requirements.txt`, `packages.txt`) along with the `knowledge/` directory to your GitHub repository:
   ```bash
   git add app.py rag_utils.py requirements.txt packages.txt apt_packages.txt README.md knowledge/
   git commit -m "Upgrade PowerSense AI to 10/10 interface with real bill history and PDF verification"
   git push origin main
   ```

2. In [Streamlit Community Cloud](https://share.streamlit.io):
   - Select your repository: `ahmad123-567/PowerSense-Ai`
   - Main file path: `app.py`
   - Under **App Settings → Secrets**, add your Groq API key (optional for basic extraction, required for LLM chat):
     ```toml
     GROQ_API_KEY = "gsk_your_groq_api_key_here"
     ```

3. Deploy! Streamlit Cloud will automatically install Tesseract OCR using `packages.txt` and Python packages using `requirements.txt`.

---

## 💻 Local Quickstart

```bash
# 1. Clone repository
git clone https://github.com/ahmad123-567/PowerSense-Ai.git
cd PowerSense-Ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit app
streamlit run app.py
```

---

## ⚖️ Regulatory Disclaimer

PowerSense AI is an independent informational analysis tool designed to assist Pakistani electricity consumers. It does not replace official tariff determinations by the National Electric Power Regulatory Authority (NEPRA) or official billings issued by distribution companies (DISCOs). All verification findings should be confirmed with the respective utility provider.
