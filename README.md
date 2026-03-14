# 🔍 AI DDR Generator
### Applied AI Builder Assignment

An AI-powered system that reads a building **Inspection Report** and a **Thermal Report** and automatically generates a professional, client-ready **Detailed Diagnostic Report (DDR)**.

---

## 🚀 Live Demo
"coming soon"
> Upload both PDFs → Get a structured DDR in under a minute.

---

## 🧠 How It Works

The system runs a **two-model pipeline** to avoid payload limits and quota issues:

```
Thermal PDF ──► extract images ──► Gemini 1.5 Flash (captions 10 sampled images)
                                                          │
Inspection PDF ──► extract text ──────────────────────── ▼
Thermal PDF    ──► extract text ──────────────► Groq Llama 3.1 8B ──► DDR JSON
```

**Step 1 — Gemini 1.5 Flash (Vision)**
- Extracts images from the thermal PDF
- Samples 10 images evenly across the document (handles PDFs with 5000+ images)
- Sends one image at a time to Gemini → gets a short technical caption per image
- Never hits quota limits because each call is tiny (~200 tokens)

**Step 2 — Groq Llama 3.1 8B (Text)**
- Receives inspection text + thermal text + image captions (all text, no images)
- Generates the full structured DDR as JSON
- Fast, free, and zero payload issues

---

## 📋 Output Structure

The generated DDR contains all 7 required sections:

| # | Section |
|---|---------|
| 1 | Property Issue Summary |
| 2 | Area-wise Observations (with thermal images) |
| 3 | Probable Root Cause |
| 4 | Severity Assessment (with reasoning) |
| 5 | Recommended Actions |
| 6 | Additional Notes |
| 7 | Missing or Unclear Information |

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/ai-ddr-generator.git
cd ai-ddr-generator
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add API keys
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

Get your keys here:
- Gemini → [Google AI Studio](https://aistudio.google.com/app/apikey) (free)
- Groq → [Groq Console](https://console.groq.com/keys) (free)

### 4. Run the app
```bash
streamlit run app.py
```

---

## 🗂️ Project Structure

```
ai-ddr-generator/
├── app.py               # Main Streamlit app
├── requirements.txt     # Python dependencies
├── .env                 # API keys (never commit this)
├── .gitignore           # Should include .env
└── README.md
```

---

## ⚠️ Limitations

- **Image assignment** — only 10 images are sampled from the thermal PDF, so many report sections show "Image Not Available" on very large documents
- **Scanned PDFs** — if the inspection report is a scanned image with no selectable text, text extraction degrades
- **Gemini free tier** — has a per-minute token quota; very large documents processed rapidly may hit rate limits

---

## 🔮 Future Improvements

- **Smarter image selection** — use clustering or similarity search to pick the most unique and relevant thermal images rather than evenly spaced ones
- **Direct vision DDR** — upgrade to a paid vision model (GPT-4o / Llama 4 Maverick) to send images directly into the DDR generation step for better accuracy
- **PDF export** — generate a formatted downloadable PDF report instead of just JSON
- **Multi-document support** — handle more than two input documents

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit |
| PDF parsing | PyMuPDF (fitz) |
| Image captioning | Gemini 1.5 Flash |
| DDR generation | Groq — Llama 3.1 8B Instant |
| Language | Python 3.10+ |

---

## 📝 .gitignore reminder

Make sure your `.env` is never committed:
```
.env
__pycache__/
*.pyc
.streamlit/
```
