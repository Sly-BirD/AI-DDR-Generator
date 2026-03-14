import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai
import requests
import io
import json
import os
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI DDR Generator", layout="wide")
st.title("🔍 AI DDR Report Generator - Applied AI Builder Assignment")
st.markdown("**Inspection Report + Thermal Report → Professional DDR** (images placed automatically)")

# ==============================================================
# TWO-MODEL PIPELINE
# ┌─────────────────────────────────────────────────────────┐
# │ Step 1 — Gemini 1.5 Flash (image captioning)            │
# │   • Samples ~10 images evenly from thermal PDF          │
# │   • Sends ONE image at a time → never hits quota        │
# │   • Returns a short text caption per image              │
# ├─────────────────────────────────────────────────────────┤
# │ Step 2 — Groq Llama 3.1 8B (DDR generation)            │
# │   • Receives inspection text + thermal text +           │
# │     image captions (all text, no images)                │
# │   • No payload issues ever                              │
# │   • Generates full structured DDR JSON                  │
# └─────────────────────────────────────────────────────────┘
# ==============================================================

# ── API KEYS ──────────────────────────────────────────────────
gemini_key = os.getenv("GEMINI_API_KEY")
groq_key   = os.getenv("GROQ_API_KEY")

if not gemini_key:
    st.error("❌ GEMINI_API_KEY not found in .env file.")
    st.stop()
if not groq_key:
    st.error("❌ GROQ_API_KEY not found in .env file.")
    st.stop()

# ── GEMINI SETUP (image captioning + thermal verification) ──
genai.configure(api_key=gemini_key)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ── GROQ SETUP (text-only DDR generation — no payload issues) ──
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_HEADERS = {
    "Authorization": f"Bearer {groq_key}",
    "Content-Type": "application/json",
}

MAX_SAMPLE_IMAGES = 10   # evenly spaced across the PDF

# ============== ROTATING SPINNER ==============
SPINNER_MESSAGES = [
    "🔬 Extracting knowledge from your PDFs...",
    "🧠 Teaching Gemini what thermal looks like...",
    "🌡️ Decoding thermal mysteries one image at a time...",
    "🏗️ Consulting with imaginary building inspectors...",
    "📐 Measuring things that can't be measured...",
    "🚀 Groq going brrr on the DDR...",
    "🔍 Finding cracks in more ways than one...",
    "📊 Turning image captions into professional opinions...",
    "🛠️ Almost there... probably...",
    "☕ If this takes long, go grab a coffee. We'll wait.",
    "🧱 Building your report, brick by digital brick...",
    "🌡️ Gemini is squinting at your thermal images...",
    "📋 Formatting jargon into readable English...",
    "🤝 Gemini + Groq tag-teaming your report...",
    "✅ Dotting i's and crossing load-bearing beams...",
]

def run_with_spinner(placeholder, fn, *args, **kwargs):
    """Run fn in a background thread; cycle spinner messages on the main thread."""
    result_box = {}

    def worker():
        try:
            result_box["result"] = fn(*args, **kwargs)
        except Exception as e:
            result_box["error"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    i = 0
    while t.is_alive():
        placeholder.info(SPINNER_MESSAGES[i % len(SPINNER_MESSAGES)])
        time.sleep(2.5)
        i += 1
    t.join()
    placeholder.empty()
    if "error" in result_box:
        raise result_box["error"]
    return result_box["result"]

# ============== FILE VALIDATION (stage 1 & 2 only — fast, no content scan) ==============
FUNNY_WRONG_FILE_MESSAGES = [
    "🙈 Bro uploaded a {ext} file. Are you inspecting a building or submitting homework?",
    "😭 A {ext} file? This tool eats PDFs for breakfast, not {ext} snacks.",
    "🤦 That's a {ext} file. Our AI has standards. Please feed it a PDF.",
    "🧐 Interesting. A {ext} file. Bold choice. Wrong choice. Try a PDF.",
    "🚫 {ext} detected. The AI has officially gone on strike until you upload a PDF.",
    "💀 You uploaded a {ext}. The building inspector ghost is disappointed in you.",
]

def validate_pdf(uploaded_file):
    """Stage 1: extension check. Stage 2: magic bytes check."""
    if uploaded_file is None:
        return True, None
    name = uploaded_file.name.lower()
    if not name.endswith(".pdf"):
        ext = name.rsplit(".", 1)[-1].upper() if "." in name else "MYSTERY"
        return False, random.choice(FUNNY_WRONG_FILE_MESSAGES).format(ext=ext)
    header = uploaded_file.read(5)
    uploaded_file.seek(0)
    if header != b"%PDF-":
        return False, "🤡 That file LIES. It says .pdf but it is not a PDF. Nice try."
    return True, None

def scan_pdf_for_keywords(pdf_bytes, keywords, max_pages=2):
    """Scan first max_pages for keywords (case-insensitive). Fast — text only."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(min(max_pages, len(doc))):
            text = doc[page_num].get_text("text").lower()
            if any(kw.lower() in text for kw in keywords):
                return True
    except Exception:
        pass
    return False

# ============== PDF EXTRACTION (cached) ==============
@st.cache_data(show_spinner=False)
def extract_text_and_images(pdf_bytes, is_thermal=False):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    images = []
    for page in doc:
        text += page.get_text("text") + "\n"
        if is_thermal:
            for img in page.get_images(full=True):
                base = doc.extract_image(img[0])
                if base and "image" in base:
                    images.append(base["image"])
    return text.strip(), images

# ============== SMART IMAGE SAMPLING ==============
def sample_images_evenly(images, n=MAX_SAMPLE_IMAGES):
    """Pick n images evenly spaced across the full list."""
    if len(images) <= n:
        return list(range(len(images))), images
    step = len(images) / n
    indices = [int(i * step) for i in range(n)]
    return indices, [images[i] for i in indices]

# ============== STEP 1: GEMINI IMAGE CAPTIONING ==============
def caption_image_with_gemini(img_bytes, img_index):
    pil = Image.open(io.BytesIO(img_bytes))
    pil.thumbnail((512, 512), Image.Resampling.LANCZOS)
    try:
        response = gemini_model.generate_content([
            pil,
            "You are analyzing a thermal image from a building inspection report. "
            "In 2-3 sentences, describe: what area of the building this shows, "
            "any visible thermal anomalies (hot/cold spots), and what issue it might indicate. "
            "Be concise and technical."
        ])
        return img_index, response.text.strip()
    except Exception as e:
        return img_index, f"[Caption unavailable: {str(e)}]"

def caption_all_images(sampled_images, original_indices):
    results = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(caption_image_with_gemini, img, orig_idx): orig_idx
            for img, orig_idx in zip(sampled_images, original_indices)
        }
        for f in as_completed(futures):
            idx, caption = f.result()
            results[idx] = caption
    return [(f"thermal_{i}", results[i]) for i in sorted(results.keys())]

# ============== STEP 2: GROQ DDR GENERATION ==============
def generate_ddr_with_groq(insp_text, therm_text, image_captions):
    captions_block = ""
    if image_captions:
        captions_block = "\n\nTHERMAL IMAGE ANALYSIS (AI-captioned):\n"
        for img_id, caption in image_captions:
            captions_block += f"\n[{img_id}]: {caption}\n"

    prompt = f"""You are an expert building engineer. Create a client-ready DDR.

INSPECTION REPORT TEXT:
{insp_text}

THERMAL REPORT TEXT:
{therm_text}
{captions_block}

You MUST return **only** valid JSON. No explanation, no markdown, no extra text at all.

Exact structure:

{{
  "property_issue_summary": "2-3 sentence overview",
  "area_wise_observations": [
    {{
      "area": "e.g. Hall / Master Bedroom",
      "observations": "bullet points",
      "image_ids": ["thermal_0", null, "thermal_3"]
    }}
  ],
  "probable_root_cause": "...",
  "severity_assessment": "High/Medium/Low + reasoning",
  "recommended_actions": "bullet list",
  "additional_notes": "...",
  "missing_or_unclear": "Not Available or conflict details"
}}

Rules (obey strictly):
- Only use facts from the input
- If missing → exactly "Not Available"
- If conflict → "Conflict: Inspection says X, Thermal shows Y"
- Simple client-friendly language
- Assign image_ids only when caption matches the area
- No duplicates

Respond with NOTHING but the JSON object."""

    data = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},  # forces JSON — no narrative bleed
    }

    resp = requests.post(GROQ_URL, headers=GROQ_HEADERS, json=data, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

# ============== UPLOADS ==============
col1, col2 = st.columns(2)
with col1:
    insp_file = st.file_uploader("📄 Inspection Report (Sample Report)", type=["pdf"])
with col2:
    therm_file = st.file_uploader("🔥 Thermal Report (Thermal Images Document)", type=["pdf"])

insp_ok, insp_err = validate_pdf(insp_file)
therm_ok, therm_err = validate_pdf(therm_file)
if not insp_ok:
    st.error(insp_err)
if not therm_ok:
    st.error(therm_err)

# ── Stage 3: Inspection keyword scan (fast, text-only, no API call) ──
if insp_ok and insp_file:
    insp_bytes_check = insp_file.read()
    insp_file.seek(0)
    if not scan_pdf_for_keywords(insp_bytes_check, ["inspection"], max_pages=2):
        insp_ok = False
        st.error(
            "🧐 This doesn't look like an Inspection Report — "
            "couldn't find 'inspection' in the first 2 pages. "
            "Are you sure you uploaded the right file?"
        )

# ============== GENERATE ==============
can_generate = insp_file and therm_file and insp_ok and therm_ok

if st.button("🚀 Generate DDR Report", type="primary", disabled=not can_generate):

    insp_bytes = insp_file.read()
    therm_bytes = therm_file.read()

    spinner_placeholder = st.empty()

    # ── EXTRACTION ──
    spinner_placeholder.info("📂 Extracting text and images from PDFs...")
    insp_text, _           = extract_text_and_images(insp_bytes, is_thermal=False)
    therm_text, therm_images = extract_text_and_images(therm_bytes, is_thermal=True)

    # ── STAGE 3: THERMAL CONTENT VALIDATION ──────────────────────────────────
    # Three-tier check:
    #   Tier A — thermal keywords in extracted text → pass immediately
    #   Tier B — image-heavy but no text (scanned PDF) → ask Gemini to verify
    #             one sample image actually looks thermal/building-related
    #   Tier C — no text, no images → reject immediately
    thermal_keywords     = ["thermal", "infrared", "ir ", "temperature", "heat",
                            "cold spot", "hot spot", "emissivity", "celsius", "fahrenheit"]
    thermal_text_ok      = any(kw in therm_text.lower() for kw in thermal_keywords)
    thermal_has_images   = len(therm_images) > 0

    if not thermal_text_ok and not thermal_has_images:
        # Tier C — completely wrong file
        spinner_placeholder.empty()
        st.error(
            "🌡️ **Wrong file for Thermal Report.** "
            "No thermal text and no images found. Wrong file maybe?"
        )
        st.stop()

    elif not thermal_text_ok and thermal_has_images:
        # Tier B — scanned/image-only PDF → verify with Gemini
        spinner_placeholder.info("🔍 Verifying thermal report via Gemini (takes ~3s)...")

        def verify_thermal_image():
            sample = therm_images[len(therm_images) // 2]  # pick middle image
            pil = Image.open(io.BytesIO(sample))
            pil.thumbnail((384, 384), Image.Resampling.LANCZOS)
            response = gemini_model.generate_content([
                pil,
                "Look at this image carefully. "
                "Is this a thermal infrared image OR a building/construction inspection image? "
                "Reply with YES if it is either of those. Reply NO if it is something else entirely "
                "(e.g. a document, a photo of people, a graph, a random picture). "
                "Reply with ONE word only: YES or NO."
            ])
            return "YES" in response.text.strip().upper()

        is_valid_thermal = run_with_spinner(spinner_placeholder, verify_thermal_image)

        if not is_valid_thermal:
            st.error(
                "🌡️ **Wrong file for Thermal Report.** "
                " it doesn't look like a thermal or building inspection image. "
                "Please upload the correct Thermal Report."
            )
            st.stop()
    # Tier A falls through here — no extra check needed

    # ── IMAGE SAMPLING ──
    total_images = len(therm_images)
    sampled_indices, sampled_images = sample_images_evenly(therm_images, MAX_SAMPLE_IMAGES)

    if total_images > MAX_SAMPLE_IMAGES:
        st.info(f"🖼️ Found {total_images} thermal images — sampling {len(sampled_images)} evenly spaced for analysis.")

    image_dict = {
        f"thermal_{i}": Image.open(io.BytesIO(b))
        for i, b in enumerate(therm_images)
    }

    # ── STEP 1: CAPTION IMAGES WITH GEMINI ──
    def run_captioning():
        return caption_all_images(sampled_images, sampled_indices)

    spinner_placeholder.info("🌡️ Step 1/2 — Gemini is analyzing thermal images...")
    image_captions = run_with_spinner(spinner_placeholder, run_captioning)

    # ── STEP 2: GENERATE DDR WITH GROQ ──
    def run_groq():
        return generate_ddr_with_groq(insp_text, therm_text, image_captions)

    spinner_placeholder.info("📋 Step 2/2 — Groq is generating your DDR...")
    raw = run_with_spinner(spinner_placeholder, run_groq)

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        report = json.loads(raw)
    except json.JSONDecodeError:
        st.error("⚠️ Model returned invalid JSON. Raw output shown below for debugging:")
        st.code(raw)
        st.stop()

    # ============== RENDER ==============
    st.success("✅ Professional DDR Generated!")
    st.caption(f"Analyzed via: Gemini 1.5 Flash (image captioning) + Groq {GROQ_MODEL} (report generation)")

    st.header("1. Property Issue Summary")
    st.markdown(report.get("property_issue_summary", "Not Available"))

    st.header("2. Area-wise Observations")
    for obs in report.get("area_wise_observations", []):
        st.subheader(obs.get("area", "Unnamed Area"))
        obs_text = obs.get("observations", [])
        if isinstance(obs_text, list):
            st.markdown("\n".join([f"- {item}" for item in obs_text]))
        else:
            st.markdown(obs_text)
        for img_id in obs.get("image_ids", []):
            if img_id and img_id in image_dict:
                st.image(image_dict[img_id], use_container_width=True, caption=f"📷 {img_id}")
            elif img_id is None:
                st.info("🖼️ Image Not Available")

    st.header("3. Probable Root Cause")
    st.markdown(report.get("probable_root_cause", "Not Available"))

    st.header("4. Severity Assessment (with reasoning)")
    st.markdown(report.get("severity_assessment", "Not Available"))

    st.header("5. Recommended Actions")
    st.markdown(report.get("recommended_actions", "Not Available"))

    st.header("6. Additional Notes")
    st.markdown(report.get("additional_notes", "None"))

    st.header("7. Missing or Unclear Information")
    st.markdown(report.get("missing_or_unclear", "None"))

    st.divider()
    st.download_button("📥 Download Report as JSON", json.dumps(report, indent=2), "DDR_Report.json")

elif not can_generate and (insp_file or therm_file):
    if insp_ok and therm_ok:
        st.info("📂 Upload both files to enable report generation.")

st.caption("⚡ Gemini 1.5 Flash (vision) + Groq Llama 3.1 8B (reasoning) • smart image sampling • cached extraction")