import streamlit as st
import os
import json
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import vision
from google.oauth2 import service_account
from openai import OpenAI
from PIL import Image

# ==========================================
#        CONFIGURATION & AUTHENTICATION
# ==========================================
st.set_page_config(
    page_title="CLV-ID Verification",
    page_icon="🛡️",
    layout="centered"  # centered layout looks more like a tool/app
)

# Custom CSS for centering and styling
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        text-align: center;
        color: #0E1117;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent

# Load Secrets
ENV_PATH = BASE_DIR / "secrets" / ".env"
load_dotenv(dotenv_path=ENV_PATH)
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GCV_KEY_PATH = BASE_DIR / "secrets" / "clv-id-ocr-icpr-53133e7bd944.json"


# ==========================================
#          CLIENT INITIALIZATION
# ==========================================
@st.cache_resource
def get_clients():
    # 1. Setup OpenAI
    # Try getting from Streamlit Secrets first (Cloud), then Environment (Local)
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    
    if not api_key:
        st.error("❌ OpenAI API Key not found.")
        st.stop()
    openai_client = OpenAI(api_key=api_key)

    # 2. Setup Google Cloud Vision
    # STRATEGY: Cloud First, Local Second
    
    # Check if we are in Streamlit Cloud (looking for the TOML section we pasted)
    if "google_credentials" in st.secrets:
        # Create credentials directly from the dictionary in memory
        creds_dict = dict(st.secrets["google_credentials"])
        creds = service_account.Credentials.from_service_account_info(creds_dict)
    
    # Fallback: Check for local file (for when you run on your laptop)
    elif GCV_KEY_PATH.exists():
        creds = service_account.Credentials.from_service_account_file(str(GCV_KEY_PATH))
    
    else:
        st.error("❌ Google Cloud Credentials not found.\n\nOn Local: Check 'secrets/' folder.\nOn Cloud: Check 'Advanced Settings > Secrets'.")
        st.stop()
    
    vision_client = vision.ImageAnnotatorClient(credentials=creds)
    
    return openai_client, vision_client

client_openai, client_vision = get_clients()

# ==========================================
#          HELPER FUNCTIONS
# ==========================================

def get_ocr_text(content):
    """Stage 1: Google Cloud Vision OCR"""
    image = vision.Image(content=content)
    lang_hints = ["ar", "zh", "fr", "hi", "fa", "pt", "ru", "tr", "en"]
    response = client_vision.document_text_detection(
        image=image,
        image_context={"language_hints": lang_hints}
    )
    if response.error.message:
        st.error(f"OCR Error: {response.error.message}")
        return None
    return response.full_text_annotation.text

def simple_candidate_from_text(text):
    """Helper: Heuristic Hint Extraction"""
    name_labels = [
        r"\bname\b", r"\bgiven\b", r"\bsurname\b", r"姓名", r"名字",
        r"نام", r"نام خانوادگی", r"الاسم", r"اللقب", r"прізвище", r"ім'?я",
        r"имя", r"фамилия"
    ]
    stop_labels = [
        r"date", r"birth", r"expiry", r"issue", r"nationality", r"sex", 
        r"id", r"number", r"signature", r"place", r"gender", r"дата", 
        r"номер", r"стать", r"تاريخ", r"شماره"
    ]
    lines = [ln.strip() for ln in re.split(r"\r?\n", text) if ln.strip()]
    collected = []
    collecting = False
    for ln in lines:
        if collecting and any(re.search(p, ln, re.I) for p in stop_labels): break
        if any(re.search(p, ln, re.I) for p in name_labels):
            collecting = True; continue
        if collecting and not re.search(r"\d", ln): collected.append(ln)
    return " ".join(collected[:4])

def extract_semantics(text):
    """Stage 2: OpenAI Semantic Extraction"""
    hint = simple_candidate_from_text(text)
    prompt = (
        "Extract person name fields exactly as printed.\n"
        "Do not infer or correct.\n"
        "Copy the English name verbatim.\n"
        "Produce ASCII transliteration from local_name only.\n"
        "Return JSON: {local_name, english_name, transliteration}.\n\n"
        f"Hint: {hint}\n"
        f"Text: {text}."
    )
    try:
        response = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a helpful assistant that outputs strict JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"OpenAI Error: {e}"); return {}

def levenshtein(a, b):
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1): dp[i][0] = i
    for j in range(len(b) + 1): dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1]

def perform_matching(english, translit):
    a, b = english.strip().lower(), translit.strip().lower()
    if not a or not b: return -1, False, False
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    dist = levenshtein(short, long[:len(short)])
    return dist, dist == 0, dist <= 2

# ==========================================
#            MAIN UI LAYOUT
# ==========================================

# 1. HEADER & LOGO
col_spacer1, col_logo, col_spacer2 = st.columns([1, 2, 1])

with col_logo:
    logo_path = BASE_DIR / "CLV-ID.png"
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
    else:
        # Fallback if no logo file found
        st.markdown("## 🛡️ CLV-ID")

st.markdown("<h1>CLV-ID: Cross-Lingual Verification for AI-Generated ID Forgery Detection</h1>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center; color: #555;'>Gourab Das, Manasa, Dr. Pavan Kumar C, Dr. Raghavendra Ramachandra</h4>", unsafe_allow_html=True)
st.markdown("---")

# 2. UPLOAD SECTION
uploaded_file = st.file_uploader("📂 Upload ID Card (JPG/PNG)", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # Display Input Image
    st.image(uploaded_file, caption="Input Document", use_container_width=True)
    
    # Run Button
    if st.button("🚀 INITIATE VERIFICATION PROTOCOL"):
        
        # --- STAGE 1: OCR ---
        st.markdown("### 1️⃣ Text Extraction (OCR)")
        with st.spinner("Scanning document surface..."):
            image_bytes = uploaded_file.getvalue()
            raw_text = get_ocr_text(image_bytes)
            
            if not raw_text:
                st.error("No text detected. Aborting.")
                st.stop()
            
            # Display Output Step 1
            with st.expander("📄 View Raw Extracted Text", expanded=False):
                st.code(raw_text, language='text')
            st.success("OCR Extraction Complete")
            time.sleep(0.5) # UI smoothing

        # --- STAGE 2: LLM ---
        st.markdown("### 2️⃣ Semantic Parsing (LLM)")
        with st.spinner("Identifying and Transliterating Name Fields..."):
            json_data = extract_semantics(raw_text)
            
            english_name = json_data.get("english_name", "N/A")
            local_name = json_data.get("local_name", "N/A")
            transliteration = json_data.get("transliteration", "N/A")

            # Display Output Step 2
            c1, c2, c3 = st.columns(3)
            c1.metric("Local Script", local_name)
            c2.metric("English Field", english_name)
            c3.metric("Transliteration", transliteration)
            st.success("Semantic Parsing Complete")
            time.sleep(0.5)

        # --- STAGE 3: MATCHING ---
        st.markdown("### 3️⃣ Cross-Lingual Verification")
        with st.spinner("Calculating Semantic Distance..."):
            dist, is_exact, is_close = perform_matching(english_name, transliteration)
            
            # Display Output Step 3
            st.metric("Levenshtein Distance", f"{dist}")

        # --- FINAL VERDICT ---
        st.markdown("---")
        st.subheader("🛡️ Final Verdict")
        
        if is_close: # Distance <= 2
            st.success(f"✅ **BONAFIDE (GENUINE DOCUMENT)**")
            if is_exact:
                st.caption(f"Reason: Perfect Match between English and Local Script (Distance: {dist})")
            else:
                st.caption(f"Reason: Acceptable phonetic variance within tolerance (Distance: {dist})")
        else:
            st.error(f"🚨 **FORGERY DETECTED (ATTACK)**")
            st.caption(f"Reason: Significant semantic mismatch detected (Distance: {dist} > 2)")