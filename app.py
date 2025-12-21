import streamlit as st
import os
import json
import re
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
    layout="centered"
)

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

ENV_PATH = BASE_DIR / "secrets" / ".env"
load_dotenv(dotenv_path=ENV_PATH)

<<<<<<< HEAD
GCV_KEY_PATH = BASE_DIR / "secrets" / "clv-id-ocr-icpr-53133e7bd944.json"

=======
>>>>>>> 41aa07ce7220e5720c29afdb16f09b5af5d4c406
# ==========================================
#          CLIENT INITIALIZATION
# ==========================================
@st.cache_resource
def get_clients():
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not api_key:
        st.error("❌ OpenAI API Key not found.")
        st.stop()
    openai_client = OpenAI(api_key=api_key)

    if "google_credentials" in st.secrets:
        creds_dict = dict(st.secrets["google_credentials"])
        creds = service_account.Credentials.from_service_account_info(creds_dict)
    elif GCV_KEY_PATH.exists():
        creds = service_account.Credentials.from_service_account_file(str(GCV_KEY_PATH))
    else:
        st.error("❌ Google Cloud Credentials not found.")
        st.stop()

    vision_client = vision.ImageAnnotatorClient(credentials=creds)
    return openai_client, vision_client

client_openai, client_vision = get_clients()

# ==========================================
#          HELPER FUNCTIONS
# ==========================================
def get_ocr_text(content):
    image = vision.Image(content=content)
    lang_hints = [
        "ar", "zh", "fr", "hi", "fa", "pt", "ru", "tr", "en",
        "bn", "kn", "or", "ta", "te", "ml", "gu", "mr", "pa"
    ]
    response = client_vision.document_text_detection(
        image=image,
        image_context={"language_hints": lang_hints}
    )
    if response.error.message:
        st.error(f"OCR Error: {response.error.message}")
        return None
    return response.full_text_annotation.text


def simple_candidate_from_text(text):
    name_labels = [
        r"\bname\b", r"\bgiven\b", r"\bsurname\b", r"姓名", r"名字",
        r"نام", r"نام خانوادگی", r"الاسم", r"اللقب", r"прізвище",
        r"ім'?я", r"имя", r"фамилия"
    ]
    stop_labels = [
        r"date", r"birth", r"expiry", r"issue", r"nationality", r"sex",
        r"id", r"number", r"signature", r"place", r"gender", r"дата",
        r"номер", r"стать", r"تاريخ", r"شماره"
    ]

    lines = [ln.strip() for ln in re.split(r"\r?\n", text) if ln.strip()]
    collected, collecting = [], False

    for ln in lines:
        if collecting and any(re.search(p, ln, re.I) for p in stop_labels):
            break
        if any(re.search(p, ln, re.I) for p in name_labels):
            collecting = True
            continue
        if collecting and not re.search(r"\d", ln):
            collected.append(ln)

    return " ".join(collected[:4])


def extract_semantics(text):
    hint = simple_candidate_from_text(text)
    prompt = (
        "Extract person name fields exactly as printed.\n"
        "Do not infer or correct.\n"
        "Copy the English name verbatim.\n"
        "Produce ASCII transliteration from local_name only.\n"
        "Return JSON: {local_name, english_name, transliteration}.\n\n"
        f"Hint: {hint}\n"
        f"Text: {text}"
    )

<<<<<<< HEAD
    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)
=======
    try:
        response = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You output strict JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"OpenAI Error: {e}")
        return {}
>>>>>>> 41aa07ce7220e5720c29afdb16f09b5af5d4c406


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[-1][-1]


def perform_matching(english, translit, threshold):
    a, b = english.strip().lower(), translit.strip().lower()
    if not a or not b:
        return -1, False, False

    short, long = (a, b) if len(a) <= len(b) else (b, a)
    dist = levenshtein(short, long[:len(short)])

    return dist, dist == 0, dist <= threshold

# ==========================================
#            MAIN UI
# ==========================================
col_spacer1, col_logo, col_spacer2 = st.columns([1, 2, 1])
with col_logo:
    logo_path = BASE_DIR / "CLV-ID.png"
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
    else:
        st.markdown("## 🛡️ CLV-ID")

<<<<<<< HEAD
st.markdown(
    "<h1>CLV-ID: Cross-Lingual Verification for AI-Generated ID Forgery Detection</h1>",
    unsafe_allow_html=True
)
st.markdown("---")

uploaded_file = st.file_uploader(
    "📂 Upload ID Card (JPG/PNG)",
    type=['jpg', 'png', 'jpeg']
)
=======
st.markdown("<h1>CLV-ID: Cross-Lingual Verification for AI-Generated ID Forgery Detection</h1>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("📂 Upload ID Card (JPG/PNG)", type=['jpg', 'png', 'jpeg'])
>>>>>>> 41aa07ce7220e5720c29afdb16f09b5af5d4c406

if uploaded_file:
    st.image(uploaded_file, caption="Input Document", use_container_width=True)

    if st.button("🚀 INITIATE VERIFICATION PROTOCOL"):

<<<<<<< HEAD
        # -------- OCR --------
=======
>>>>>>> 41aa07ce7220e5720c29afdb16f09b5af5d4c406
        st.markdown("### 1️⃣ Text Extraction (OCR)")
        with st.spinner("Scanning document surface..."):
            raw_text = get_ocr_text(uploaded_file.getvalue())
            if not raw_text:
                st.stop()
<<<<<<< HEAD
            st.session_state["raw_text"] = raw_text
            with st.expander("📄 View Raw Extracted Text"):
                st.code(raw_text)
        st.success("OCR Extraction Complete")

        # -------- LLM --------
        st.markdown("### 2️⃣ Semantic Parsing (LLM)")
        with st.spinner("Parsing name fields..."):
            data = extract_semantics(raw_text)

            st.session_state["local_name"] = data.get("local_name", "N/A")
            st.session_state["english_name"] = data.get("english_name", "N/A")
            st.session_state["transliteration"] = data.get("transliteration", "N/A")

            c1, c2, c3 = st.columns(3)
            c1.metric("Local Script", st.session_state["local_name"])
            c2.metric("English Field", st.session_state["english_name"])
            c3.metric("Transliteration", st.session_state["transliteration"])
        st.success("Semantic Parsing Complete")

# -------- Slider (NO API CALLS) --------
if "english_name" in st.session_state and "transliteration" in st.session_state:

    lev_threshold = st.slider(
        "🔧 Set Levenshtein Distance Tolerance",
        min_value=0,
        max_value=10,
        value=2,
        help="Maximum allowed edit distance"
    )

    st.markdown("### 3️⃣ Cross-Lingual Verification")
    dist, is_exact, is_close = perform_matching(
        st.session_state["english_name"],
        st.session_state["transliteration"],
        lev_threshold
    )

    st.metric("Levenshtein Distance", dist)

    st.markdown("---")
    st.subheader("🛡️ Final Verdict")

    if is_close:
        st.success("✅ **BONAFIDE (GENUINE DOCUMENT)**")
        if is_exact:
            st.caption("Perfect character-level match")
        else:
            st.caption(f"Within tolerance (threshold = {lev_threshold})")
    else:
        st.error("🚨 **FORGERY DETECTED (ATTACK)**")
        st.caption(f"Distance {dist} > threshold {lev_threshold}")
=======
            with st.expander("📄 View Raw Extracted Text"):
                st.code(raw_text)
            st.success("OCR Extraction Complete")

        st.markdown("### 2️⃣ Semantic Parsing (LLM)")
        with st.spinner("Parsing name fields..."):
            data = extract_semantics(raw_text)
            local_name = data.get("local_name", "N/A")
            english_name = data.get("english_name", "N/A")
            transliteration = data.get("transliteration", "N/A")

            c1, c2, c3 = st.columns(3)
            c1.metric("Local Script", local_name)
            c2.metric("English Field", english_name)
            c3.metric("Transliteration", transliteration)
            st.success("Semantic Parsing Complete")

        # 🔧 USER INPUT HERE
        lev_threshold = st.slider(
            "🔧 Set Levenshtein Distance Tolerance",
            min_value=0,
            max_value=10,
            value=2,
            help="Maximum allowed edit distance"
        )

        st.markdown("### 3️⃣ Cross-Lingual Verification")
        with st.spinner("Computing semantic distance..."):
            dist, is_exact, is_close = perform_matching(
                english_name,
                transliteration,
                lev_threshold
            )
            st.metric("Levenshtein Distance", dist)

        st.markdown("---")
        st.subheader("🛡️ Final Verdict")

        if is_close:
            st.success("✅ **BONAFIDE (GENUINE DOCUMENT)**")
            if is_exact:
                st.caption("Perfect character-level match")
            else:
                st.caption(f"Within tolerance (threshold = {lev_threshold})")
        else:
            st.error("🚨 **FORGERY DETECTED (ATTACK)**")
            st.caption(f"Distance {dist} > threshold {lev_threshold}")
>>>>>>> 41aa07ce7220e5720c29afdb16f09b5af5d4c406
