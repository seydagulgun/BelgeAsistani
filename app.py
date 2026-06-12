import io
import json
import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv, set_key

ENV_FILE          = Path(__file__).parent / ".env"
UPLOAD_DIR        = Path(__file__).parent / "uploads"
CHAT_FILE         = Path(__file__).parent / "chat_history.json"
MAX_CHARS_PER_DOC = 60_000
MAX_TOTAL_CHARS   = 180_000
GEMINI_URL        = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
FREE_MODELS       = ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"]

UPLOAD_DIR.mkdir(exist_ok=True)
load_dotenv(ENV_FILE)


# ── Dosya okuma ──────────────────────────────────────────────────────────────

def extract_text_from_bytes(data: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        return _from_pdf(data)
    if name.endswith(".docx"):
        return _from_docx(data)
    if name.endswith(".csv"):
        return _from_csv(data)
    if name.endswith((".xlsx", ".xls")):
        return _from_excel(data)
    return data.decode("utf-8", errors="ignore")


def _from_pdf(data):
    import pdfplumber
    text = ""
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text


def _from_docx(data):
    from docx import Document
    return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)


def _from_csv(data):
    import pandas as pd
    return pd.read_csv(io.BytesIO(data)).to_string(index=False)


def _from_excel(data):
    import pandas as pd
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
    return "\n\n".join(
        f"--- {n} ---\n{df.to_string(index=False)}" for n, df in sheets.items()
    )


# ── Dosya yönetimi ───────────────────────────────────────────────────────────

def save_file(uploaded_file) -> str:
    raw_path  = UPLOAD_DIR / uploaded_file.name
    text_path = UPLOAD_DIR / (uploaded_file.name + ".txt")
    data = uploaded_file.read()
    raw_path.write_bytes(data)
    text = extract_text_from_bytes(data, uploaded_file.name)
    text_path.write_text(text, encoding="utf-8")
    return uploaded_file.name


def load_saved_files() -> list[str]:
    return sorted(
        p.name for p in UPLOAD_DIR.iterdir()
        if not p.name.endswith(".txt")
    )


def load_file_text(filename: str) -> str:
    text_path = UPLOAD_DIR / (filename + ".txt")
    if text_path.exists():
        return text_path.read_text(encoding="utf-8")
    raw_path = UPLOAD_DIR / filename
    text = extract_text_from_bytes(raw_path.read_bytes(), filename)
    text_path.write_text(text, encoding="utf-8")
    return text


def delete_file(filename: str):
    for p in [UPLOAD_DIR / filename, UPLOAD_DIR / (filename + ".txt")]:
        p.unlink(missing_ok=True)


# ── Global sohbet geçmişi ────────────────────────────────────────────────────

def load_chat() -> list:
    if CHAT_FILE.exists():
        return json.loads(CHAT_FILE.read_text(encoding="utf-8"))
    return []


def save_chat(messages: list):
    CHAT_FILE.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Gemini REST ──────────────────────────────────────────────────────────────

def list_available_models(api_key: str) -> list[str]:
    r = requests.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": api_key}, timeout=10,
    )
    if r.status_code != 200:
        return []
    names = [m["name"].replace("models/", "") for m in r.json().get("models", [])]
    return [n for n in names if "flash" in n or "pro" in n]


def build_system_prompt(selected_docs: list[str]) -> str:
    parts = []
    total = 0
    for fname in selected_docs:
        text = load_file_text(fname)
        chunk = text[:MAX_CHARS_PER_DOC]
        if len(text) > MAX_CHARS_PER_DOC:
            chunk += "\n[...bu belge karakter sınırı nedeniyle kesildi]"
        parts.append(f"=== BELGE: {fname} ===\n{chunk}\n=== BELGE SONU: {fname} ===")
        total += len(chunk)
        if total >= MAX_TOTAL_CHARS:
            parts.append("[...toplam karakter sınırına ulaşıldı, kalan belgeler dahil edilmedi]")
            break

    return (
        "Sen bir belge asistanısın. Sana verilen belgelerden YALNIZCA bu belgelerdeki bilgilere "
        "dayanarak sorulara cevap ver. Belgelerde olmayan bilgileri uydurma. "
        "Cevabını hangi belgeden aldığını her zaman belirt.\n\n"
        + "\n\n".join(parts)
    )


def get_response(api_key: str, model: str, system_prompt: str, messages: list) -> str:
    contents = [
        {"role": "user" if m["role"] == "user" else "model",
         "parts": [{"text": m["content"]}]}
        for m in messages
    ]
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 2048},
    }
    r = requests.post(
        GEMINI_URL.format(model=model),
        params={"key": api_key},
        json=payload,
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"API Hatası {r.status_code}: {r.json().get('error', {}).get('message', r.text)}"
        )
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


# ── UI ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Belge Asistanı", page_icon="📚", layout="wide")
st.title("📚 Belge Asistanı")
st.caption("Belgelerini yükle, seçmek istediklerine tik at ve sorularını sor.")

if "messages" not in st.session_state:
    st.session_state["messages"] = load_chat()
if "selected_docs" not in st.session_state:
    st.session_state["selected_docs"] = set()

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Ayarlar")

    saved_key = os.getenv("GEMINI_API_KEY", "")
    api_key = st.text_input(
        "Gemini API Key", type="password",
        value=st.session_state.get("api_key", saved_key),
        help="aistudio.google.com/app/apikey adresinden ücretsiz alabilirsin",
    )
    if api_key and api_key != st.session_state.get("api_key"):
        st.session_state["api_key"] = api_key
        st.session_state.pop("available_models", None)
        if api_key != saved_key:
            set_key(str(ENV_FILE), "GEMINI_API_KEY", api_key)
    elif api_key:
        st.session_state["api_key"] = api_key

    if st.session_state.get("api_key"):
        if "available_models" not in st.session_state:
            with st.spinner("Modeller yükleniyor..."):
                found = list_available_models(st.session_state["api_key"])
                st.session_state["available_models"] = found if found else FREE_MODELS
        models = st.session_state["available_models"]
        default_idx = next((i for i, m in enumerate(models) if "1.5-flash" in m and "8b" not in m), 0)
        st.session_state["model"] = st.selectbox("Model", models, index=default_idx)

    st.divider()
    st.subheader("📁 Belge Yükle")
    uploaded_files = st.file_uploader(
        "Belge ekle (çoklu seçim desteklenir)",
        type=["pdf", "docx", "csv", "xlsx", "xls", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        for f in uploaded_files:
            with st.spinner(f"{f.name} kaydediliyor..."):
                name = save_file(f)
                st.session_state["selected_docs"].add(name)
        st.rerun()

    st.divider()
    saved_files = load_saved_files()

    if saved_files:
        st.subheader("📋 Belgeler")

        col_a, col_b = st.columns(2)
        if col_a.button("Tümünü Seç", use_container_width=True):
            st.session_state["selected_docs"] = set(saved_files)
            st.rerun()
        if col_b.button("Seçimi Kaldır", use_container_width=True):
            st.session_state["selected_docs"] = set()
            st.rerun()

        st.markdown("")
        for fname in saved_files:
            col1, col2 = st.columns([6, 1])
            checked = fname in st.session_state["selected_docs"]
            new_val = col1.checkbox(fname, value=checked, key=f"chk_{fname}")
            if new_val != checked:
                if new_val:
                    st.session_state["selected_docs"].add(fname)
                else:
                    st.session_state["selected_docs"].discard(fname)
                st.rerun()
            if col2.button("🗑", key=f"del_{fname}", help=f"{fname} sil"):
                delete_file(fname)
                st.session_state["selected_docs"].discard(fname)
                st.rerun()

    st.divider()
    if st.button("🗑 Sohbeti Temizle", use_container_width=True):
        st.session_state["messages"] = []
        save_chat([])
        st.rerun()


# ── Ana Alan ─────────────────────────────────────────────────────────────────

if not st.session_state.get("api_key"):
    st.info("Sol panele Gemini API Key girin. [Ücretsiz al →](https://aistudio.google.com/app/apikey)")
    st.stop()

selected = sorted(st.session_state["selected_docs"])

if not selected:
    if not load_saved_files():
        st.info("Başlamak için sol panelden bir belge yükleyin.")
    else:
        st.info("Sol panelden taramak istediğiniz belgelere tik atın.")
    st.stop()

# Aktif belgeler
st.markdown(
    "**Aktif Belgeler:** " + "  ".join(f"`{d}`" for d in selected),
    help="Sorular yalnızca bu belgelerde aranacak."
)
st.divider()

# Sohbet geçmişi
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Yeni soru
if prompt := st.chat_input("Seçili belgeler hakkında soru sor..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"{len(selected)} belge taranıyor..."):
            try:
                answer = get_response(
                    api_key=st.session_state["api_key"],
                    model=st.session_state.get("model", FREE_MODELS[0]),
                    system_prompt=build_system_prompt(selected),
                    messages=st.session_state["messages"],
                )
            except RuntimeError as e:
                answer = f"⚠️ {e}"
        st.markdown(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})
    save_chat(st.session_state["messages"])
