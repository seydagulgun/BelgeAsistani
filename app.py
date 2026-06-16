import io
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv, set_key

ENV_FILE          = Path(__file__).parent / ".env"
UPLOAD_DIR        = Path(__file__).parent / "uploads"
SESSIONS_DIR      = Path(__file__).parent / "sessions"
MAX_CHARS_PER_DOC = 60_000
MAX_TOTAL_CHARS   = 180_000
GEMINI_URL        = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
FREE_MODELS       = ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"]

UPLOAD_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)
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
    import fitz
    doc = fitz.open(stream=data, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


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


# ── Sohbet oturumları ────────────────────────────────────────────────────────

def list_sessions() -> list[dict]:
    sessions = []
    for f in SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            sessions.append({
                "id": data["id"],
                "title": data.get("title", "Yeni Sohbet"),
                "created_at": data.get("created_at", ""),
            })
        except Exception:
            pass
    return sorted(sessions, key=lambda x: x["created_at"], reverse=True)


def load_session(session_id: str) -> dict | None:
    p = SESSIONS_DIR / f"{session_id}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def save_session(session: dict):
    (SESSIONS_DIR / f"{session['id']}.json").write_text(
        json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def delete_session(session_id: str):
    (SESSIONS_DIR / f"{session_id}.json").unlink(missing_ok=True)


def create_session() -> dict:
    now = datetime.now()
    session = {
        "id": uuid.uuid4().hex[:10],
        "title": now.strftime("%d %b, %H:%M"),
        "created_at": now.isoformat(),
        "messages": [],
    }
    save_session(session)
    return session


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


def build_system_prompt(selected_docs: list[str]) -> str | None:
    if not selected_docs:
        return None
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
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 2048},
    }
    if system_prompt:
        payload["system_instruction"] = {"parts": [{"text": system_prompt}]}
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

# Session state init
if "active_session_id" not in st.session_state:
    existing = list_sessions()
    if existing:
        st.session_state["active_session_id"] = existing[0]["id"]
    else:
        st.session_state["active_session_id"] = create_session()["id"]

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

    # ── Sohbet listesi ────────────────────────────────────────────────────────
    st.subheader("💬 Sohbetler")

    if st.button("➕ Yeni Sohbet", use_container_width=True, type="primary"):
        new = create_session()
        st.session_state["active_session_id"] = new["id"]
        st.rerun()

    sessions = list_sessions()
    active_id = st.session_state["active_session_id"]

    for s in sessions:
        col1, col2 = st.columns([5, 1])
        is_active = s["id"] == active_id
        btn_type = "primary" if is_active else "secondary"
        if col1.button(s["title"], key=f"sess_{s['id']}", use_container_width=True, type=btn_type):
            st.session_state["active_session_id"] = s["id"]
            st.rerun()
        if col2.button("🗑", key=f"delsess_{s['id']}"):
            delete_session(s["id"])
            remaining = [x for x in sessions if x["id"] != s["id"]]
            if remaining:
                st.session_state["active_session_id"] = remaining[0]["id"]
            else:
                st.session_state["active_session_id"] = create_session()["id"]
            st.rerun()

    st.divider()

    # ── Belge yönetimi ────────────────────────────────────────────────────────
    st.subheader("📁 Belgeler")

    uploaded_files = st.file_uploader(
        "Belge ekle (çoklu seçim desteklenir)",
        type=["pdf", "docx", "csv", "xlsx", "xls", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        for f in uploaded_files:
            with st.spinner(f"{f.name} kaydediliyor..."):
                name = save_file(f)
                st.session_state[f"chk_{name}"] = True
        st.rerun()

    saved_files = load_saved_files()
    if saved_files:
        col_a, col_b = st.columns(2)
        if col_a.button("Tümünü Seç", use_container_width=True):
            for f in saved_files:
                st.session_state[f"chk_{f}"] = True
            st.rerun()
        if col_b.button("Seçimi Kaldır", use_container_width=True):
            for f in saved_files:
                st.session_state[f"chk_{f}"] = False
            st.rerun()

        st.markdown("")
        for fname in saved_files:
            col1, col2 = st.columns([6, 1])
            col1.checkbox(fname, key=f"chk_{fname}")
            if col2.button("🗑", key=f"del_{fname}", help=f"{fname} sil"):
                delete_file(fname)
                st.session_state.pop(f"chk_{fname}", None)
                st.rerun()


# ── Ana Alan ─────────────────────────────────────────────────────────────────

st.title("📚 Belge Asistanı")
st.caption("Belgelerini yükle, seçmek istediklerine tik at ve sorularını sor.")

selected = sorted(f for f in load_saved_files() if st.session_state.get(f"chk_{f}", False))

# Aktif oturumu yükle
session = load_session(st.session_state["active_session_id"])
if session is None:
    session = create_session()
    st.session_state["active_session_id"] = session["id"]

# Aktif belgeler göstergesi (yalnızca seçili belge varsa)
if selected:
    st.caption("**Aktif Belgeler:** " + "  ".join(f"`{d}`" for d in selected))

# Sohbet geçmişi
for msg in session["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Yeni soru
no_key = not st.session_state.get("api_key")
if no_key:
    placeholder = "Önce sol panele Gemini API Key girin..."
else:
    placeholder = "Bir şeyler sor..." if not selected else "Seçili belgeler hakkında soru sor..."

if prompt := st.chat_input(placeholder, disabled=no_key):
    session["messages"].append({"role": "user", "content": prompt})

    # İlk mesajdan başlık üret
    if len(session["messages"]) == 1:
        session["title"] = prompt[:45] + ("…" if len(prompt) > 45 else "")

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        spinner_text = f"{len(selected)} belge taranıyor..." if selected else "Yanıt oluşturuluyor..."
        with st.spinner(spinner_text):
            try:
                answer = get_response(
                    api_key=st.session_state["api_key"],
                    model=st.session_state.get("model", FREE_MODELS[0]),
                    system_prompt=build_system_prompt(selected),
                    messages=session["messages"],
                )
            except RuntimeError as e:
                answer = f"⚠️ {e}"
        st.markdown(answer)

    session["messages"].append({"role": "assistant", "content": answer})
    save_session(session)
