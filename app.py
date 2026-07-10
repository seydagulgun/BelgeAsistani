import io
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv, set_key

# ── Sabitler ──────────────────────────────────────────────────────────────────
ENV_FILE          = Path(__file__).parent / ".env"
_BASE_UPLOAD      = Path(__file__).parent / "uploads"
_BASE_SESSIONS    = Path(__file__).parent / "sessions"
MAX_CHARS_PER_DOC = 60_000
MAX_TOTAL_CHARS   = 180_000
GEMINI_URL        = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_STREAM_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
FREE_MODELS       = ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"]

_BASE_UPLOAD.mkdir(exist_ok=True)
_BASE_SESSIONS.mkdir(exist_ok=True)
load_dotenv(ENV_FILE)


# ── Kullanıcı kimliği ─────────────────────────────────────────────────────────

def _user_id() -> str:
    """Giriş yapılmışsa kullanıcı e-postası, değilse 'local'."""
    try:
        if st.user.is_logged_in and st.user.email:
            return re.sub(r"[^\w@.\-]", "_", st.user.email)
    except AttributeError:
        pass
    return "local"


def _upload_dir() -> Path:
    p = _BASE_UPLOAD / _user_id()
    p.mkdir(exist_ok=True)
    return p


def _cache_dir() -> Path:
    p = _upload_dir() / ".cache"
    p.mkdir(exist_ok=True)
    return p


def _sessions_dir() -> Path:
    p = _BASE_SESSIONS / _user_id()
    p.mkdir(exist_ok=True)
    return p

# ── Supabase (isteğe bağlı) ───────────────────────────────────────────────────
try:
    from supabase import create_client
    _SUPABASE_AVAILABLE = True
except ImportError:
    _SUPABASE_AVAILABLE = False


def _sb():
    if not _SUPABASE_AVAILABLE:
        return None
    url = (st.secrets.get("SUPABASE_URL") if hasattr(st, "secrets") else None) or os.getenv("SUPABASE_URL", "")
    key = (st.secrets.get("SUPABASE_KEY") if hasattr(st, "secrets") else None) or os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        return None
    return create_client(url, key)


# ── Dosya okuma ───────────────────────────────────────────────────────────────

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
    return "\n\n".join(f"--- {n} ---\n{df.to_string(index=False)}" for n, df in sheets.items())


# ── Dosya yönetimi ────────────────────────────────────────────────────────────

def save_file(uploaded_file) -> str:
    uid = _user_id()
    raw_path = _upload_dir() / uploaded_file.name
    data = uploaded_file.read()
    raw_path.write_bytes(data)
    text = extract_text_from_bytes(data, uploaded_file.name)
    (_cache_dir() / (uploaded_file.name + ".txt")).write_text(text, encoding="utf-8")
    sb = _sb()
    if sb:
        try:
            sb.table("documents").upsert({"name": uploaded_file.name, "text_content": text, "user_id": uid}).execute()
        except Exception:
            pass
    return uploaded_file.name


def load_saved_files() -> list[str]:
    uid = _user_id()
    sb = _sb()
    if sb:
        try:
            res = sb.table("documents").select("name").eq("user_id", uid).execute()
            return sorted(r["name"] for r in res.data)
        except Exception:
            pass
    return sorted(p.name for p in _upload_dir().iterdir() if p.is_file())


def load_file_text(filename: str) -> str:
    uid = _user_id()
    sb = _sb()
    if sb:
        try:
            res = sb.table("documents").select("text_content").eq("name", filename).eq("user_id", uid).single().execute()
            return res.data["text_content"]
        except Exception:
            pass
    cache = _cache_dir() / (filename + ".txt")
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    raw = _upload_dir() / filename
    text = extract_text_from_bytes(raw.read_bytes(), filename)
    cache.write_text(text, encoding="utf-8")
    return text


def delete_file(filename: str):
    uid = _user_id()
    sb = _sb()
    if sb:
        try:
            sb.table("documents").delete().eq("name", filename).eq("user_id", uid).execute()
        except Exception:
            pass
    for p in [
        _upload_dir() / filename,
        _cache_dir() / (filename + ".txt"),
        _cache_dir() / (filename + ".summary.txt"),
        _cache_dir() / (filename + ".questions.json"),
    ]:
        p.unlink(missing_ok=True)


def load_summary(filename: str) -> str:
    uid = _user_id()
    sb = _sb()
    if sb:
        try:
            res = sb.table("documents").select("summary").eq("name", filename).eq("user_id", uid).single().execute()
            return res.data.get("summary") or ""
        except Exception:
            pass
    p = _cache_dir() / (filename + ".summary.txt")
    return p.read_text(encoding="utf-8") if p.exists() else ""


def load_questions(filename: str) -> list[str]:
    uid = _user_id()
    sb = _sb()
    if sb:
        try:
            res = sb.table("documents").select("questions").eq("name", filename).eq("user_id", uid).single().execute()
            return res.data.get("questions") or []
        except Exception:
            pass
    p = _cache_dir() / (filename + ".questions.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def save_insights(filename: str, summary: str, questions: list[str]):
    uid = _user_id()
    sb = _sb()
    if sb:
        try:
            sb.table("documents").update({"summary": summary, "questions": questions}).eq("name", filename).eq("user_id", uid).execute()
        except Exception:
            pass
    (_cache_dir() / (filename + ".summary.txt")).write_text(summary, encoding="utf-8")
    (_cache_dir() / (filename + ".questions.json")).write_text(
        json.dumps(questions, ensure_ascii=False), encoding="utf-8"
    )


# ── Sohbet oturumları ─────────────────────────────────────────────────────────

def list_sessions() -> list[dict]:
    uid = _user_id()
    sb = _sb()
    if sb:
        try:
            res = sb.table("sessions").select("id, title, created_at").eq("user_id", uid).order("created_at", desc=True).execute()
            return res.data
        except Exception:
            pass
    sessions = []
    for f in _sessions_dir().glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            sessions.append({"id": d["id"], "title": d.get("title", ""), "created_at": d.get("created_at", "")})
        except Exception:
            pass
    return sorted(sessions, key=lambda x: x["created_at"], reverse=True)


def load_session(session_id: str) -> dict | None:
    uid = _user_id()
    sb = _sb()
    if sb:
        try:
            res = sb.table("sessions").select("*").eq("id", session_id).eq("user_id", uid).single().execute()
            return res.data
        except Exception:
            pass
    p = _sessions_dir() / f"{session_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save_session(session: dict):
    uid = _user_id()
    sb = _sb()
    if sb:
        try:
            sb.table("sessions").upsert({**session, "user_id": uid}).execute()
            return
        except Exception:
            pass
    (_sessions_dir() / f"{session['id']}.json").write_text(
        json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def delete_session(session_id: str):
    uid = _user_id()
    sb = _sb()
    if sb:
        try:
            sb.table("sessions").delete().eq("id", session_id).eq("user_id", uid).execute()
        except Exception:
            pass
    (_sessions_dir() / f"{session_id}.json").unlink(missing_ok=True)


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


# ── Gemini ────────────────────────────────────────────────────────────────────

def list_available_models(api_key: str) -> list[str]:
    try:
        r = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": api_key}, timeout=10,
        )
        if r.status_code != 200:
            return []
        names = [m["name"].replace("models/", "") for m in r.json().get("models", [])]
        return [n for n in names if "flash" in n or "pro" in n]
    except Exception:
        return []


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


def generate_insights(text: str, api_key: str, model: str) -> tuple[str, list[str]]:
    prompt = (
        "Aşağıdaki belgeyi analiz et. YALNIZCA şu JSON formatında yanıt ver, başka hiçbir şey yazma:\n"
        '{"summary": "2-3 cümlelik Türkçe özet", "questions": ["soru1", "soru2", "soru3", "soru4"]}\n\n'
        f"Belge:\n{text[:8000]}"
    )
    try:
        r = requests.post(
            GEMINI_URL.format(model=model),
            params={"key": api_key},
            json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                  "generationConfig": {"maxOutputTokens": 512}},
            timeout=30,
        )
        raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return data.get("summary", ""), data.get("questions", [])[:4]
    except Exception:
        pass
    return "", []


def stream_response(api_key: str, model: str, system_prompt: str | None, messages: list):
    contents = [
        {"role": "user" if m["role"] == "user" else "model",
         "parts": [{"text": m["content"]}]}
        for m in messages
    ]
    payload = {"contents": contents, "generationConfig": {"maxOutputTokens": 2048}}
    if system_prompt:
        payload["system_instruction"] = {"parts": [{"text": system_prompt}]}

    r = requests.post(
        GEMINI_STREAM_URL.format(model=model),
        params={"key": api_key, "alt": "sse"},
        json=payload,
        stream=True,
        timeout=120,
    )
    if r.status_code != 200:
        try:
            msg = r.json().get("error", {}).get("message", r.text)
        except Exception:
            msg = r.text
        raise RuntimeError(f"API Hatası {r.status_code}: {msg}")

    for line in r.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        raw = line[6:]
        if raw == b"[DONE]":
            break
        try:
            data = json.loads(raw)
            yield data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, json.JSONDecodeError):
            pass


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Belge Asistanı", page_icon="📚", layout="wide")

# ── Giriş zorunluluğu (yalnızca auth yapılandırılmışsa) ──────────────────────
def _auth_configured() -> bool:
    try:
        return bool(st.secrets.get("auth"))
    except Exception:
        return False

if _auth_configured():
    try:
        if not st.user.is_logged_in:
            st.title("📚 Belge Asistanı")
            st.write("Devam etmek için giriş yapın.")
            st.login()
            st.stop()
    except Exception:
        pass

if "active_session_id" not in st.session_state:
    existing = list_sessions()
    st.session_state["active_session_id"] = existing[0]["id"] if existing else create_session()["id"]

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    if _auth_configured():
        try:
            if st.user.is_logged_in:
                st.caption(f"👤 {st.user.email}")
                if st.button("Çıkış Yap", use_container_width=True):
                    st.logout()
        except Exception:
            pass

    st.header("⚙️ Ayarlar")

    env_key = (st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("GEMINI_API_KEY", "")
    if env_key:
        st.session_state["api_key"] = env_key
    else:
        manual_key = st.text_input(
            "Gemini API Key", type="password",
            value=st.session_state.get("api_key", ""),
            help="aistudio.google.com/app/apikey adresinden ücretsiz alabilirsin",
        )
        if manual_key and manual_key != st.session_state.get("api_key"):
            st.session_state["api_key"] = manual_key
            st.session_state.pop("available_models", None)
            set_key(str(ENV_FILE), "GEMINI_API_KEY", manual_key)
        elif manual_key:
            st.session_state["api_key"] = manual_key

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
        st.session_state["active_session_id"] = create_session()["id"]
        st.rerun()

    sessions = list_sessions()
    active_id = st.session_state["active_session_id"]

    for s in sessions:
        col1, col2 = st.columns([5, 1])
        is_active = s["id"] == active_id
        if col1.button(s["title"], key=f"sess_{s['id']}", use_container_width=True,
                       type="primary" if is_active else "secondary"):
            st.session_state["active_session_id"] = s["id"]
            st.rerun()
        if col2.button("🗑", key=f"delsess_{s['id']}"):
            delete_session(s["id"])
            remaining = [x for x in sessions if x["id"] != s["id"]]
            st.session_state["active_session_id"] = remaining[0]["id"] if remaining else create_session()["id"]
            st.rerun()

    # Sohbet yeniden adlandırma (aktif sohbet için)
    active_meta = next((s for s in sessions if s["id"] == active_id), None)
    if active_meta:
        new_title = st.text_input(
            "Sohbet adı",
            value=active_meta["title"],
            key=f"title_{active_id}",
            placeholder="Sohbet adını düzenle...",
        )
        if new_title and new_title != active_meta["title"]:
            sess = load_session(active_id)
            if sess:
                sess["title"] = new_title
                save_session(sess)
                st.rerun()

    st.divider()

    # ── Belgeler ──────────────────────────────────────────────────────────────
    st.subheader("📁 Belgeler")

    uploaded_files = st.file_uploader(
        "Belge ekle",
        type=["pdf", "docx", "csv", "xlsx", "xls", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        for f in uploaded_files:
            with st.spinner(f"{f.name} kaydediliyor..."):
                name = save_file(f)
                st.session_state[f"chk_{name}"] = True
            if st.session_state.get("api_key") and not load_summary(name):
                with st.spinner(f"{name} analiz ediliyor..."):
                    text = load_file_text(name)
                    summary, questions = generate_insights(
                        text,
                        st.session_state["api_key"],
                        st.session_state.get("model") or FREE_MODELS[0],
                    )
                    if summary or questions:
                        save_insights(name, summary, questions)
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


# ── Ana Alan ──────────────────────────────────────────────────────────────────

st.title("📚 Belge Asistanı")
st.caption("Belgelerini yükle, seçmek istediklerine tik at ve sorularını sor.")

selected = sorted(f for f in load_saved_files() if st.session_state.get(f"chk_{f}", False))

session = load_session(st.session_state["active_session_id"])
if session is None:
    session = create_session()
    st.session_state["active_session_id"] = session["id"]

if selected:
    st.caption("**Aktif Belgeler:** " + "  ".join(f"`{d}`" for d in selected))

# Sohbeti dışa aktar
if session["messages"]:
    lines = [
        f"{'**Siz**' if m['role'] == 'user' else '**Asistan**'}\n\n{m['content']}"
        for m in session["messages"]
    ]
    st.download_button(
        "⬇️ Sohbeti İndir (.md)",
        "\n\n---\n\n".join(lines),
        file_name=f"{session['title']}.md",
        mime="text/markdown",
    )

st.divider()

# Özet ve soru önerileri (boş sohbet + seçili belgeler)
if not session["messages"] and selected:
    for fname in selected:
        summary  = load_summary(fname)
        questions = load_questions(fname)
        if summary:
            with st.expander(f"📄 {fname} — Özet", expanded=True):
                st.markdown(summary)
        if questions:
            st.markdown(f"**{fname} için önerilen sorular:**")
            cols = st.columns(2)
            for i, q in enumerate(questions):
                if cols[i % 2].button(q, key=f"sug_{fname}_{i}", use_container_width=True):
                    st.session_state["pending_prompt"] = q
                    st.rerun()
        if summary or questions:
            st.divider()

# Sohbet geçmişi
for msg in session["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Giriş
no_key = not st.session_state.get("api_key")
placeholder = (
    "Önce sol panele Gemini API Key girin..." if no_key
    else ("Seçili belgeler hakkında soru sor..." if selected else "Bir şeyler sor...")
)

pending    = st.session_state.pop("pending_prompt", None)
chat_input = st.chat_input(placeholder, disabled=no_key)
prompt     = pending or chat_input

if prompt:
    session["messages"].append({"role": "user", "content": prompt})
    if len(session["messages"]) == 1:
        session["title"] = prompt[:45] + ("…" if len(prompt) > 45 else "")
        save_session(session)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(
                stream_response(
                    api_key=st.session_state["api_key"],
                    model=st.session_state.get("model") or FREE_MODELS[0],
                    system_prompt=build_system_prompt(selected),
                    messages=session["messages"],
                )
            )
        except Exception as e:
            answer = f"⚠️ Hata: {e}"
            st.markdown(answer)

    session["messages"].append({"role": "assistant", "content": answer})
    save_session(session)
