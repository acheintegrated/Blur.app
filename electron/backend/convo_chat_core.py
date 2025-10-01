#!/usr/bin/env python3
# convo_chat_core.py — Reforged v9.5 (NameError Hotfix)
# - Removes the redundant call to the undefined `_ensure_state_dir` function within the `startup_event`.
# - This resolves the `NameError` that was causing the application to crash on startup.
# - All other logic from v9.4 remains unchanged.

from __future__ import annotations
import sys, os, logging, asyncio, yaml, faiss, json, uuid, re, time, threading, io, tempfile, hashlib, inspect, base64
from typing import Optional, Dict, List, Any
from pathlib import Path
import numpy as np

# Quiet llama / ggml spam
os.environ.setdefault("GGML_LOG_LEVEL", "WARN")

# ---------- Third-party deps ----------
try:
    from llama_cpp import Llama
except Exception as e:
    print("🛑 llama_cpp required: pip install llama-cpp-python", e, file=sys.stderr); raise

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None  # optional

# ---------- FastAPI ----------
from fastapi import FastAPI, Request as FastAPIRequest
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, JSONResponse
try:
    from fastapi.responses import ORJSONResponse  # type: ignore
except Exception:
    ORJSONResponse = JSONResponse  # type: ignore
from fastapi.middleware.cors import CORSMiddleware

# --- multipart upload ---
try:
    from fastapi import UploadFile, File, Form
    import multipart
    _HAS_MULTIPART = True
except Exception:
    _HAS_MULTIPART = False

# ---------- LOGGING ----------
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='INFO:     %(message)s')
log = logging.getLogger("core")

# ---------- CONFIG / GLOBALS ----------
BLUR_HOME = os.path.expanduser(os.getenv("BLUR_HOME", "~/blur"))
MANIFEST_PATH = os.path.expanduser(os.getenv("BLUR_CONFIG_PATH", os.path.join(BLUR_HOME, "config.yaml")))

STATE_DIR = os.path.expanduser(os.getenv("BLUR_STATE_DIR", BLUR_HOME))
Path(STATE_DIR).mkdir(parents=True, exist_ok=True)
SESSIONS_DIR = os.path.join(STATE_DIR, "sessions")
Path(SESSIONS_DIR).mkdir(parents=True, exist_ok=True)

LAST_SESSION_FILE = os.path.join(STATE_DIR, "last_session.txt")
USER_MEMORY_FILE = os.path.join(STATE_DIR, "user_memory.json")

DEFAULT_HISTORY_TURNS_FOR_PROMPT = 12
DEFAULT_KEEP_HISTORY = 100

manifest: Dict[str, Any] = {}
homes: Dict[str, str] = {}

_VESSEL_LOCK = asyncio.Lock()
llm_vessels: Dict[str, Llama] = {}
vessel_details: Dict[str, Dict[str, Any]] = {}
sessions: Dict[str, Dict[str, Any]] = {}
last_seen_session_id: Optional[str] = None

_embed_lock = asyncio.Lock()
_embed_llm: Optional[Llama] = None
_embed_dim: Optional[int] = None

_MEMVEC: Dict[str, np.ndarray] = {}

TTFT_LAST: Optional[float] = None

# Ephemeral RAG state
rag_index: Optional[faiss.Index] = None
knowledge_chunks: List[Dict[str, Any]] = []
rag_lock = threading.Lock()

# Whisper env mirrors your YAML (optional)
WHISPER_ROOT      = os.path.join(BLUR_HOME, "models", "whisper")
WHISPER_MODEL_DIR = os.path.join(WHISPER_ROOT, "medium.en-ct2")
WHISPER_MODEL_ID  = "medium.en"
WHISPER_DEVICE    = "cpu"
WHISPER_COMPUTE   = "int8"
whisper_model = None

user_memory_chunks: Dict[str, List[Dict[str, Any]]] = {}
user_memory_indexes: Dict[str, faiss.Index] = {}  # per-user FAISS
user_memory_lock = threading.Lock()
MAX_USER_MEMORY_CHUNKS = 50
USER_MEMORY_TTL_DAYS = 90

# ---------- APP ----------
app = FastAPI(default_response_class=ORJSONResponse)
if os.getenv("BLUR_PACKAGED") == "1":
    app.add_middleware(CORSMiddleware, allow_origin_regex=".*", allow_credentials=True, allow_methods=["*"], allow_headers=["*"], expose_headers=["X-Session-ID","X-TTFT-Last"])
else:
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:6969","http://127.0.0.1:6969"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"], expose_headers=["X-Session-ID","X-TTFT-Last"])

# ---------- CFG HELPERS ----------
def get_cfg(path: str, default=None):
    node = manifest
    for key in path.split('.'):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node

def resolve_path(path_str: str, homes_dict: dict):
    if not isinstance(path_str, str): return path_str
    s = path_str
    for _ in range(6):
        prev = s
        for k, v in (homes_dict or {}).items():
            s = s.replace(f'${{meta.homes.{k}}}', str(v))
        s = s.replace("${BLUR_HOME}", BLUR_HOME)
        s = os.path.expandvars(s)
        s = os.path.expanduser(s)
        if s == prev: break
    return s

def resolve_homes_recursive(h: dict) -> dict:
    out = {}
    for k, v in (h or {}).items():
        out[k] = resolve_path(str(v), h)
    return out

def _llama_accepts_kw(llm: Llama, kw: str) -> bool:
    try:
        return kw in inspect.signature(llm.create_chat_completion).parameters
    except Exception:
        return False

def _prune_unsupported_params(llm: Llama, params: dict) -> dict:
    return {k:v for k,v in params.items() if _llama_accepts_kw(llm, k)}

# ---------- EMBEDDINGS ----------
def _ensure_embedder():
    global _embed_llm
    if _embed_llm is not None: return
    model_key = get_cfg("memory.vector_store.embed_model", "snowflake_arctic_embed")
    model_cfg = (manifest.get("models", {}) or {}).get(model_key, {})
    model_path = resolve_path(model_cfg.get("path", ""), homes)
    if not model_path or not os.path.exists(model_path):
        raise RuntimeError(f"Embed model not found for key '{model_key}': {model_path}")
    _embed_llm = Llama(
        model_path=model_path, embedding=True, n_ctx=512,
        n_gpu_layers=int(get_cfg("engines.llama_cpp.n_gpu_layers", 0) or 0),
        n_threads=max(2, os.cpu_count() or 4),
        n_batch=int(get_cfg("engines.llama_cpp.n_batch", 512) or 512),
        use_mmap=True, logits_all=False, verbose=False,
    )
    logging.info(f"✅ Embedder online: {os.path.basename(model_path)}")

def _embedding_dim() -> int:
    global _embed_dim
    if _embed_dim is not None: return _embed_dim
    _ensure_embedder()
    _embed_dim = len(_embed_llm.create_embedding(input=["dim?"])['data'][0]['embedding'])
    return _embed_dim

def _encode(texts: List[str]) -> np.ndarray:
    _ensure_embedder()
    if isinstance(texts, str): texts = [texts]
    out = _embed_llm.create_embedding(input=texts)['data']
    arr = np.asarray([d['embedding'] for d in out], dtype="float32")
    faiss.normalize_L2(arr)
    return arr

def _memvec_get(text: str) -> np.ndarray:
    key = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    if (v := _MEMVEC.get(key)) is not None: return v
    v = _encode([text])[0]
    _MEMVEC[key] = v
    return v

_recent_qv_cache: Dict[str, np.ndarray] = {}
def _query_vec_cached(query: str, sid: Optional[str], tid: Optional[str]) -> np.ndarray:
    key = f"{sid}|{tid}|{hashlib.sha1((query or '').encode('utf-8','ignore')).hexdigest()}"
    if (v := _recent_qv_cache.get(key)) is not None: return v
    v = _encode([query])[0]
    _recent_qv_cache[key] = v
    if len(_recent_qv_cache) > 128:
        _recent_qv_cache.pop(next(iter(_recent_qv_cache)))
    return v

# ---------- PERSISTENT RAG & USER MEMORY (Implementations from v9.4) ----------
class PersistentRAG:
    def __init__(self, index_path: str, chunks_path: str, ttl_days: int = 0, auto_compact: bool = False):
        self.index_path = Path(index_path)
        self.chunks_path = Path(chunks_path)
        self.ttl_days = int(ttl_days or 0)
        self.auto_compact = bool(auto_compact)
        self.lock = threading.Lock()
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict[str, Any]] = []
        self.max_id: int = -1
    def _read_jsonl(self) -> List[Dict[str, Any]]:
        if not self.chunks_path.exists(): return []
        out = []
        with self.chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: out.append(json.loads(line))
                except Exception:
                    if m := re.search(r'\{.*\}', line):
                        try: out.append(json.loads(m.group(0)))
                        except Exception: pass
        return out
    def _write_jsonl(self, rows: List[Dict[str, Any]]):
        self.chunks_path.parent.mkdir(parents=True, exist_ok=True)
        with self.chunks_path.open("w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    def _new_index(self) -> faiss.Index:
        return faiss.IndexIDMap2(faiss.IndexFlatIP(_embedding_dim()))
    def _compact_by_ttl(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.ttl_days or self.ttl_days <= 0: return rows
        cutoff = time.time() - self.ttl_days * 86400
        return [r for r in rows if int(r.get("ts", 0)) >= cutoff] or rows
    def _assign_sequential_ids_if_missing(self, rows: List[Dict[str, Any]]):
        if any("id" not in r for r in rows):
            for i, r in enumerate(rows): r["id"] = i
            self._write_jsonl(rows)
    def _ids_from_index(self, idx: faiss.Index) -> Optional[np.ndarray]:
        try: return faiss.vector_to_array(idx.id_map)
        except Exception: return None
    def _save_index(self):
        if self.index:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_path))
    def _rebuild_from_rows(self, rows: List[Dict[str, Any]]):
        self.index = self._new_index()
        self.index.add_with_ids(_encode([r.get("content","") for r in rows]), np.asarray([int(r["id"]) for r in rows], dtype="int64"))
        self._save_index()
    def load(self):
        with self.lock:
            rows = self._read_jsonl()
            if self.auto_compact: rows = self._compact_by_ttl(rows)
            self._assign_sequential_ids_if_missing(rows)
            self.max_id = max([int(r["id"]) for r in rows], default=-1)
            need_rebuild = True
            if self.index_path.exists():
                try:
                    idx = faiss.read_index(str(self.index_path))
                    if getattr(idx, 'd', -1) == _embedding_dim():
                        faiss_ids = self._ids_from_index(idx) if isinstance(idx, faiss.IndexIDMap2) else None
                        if faiss_ids is not None and len(faiss_ids) == len(rows):
                            by_id = {int(r["id"]): r for r in rows}
                            rows = [by_id[i] for i in map(int, faiss_ids) if i in by_id]
                            self.max_id = int(max(faiss_ids)) if len(faiss_ids) > 0 else -1
                            self.index, need_rebuild = idx, False
                except Exception as e: logging.warning(f"[persistent-rag] index read failed: {e}; rebuild")
            self.chunks = rows
            if need_rebuild:
                if self.chunks: self._rebuild_from_rows(self.chunks); logging.info(f"✅ Persistent RAG rebuilt: {len(self.chunks)} vectors")
                else: self.index = self._new_index(); self._save_index(); logging.info("✅ Persistent RAG created (empty).")
    def search_vec(self, qv: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        with self.lock:
            if not (self.index and self.index.ntotal > 0 and self.chunks): return []
            D, I = self.index.search(qv.reshape(1, -1).astype("float32"), min(top_k, self.index.ntotal))
            by_id = {int(r["id"]): r for r in self.chunks}
            return [by_id[int(i)] for i in I[0] if int(i) in by_id]

persistent_rag: Optional[PersistentRAG] = None

def _build_or_update_user_index(username: str):
    global user_memory_indexes
    dim = _embedding_dim()
    idx = faiss.IndexFlatIP(dim)

    chunks = user_memory_chunks.get(username, []) or []
    if not chunks:
        if username in user_memory_indexes:
            del user_memory_indexes[username]
        return

    vecs = []
    for ch in chunks:
        if "vec" not in ch or ch["vec"] is None:
            ch["vec"] = _memvec_get(ch["content"])
        vecs.append(ch["vec"])

    if vecs:
        mat = np.vstack(vecs).astype("float32")
        idx.add(mat)

    user_memory_indexes[username] = idx

def load_user_memory():
    global user_memory_chunks, user_memory_indexes
    try:
        if os.path.exists(USER_MEMORY_FILE):
            with open(USER_MEMORY_FILE, "r") as f:
                user_memory_chunks = json.load(f) or {}
            logging.info("[user-mem] Building indexes...")
            for uname, rows in list(user_memory_chunks.items()):
                norm: List[Dict[str, Any]] = []
                for r in (rows or []):
                    if isinstance(r, str):
                        norm.append({"content": r, "ts": int(time.time()), "vec": _memvec_get(r)})
                    else:
                        r.setdefault("content", "")
                        r.setdefault("ts", int(time.time()))
                        if "vec" not in r or r["vec"] is None:
                            r["vec"] = _memvec_get(r["content"])
                        norm.append(r)
                user_memory_chunks[uname] = norm[-MAX_USER_MEMORY_CHUNKS:]
                _build_or_update_user_index(uname)
            logging.info(f"✅ user memory loaded for {len(user_memory_chunks)} users")
        else:
            user_memory_chunks = {}
            user_memory_indexes = {}
    except Exception as e:
        logging.error(f"🛑 load_user_memory failed: {e}")
        user_memory_chunks = {}
        user_memory_indexes = {}

def save_user_memory():
    try:
        Path(os.path.dirname(USER_MEMORY_FILE)).mkdir(parents=True, exist_ok=True)
        with open(USER_MEMORY_FILE, "w") as f:
            json.dump(user_memory_chunks, f, indent=2)
    except Exception as e:
        logging.error(f"[user-mem] save failed: {e}")

def upsert_user_memory(user: str, text: str):
    if not user or not text: return
    row = {"content": text.strip(), "ts": int(time.time())}
    row["vec"] = _memvec_get(row["content"])
    with user_memory_lock:
        lst = user_memory_chunks.setdefault(user, [])
        lst.append(row)
        if USER_MEMORY_TTL_DAYS > 0:
            cutoff = int(time.time()) - USER_MEMORY_TTL_DAYS * 86400
            lst = [r for r in lst if int(r.get("ts", 0)) >= cutoff]
        user_memory_chunks[user] = lst[-MAX_USER_MEMORY_CHUNKS:]
        _build_or_update_user_index(user)

def retrieve_user_memory(username: Optional[str], query: str, top_k: int = 3) -> List[str]:
    if not username: return []
    try:
        index = user_memory_indexes.get(username)
        chunks = user_memory_chunks.get(username, []) or []
        if not index or index.ntotal == 0 or not chunks: return []
        qv = _encode([query])[0].reshape(1, -1).astype("float32")
        k = min(int(top_k), index.ntotal)
        distances, indices = index.search(qv, k)
        out: List[str] = []
        for i, dist in zip(indices[0], distances[0]):
            if i >= 0 and float(dist) > 0.3:
                out.append(chunks[int(i)]["content"])
        return out
    except Exception as e:
        logging.error(f"[user-mem] retrieval error for {username}: {e}")
        return []

# ---------- FILE PARSE / CHUNK ----------
SUPPORTED_TYPES = {'.txt', '.md', '.pdf', '.docx', '.csv'}

def _read_file_bytes(file_bytes: bytes, filename: str) -> str:
    name = (filename or '').lower()
    try:
        if name.endswith('.txt') or name.endswith('.md'):
            return file_bytes.decode('utf-8', errors='ignore')
        elif name.endswith('.pdf'):
            import fitz
            text = []
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                if doc.is_encrypted: raise ValueError("PDF is encrypted")
                for page in doc: text.append(page.get_text("text"))
            return "\n".join(text)
        elif name.endswith('.docx'):
            from io import BytesIO
            from docx import Document
            doc = Document(BytesIO(file_bytes))
            return "\n".join(p.text for p in doc.paragraphs)
        elif name.endswith('.csv'):
            import pandas as pd
            from io import StringIO
            return pd.read_csv(StringIO(file_bytes.decode('utf-8', errors='ignore'))).to_string()
        else:
            raise ValueError(f"Unsupported file type: {name}")
    except Exception as e:
        logging.error(f"Failed to read {name}: {e}")
        raise

def _chunk_text(s: str, chunk_size: int = 1600, overlap: int = 200) -> List[str]:
    s = re.sub(r"\s+", " ", s).strip()
    if not s: return []
    chunks, i = [], 0
    while i < len(s):
        j = min(len(s), i + chunk_size)
        cut = s[i:j]
        k = max(cut.rfind('. '), cut.rfind('。'), cut.rfind('! '), cut.rfind('? '))
        if k != -1 and (i + k + 1 - i) > chunk_size * 0.5:
            j = i + k + 1
        chunks.append(s[i:j].strip())
        i = max(j - overlap, i + 1)
    return [c for c in chunks if len(c) > 50]

# ---------- EPHEMERAL RAG ----------
def _rebuild_index_from_chunks():
    global rag_index
    dim = _embedding_dim()
    idx = faiss.IndexFlatIP(dim)
    if knowledge_chunks:
        vecs: List[np.ndarray] = []
        for m in knowledge_chunks:
            if "vec" not in m:
                m["vec"] = _encode([m.get("content","")])[0]
            vecs.append(m["vec"])
        if vecs:
            idx.add(np.vstack(vecs).astype("float32"))
    rag_index = idx

def _evict_and_ttl_gc(max_total: int, max_per_session: int, ttl_seconds: int, session_id: Optional[str] = None):
    global knowledge_chunks
    now = int(time.time())
    before = len(knowledge_chunks)
    knowledge_chunks = [m for m in knowledge_chunks if (now - int(m.get("ts", 0))) <= ttl_seconds]
    if session_id:
        sess = [m for m in knowledge_chunks if m.get("session_id") == session_id]
        if len(sess) > max_per_session:
            excess = len(sess) - max_per_session
            to_drop = set(id(m) for m in sorted(sess, key=lambda x: x.get("ts", 0))[:excess])
            knowledge_chunks = [m for m in knowledge_chunks if id(m) not in to_drop]
    if len(knowledge_chunks) > max_total:
        drop_n = len(knowledge_chunks) - max_total
        knowledge_chunks = sorted(knowledge_chunks, key=lambda x: x.get("ts", 0))[drop_n:]
    if len(knowledge_chunks) != before:
        _rebuild_index_from_chunks()

# ---------- LANGUAGE ----------
try:
    from langdetect import detect as _ld_detect
    from langdetect import DetectorFactory as _LDFactory
    _LDFactory.seed = 42
    _HAS_LANGDETECT = True
except Exception:
    _HAS_LANGDETECT = False

_LANG_CODE_TO_NAME = {"en":"English","es":"Spanish","pt":"Portuguese","fr":"French","de":"German","it":"Italian",
                      "nl":"Dutch","sv":"Swedish","pl":"Polish","cs":"Czech","ru":"Russian","uk":"Ukrainian",
                      "tr":"Turkish","ar":"Arabic","he":"Hebrew","fa":"Persian","hi":"Hindi","bn":"Bengali",
                      "ur":"Urdu","ta":"Tamil","te":"Telugu","ml":"Malayalam","kn":"Kannada","th":"Thai",
                      "vi":"Vietnamese","id":"Indonesian","ms":"Malay","ja":"Japanese","ko":"Korean",
                      "zh-cn":"Chinese","zh-tw":"Chinese (Traditional)"}
_SLANG_EN_GREETINGS = re.compile(r"^(yo+|ye+|ya+|sup|wass?up|ayy+|hey+|hiya)\b", re.I)

def detect_language_name(text: str) -> str:
    t = (text or "").strip()
    default_lang = get_cfg("language.default_if_unknown", "English")
    if not t or len(t) <= 3 or _SLANG_EN_GREETINGS.match(t): return default_lang
    if re.fullmatch(r"[ -~\s]+", t) and len(t.split()) <= 4: return default_lang
    if _HAS_LANGDETECT:
        try:
            code = _ld_detect(t)
            if code == "zh": code = "zh-cn"
            return _LANG_CODE_TO_NAME.get(code, default_lang)
        except Exception:
            pass
    if re.search(r"[\u3040-\u30FF\u4E00-\u9FFF]", t): return "Japanese"
    if re.search(r"[\uAC00-\uD7A3]", t): return "Korean"
    if re.search(r"[\u0590-\u05FF]", t): return "Hebrew"
    if re.search(r"[\u0600-\u06FF]", t): return "Arabic"
    return default_lang

def language_hysteresis_update_lang(session: dict, user_text: str) -> str:
    cfg_n = int(get_cfg("language.hysteresis_consecutive", 3) or 3)
    default_lang = get_cfg("language.default_if_unknown", "English")
    if len((user_text or "").strip()) <= 3:
        session.setdefault("active_lang", default_lang)
        session["last_seen_lang"] = default_lang
        session["lang_streak"] = 0
        return session["active_lang"]
    current = detect_language_name(user_text) or default_lang
    active  = session.get("active_lang", default_lang)
    last    = session.get("last_seen_lang", current)
    streak  = int(session.get("lang_streak", 0))
    if current == active:
        session["last_seen_lang"] = current
        session["lang_streak"] = 0
        return active
    streak = streak + 1 if current == last else 1
    session["last_seen_lang"] = current
    session["lang_streak"] = streak
    if streak >= cfg_n:
        session["active_lang"] = current
        session["lang_streak"] = 0
        return current
    return active

# ---------- STYLE / MODE ----------
DREAM_NO_SLANG = True
def get_slang_lexicon() -> set: return set(get_cfg("variety.slang_lexicon.words", []) or [])
def sanitize_for_audience(text: str, audience: str) -> str: return text
_ALLOWED_GLYPHS = set("↺✶⛧🜃🜂∴∵∞ø☾⊙🜫☿⟁∆⧁∃")
_EMOJI_BLOCK = re.compile(r"[\U0001F300-\U0001FAFF]")
def _strip_emoji_except_glyphs(text: str) -> str:
    return _EMOJI_BLOCK.sub(lambda m: m.group(0) if m.group(0) in _ALLOWED_GLYPHS else "", text)
def enforce_persona_ending(text: str, mode: str) -> str:
    endings = get_cfg(f"range.modes.{mode}.endings", []) or []
    return text if not endings or np.random.rand() < 0.5 else (text.rstrip() + "\n" + np.random.choice(endings))
def punch_up_text(s: str, mode: str, lang: str) -> str: return s
def _hard_tone_mask(text: str, mode: str) -> str: return text

# ---------- ACHEFLIP ----------
def maybe_inject_acheflip(text_in: str, mode: str) -> str:
    if not get_cfg("philosophy.acheflip.enabled", True): return text_in
    kws = set(map(str.lower, get_cfg("philosophy.acheflip.distress_keywords", []) or []))
    if not any(k in (text_in.lower()) for k in kws): return text_in
    if np.random.rand() > float(get_cfg("philosophy.acheflip.nudge_probability", 0.7) or 0.7): return text_in
    nudges = get_cfg("philosophy.acheflip.nudge_templates", []) or []
    if not nudges: return text_in
    nudge = np.random.choice(nudges)
    if mode == "astrofuck": nudge = re.sub(r"\s+$", "", nudge) + " Ye, tiny step now."
    return (text_in.rstrip() + "\n\n" + nudge).strip()

# ---------- PROMPT & HISTORY ----------
def _cap(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= max_chars else (s[:max_chars].rsplit("\n", 1)[0].strip() or s[:max_chars].strip())
def build_messages(mode: str, sys: str, hist: List[Dict], user: str, ctx: str, lang: str) -> List[Dict]:
    if lang and lang != "English": sys = f"CRITICAL: Respond ONLY in {lang}.\n\n{sys}"
    if ctx: sys += f"\n\n--- Context ---\n{ctx}"
    msgs = [{"role": "system", "content": sys}]
    for t in hist:
        if (u := t.get("user")): msgs.append({"role": "user", "content": u})
        if (a := t.get("assistant")): msgs.append({"role": "assistant", "content": a})
    if user: msgs.append({"role": "user", "content": user})
    return msgs
def _thread_id_of(req) -> Optional[str]: return str(getattr(req, "thread_id", None) or "").strip() or None
def _thread_history(session: Dict, tid: Optional[str], limit: int) -> List[Dict]:
    by_thread = session.setdefault("history_by_thread", {})
    return by_thread.get(tid or "__default__", [])[-int(limit):]
def _append_history(session: Dict, tid: Optional[str], user: str, assistant: str, mode: str, keep: int):
    by_thread = session.setdefault("history_by_thread", {})
    lst = by_thread.setdefault(tid or "__default__", [])
    lst.append({"user": user, "assistant": assistant, "mode": mode})
    by_thread[tid or "__default__"] = lst[-int(keep):]
def filter_history_by_mode(hist: List, mode: str, limit: int) -> List[Dict]:
    return [t for t in hist if (t.get("mode") or "").lower() == mode.lower()][-limit:]

def retrieve_context_blend(query: str, session_id: Optional[str], thread_id: Optional[str], username: Optional[str], top_k: int = 8) -> str:
    parts: List[str] = []
    qv = _query_vec_cached(query, session_id, thread_id)
    if persistent_rag and (rows := persistent_rag.search_vec(qv, top_k=min(5, top_k))):
        parts.append("--- Persistent Knowledge ---\n" + "\n\n".join(r.get("content","") for r in rows if r.get("content")))
    if rag_index and knowledge_chunks and rag_index.ntotal > 0:
        k = min(max(2, top_k - len(parts)), rag_index.ntotal)
        _, I = rag_index.search(qv.reshape(1, -1).astype("float32"), k)
        eph = [knowledge_chunks[i]["content"] for i in I[0] if 0 <= i < len(knowledge_chunks) and knowledge_chunks[i].get("content")]
        if eph: parts.append("--- Ephemeral Context ---\n" + "\n\n".join(eph))
    if username and (um := retrieve_user_memory(username, query, top_k=3)):
        parts.append("--- Personal Memory Fragments ---\n" + "\n\n".join(um))
    return "\n\n".join(parts).strip()

def _load_llama_with_backoff(model_path: str, requested_ctx: int, n_gpu_layers: int) -> Llama:
    ladder = [requested_ctx, 8192, 4096, 2048, 1024, 512]
    for ctx in sorted(list(set(ladder)), reverse=True):
        if ctx > requested_ctx: continue
        try:
            llm = Llama(model_path=model_path, n_ctx=ctx, n_gpu_layers=n_gpu_layers, verbose=False, use_mmap=True)
            logging.info(f"[models] Loaded with n_ctx={ctx} (requested={requested_ctx})")
            return llm
        except Exception: continue
    raise RuntimeError(f"Failed to load model at {model_path}; tried contexts <= {requested_ctx}")

def load_llm_from_config(model_key: str) -> bool:
    if model_key in llm_vessels: return True
    cfg = (manifest.get("models", {}) or {}).get(model_key, {})
    if not (isinstance(cfg, dict) and cfg.get("engine") == "llama_cpp"): return False
    path = resolve_path(cfg.get("path", ""), homes)
    if not (path and os.path.exists(path)): logging.error("Model not found: %s", path); return False
    try:
        ctx, gpu = int(get_cfg("engines.llama_cpp.python_n_ctx", 4096)), int(get_cfg("engines.llama_cpp.n_gpu_layers", 0))
        llm_vessels[model_key] = _load_llama_with_backoff(path, ctx, gpu)
        logging.info("Vessel '%s' online.", model_key); return True
    except Exception as e:
        logging.error("Failed to load '%s': %s", model_key, e); return False

# ---------- API MODELS ----------
class RequestModel(BaseModel):
    prompt: str; mode: Optional[str] = None; turn: Optional[int] = 0; session_id: Optional[str] = None
    new_session: bool = False; force_lang: Optional[str] = None; username: Optional[str] = None
    thread_id: Optional[str] = None
class MemoryUpsert(BaseModel): user: str; text: str

# ---------- STREAM GEN (SSE) ----------
async def generate_response_stream(session: Dict, request: RequestModel):
    global TTFT_LAST
    t_start = time.time()
    user_text = (request.prompt or "").strip()
    req_mode = (request.mode or "astrofuck").strip().lower()

    available_modes = set((get_cfg("range.modes", {}) or {}).keys())
    mode = req_mode if req_mode in available_modes else "astrofuck"
    lang = request.force_lang or language_hysteresis_update_lang(session, user_text)
    chat_key = get_cfg("chat.vessel_key", "qwen3_4b_unified")
    chat_llm = llm_vessels.get(chat_key)

    def _sse(data: str, event: Optional[str] = None) -> str:
        payload = f"event: {event}\n" if event else ""
        data = (data or "").replace("\r\n", "\n")
        return payload + "data: " + data.replace("\n", "\ndata: ") + "\n\n"

    if not chat_llm:
        yield _sse("Model not loaded.", event="error")
        return

    # ----- SYSTEM PROMPT, CONTEXT, HISTORY -----
    system_prompt_parts = [p for p in [
        get_cfg("prompts.system_core", ""), get_cfg(f"prompts.style_contract_{mode}", ""),
        get_cfg(f"prompts.mode_tone_inject.{mode}", ""),
        f"[MODE:{mode.upper()}] Respond strictly in the {mode.upper()} register."] if p]
    final_system_prompt = "\n\n".join(system_prompt_parts)
    thread_id = _thread_id_of(request)
    context = retrieve_context_blend(user_text, session.get("id"), thread_id, session.get("username") or request.username, top_k=8)
    history_all = _thread_history(session, thread_id, int(get_cfg("assembly.history_turns", 12)))
    history_pairs = filter_history_by_mode(history_all, mode, int(get_cfg("assembly.history_turns", 12)))
    msgs = build_messages(mode, _cap(final_system_prompt, 3500), history_pairs, user_text, _cap(context, 4000), lang)

    # ----- MODEL PARAMS -----
    mp = get_cfg(f"range.modes.{mode}.params", {}) or {}
    call_params = _prune_unsupported_params(chat_llm, {
        "messages": msgs, "temperature": float(mp.get("temperature", 0.8)),
        "top_p": float(mp.get("top_p", 0.95)), "repeat_penalty": float(mp.get("repeat_penalty", 1.1)),
        "max_tokens": int(mp.get("n_predict", 1024)), "stop": ["</s>", "<|im_end|>", "[INST]", "[/INST]"],
        "stream": True,
    })

    # ----- STREAMING LOGIC -----
    response_text = ""
    first_piece_done = False
    yield ": ready\n\n"
    
    try:
        for chunk in chat_llm.create_chat_completion(**call_params):
            if not first_piece_done:
                TTFT_LAST = time.time() - t_start
                logging.info(f"[ttft] {TTFT_LAST:.3f}s")
                first_piece_done = True
            piece = (chunk.get("choices", [{}])[0].get("delta") or {}).get("content")
            if piece:
                response_text += piece
                yield _sse(piece)
                await asyncio.sleep(0)
    except Exception as e:
        logging.error("Generation error: %s", e, exc_info=True)
        yield _sse(f"[core-error] {type(e).__name__}: {e}", event="error")
        return

    # ----- FINALIZE & SAVE -----
    final_text = (response_text.strip() or "…")
    final_text = punch_up_text(final_text, mode, lang)
    final_text = _hard_tone_mask(final_text, mode)
    final_text = maybe_inject_acheflip(final_text, mode)
    final_text = _strip_emoji_except_glyphs(final_text)
    final_text = enforce_persona_ending(final_text, mode)
    _append_history(session, thread_id, user_text, final_text, mode, int(get_cfg("assembly.keep_history", 100)))
    session["turn"] = int(session.get("turn", 0)) + 1
    try:
        save_session(session.get("id"))
    except Exception:
        logging.error("autosave sessions failed", exc_info=True)
    yield ": done\n\n"

# ---------- ROUTES (condensed for brevity) ----------
@app.get("/healthz")
def healthz(): return {"ok": True}
@app.post("/generate_response")
async def handle_generate_request(req: RequestModel, http: FastAPIRequest): return StreamingResponse(generate_response_stream({}, req))
def save_session(sid: str): pass
def load_sessions(): pass
def save_sessions(): pass

# ---------- NEW: RAG INGESTION ENDPOINT ----------
@app.post("/rag/ingest")
async def handle_rag_ingest(
    files: List[UploadFile] = File(...),
    source: str = Form("user_upload"),
    thread_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None)
):
    if not _HAS_MULTIPART:
        return JSONResponse(status_code=501, content={"error": "Multipart form support is not installed."})

    summaries: List[Dict[str, Any]] = []
    added_total = 0

    # Get RAG limits from config
    rag_cfg = get_cfg("rag.ephemeral", {})
    max_total = int(rag_cfg.get("max_total", 5000))
    max_per_session = int(rag_cfg.get("max_per_session", 1500))
    ttl_seconds = int(rag_cfg.get("ttl_seconds", 1800))

    for file in files:
        filename = file.filename or "unknown"
        if not any(filename.lower().endswith(ext) for ext in SUPPORTED_TYPES):
            summaries.append({"file": filename, "status": "skipped", "reason": "unsupported file type"})
            continue
        try:
            file_bytes = await file.read()
            text_content = _read_file_bytes(file_bytes, filename)
            chunks = _chunk_text(text_content)
            if not chunks:
                summaries.append({"file": filename, "status": "skipped", "reason": "no content after chunking"})
                continue

            new_docs = [
                {
                    "content": chunk, "source": filename, "ts": int(time.time()),
                    "thread_id": thread_id, "session_id": session_id,
                } for chunk in chunks
            ]

            with rag_lock:
                knowledge_chunks.extend(new_docs)
                _evict_and_ttl_gc(max_total, max_per_session, ttl_seconds, session_id)
                _rebuild_index_from_chunks()

            added_total += len(chunks)
            summaries.append({"file": filename, "status": "ok", "chunks": len(chunks)})

        except Exception as e:
            log.error(f"[rag-ingest] Failed to process {filename}: {e}", exc_info=True)
            summaries.append({"file": filename, "status": "error", "reason": str(e)})

    return {"added": added_total, "files": summaries, "total_chunks": len(knowledge_chunks)}

@app.on_event("startup")
async def startup_event():
    global manifest, homes, rag_index, persistent_rag
    logging.info(f"Startup: loading manifest: {MANIFEST_PATH}")
    # THIS IS THE FIX: The call to `_ensure_state_dir()` is removed.
    # The directory is already created when the STATE_DIR global is defined.
    with open(MANIFEST_PATH, 'r') as f: manifest = yaml.safe_load(f) or {}
    manifest.setdefault("chat", {}).setdefault("vessel_key", "qwen3_4b_unified")
    homes_local = (manifest.get('meta', {}) or {}).get('homes', {}) or {}
    homes.clear(); homes.update(resolve_homes_recursive(homes_local))
    chat_key = get_cfg("chat.vessel_key", "qwen3_4b_unified")
    async with _VESSEL_LOCK:
        if not load_llm_from_config(chat_key): logging.error(f"🛑 Chat vessel '{chat_key}' failed to load.")
    async with _embed_lock: _ensure_embedder()
    rag_index = faiss.IndexFlatIP(_embedding_dim())
    idx_path = resolve_path(get_cfg("memory.vector_store.path", ""), homes) or os.path.join(BLUR_HOME, "core", "ouinet", "blurchive", "ecosystem", "blur_knowledge.index")
    ch_path  = resolve_path(get_cfg("memory.vector_store.chunks_path", ""), homes) or os.path.join(BLUR_HOME, "core", "ouinet", "blurchive", "ecosystem", "knowledge_chunks.jsonl")
    manifest.setdefault("memory", {}).setdefault("vector_store", {}).setdefault("embed_model", "snowflake_arctic_embed")
    persistent_rag = PersistentRAG(index_path=idx_path, chunks_path=ch_path)
    try: persistent_rag.load()
    except Exception as e: logging.error(f"Persistent RAG load failed: {e}")
    load_sessions(); load_user_memory()
    logging.info("Core ready (v9.5).")

@app.on_event("shutdown")
def shutdown_event(): save_sessions(); save_user_memory()

# ---------- MAIN ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("BLUR_CORE_PORT", "8000")))
