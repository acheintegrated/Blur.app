#!/usr/bin/env python3
# convo_chat_core.py — Reforged v7.1
# - BLUR_HOME-first (no bundled fallbacks). All paths resolve via BLUR_HOME/BLUR_CONFIG_PATH.
# - Chat vessel n_ctx SAFE-LOAD: backs off (requested → 8192 → 4096 → 2048 → 1024 → 512).
# - Persistent RAG: strict JSONL↔FAISS ID alignment, tolerant rebuild, hot-reload/search endpoints.
# - Ephemeral RAG + User memory unchanged.
# - Whisper optional; quiet llama logs.
# - Sessions: state under BLUR_HOME by default, legacy ~/.blur migration, autosave each turn + periodic.

import sys, os, logging, asyncio, yaml, faiss, json, uuid, re, time, threading, base64, io, tempfile, random
from typing import Optional, Dict, List, Any
from pathlib import Path
import numpy as np

# Quiet llama/ggml spam before import
os.environ.setdefault("GGML_LOG_LEVEL", "WARN")

# ---------- Third-party deps ----------
from pydub import AudioSegment
try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None  # lazy, optional

from llama_cpp import Llama

from fastapi import FastAPI, Request as FastAPIRequest
from fastapi import UploadFile, File, Form
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, JSONResponse
try:
    from fastapi.responses import ORJSONResponse  # type: ignore
except Exception:
    ORJSONResponse = JSONResponse  # type: ignore
from fastapi.middleware.cors import CORSMiddleware

# ---------- OPTIONAL: multipart ----------
try:
    import multipart  # python-multipart
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

# State lives alongside BLUR_HOME (Application Support/Blur by default)
STATE_DIR = os.path.expanduser(os.getenv("BLUR_STATE_DIR", BLUR_HOME))
Path(STATE_DIR).mkdir(parents=True, exist_ok=True)
SESSIONS_FILE = os.path.join(STATE_DIR, "sessions.json")
USER_MEMORY_FILE = os.path.join(STATE_DIR, "user_memory.json")

# Back-compat: migrate legacy ~/.blur/* → STATE_DIR on first run
LEGACY_SESSIONS_FILE = os.path.expanduser("~/.blur/sessions.json")
LEGACY_USER_MEMORY_FILE = os.path.expanduser("~/.blur/user_memory.json")
try:
    if os.path.exists(LEGACY_SESSIONS_FILE) and not os.path.exists(SESSIONS_FILE):
        with open(LEGACY_SESSIONS_FILE, "r") as f_in, open(SESSIONS_FILE, "w") as f_out:
            f_out.write(f_in.read())
        logging.info(f"Migrated legacy sessions.json → {SESSIONS_FILE}")
    if os.path.exists(LEGACY_USER_MEMORY_FILE) and not os.path.exists(USER_MEMORY_FILE):
        with open(LEGACY_USER_MEMORY_FILE, "r") as f_in, open(USER_MEMORY_FILE, "w") as f_out:
            f_out.write(f_in.read())
        logging.info(f"Migrated legacy user_memory.json → {USER_MEMORY_FILE}")
except Exception as e:
    logging.error(f"Legacy migration failed: {e}")

DEFAULT_HISTORY_TURNS_FOR_PROMPT = 24
DEFAULT_KEEP_HISTORY = 100

manifest: Dict[str, Any] = {}
homes: Dict[str, str] = {}

# Vessel registry + locks
_VESSEL_LOCK = asyncio.Lock()
llm_vessels: Dict[str, Llama] = {}
vessel_details: Dict[str, Dict[str, Any]] = {}
sessions: Dict[str, Dict[str, Any]] = {}
last_seen_session_id: Optional[str] = None

# Embedding singleton
_embed_lock = asyncio.Lock()
_embed_llm: Optional[Llama] = None
_embed_dim: Optional[int] = None

# Autosave
_AUTOSAVE_TASK: Optional[asyncio.Task] = None
_AUTOSAVE_EVERY_SEC = int(os.getenv("BLUR_AUTOSAVE_SEC", "10"))

# ---------- USER MEMORY ----------
user_memory_chunks: Dict[str, List[Dict[str, Any]]] = {}
user_memory_lock = threading.Lock()
MAX_USER_MEMORY_CHUNKS = 50
USER_MEMORY_TTL_DAYS = 90  # reserved; not enforced here

# ---------- EPHEMERAL UI RAG ----------
EPHEMERAL_RAG_ENABLED = True
MAX_RAG_CHUNKS_TOTAL = 5000
MAX_RAG_CHUNKS_PER_SESSION = 1500
EPHEMERAL_TTL_SECONDS = 60 * 30  # 30 min

rag_index: Optional[faiss.Index] = None
knowledge_chunks: List[Dict[str, Any]] = []
rag_lock = threading.Lock()

SUPPORTED_TYPES = {'.txt', '.md', '.pdf', '.docx', '.csv'}

# ---------- WHISPER ----------
WHISPER_ROOT      = os.path.expanduser(os.getenv("BLUR_WHISPER_ROOT", os.path.join(BLUR_HOME, "models", "whisper")))
WHISPER_MODEL_DIR = os.path.expanduser(os.getenv("BLUR_WHISPER_DIR",  os.path.join(WHISPER_ROOT, "medium.en-ct2")))
WHISPER_MODEL_ID  = os.getenv("BLUR_WHISPER_MODEL", "medium.en")
WHISPER_DEVICE    = os.getenv("BLUR_WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE   = os.getenv("BLUR_WHISPER_COMPUTE", "int8")
whisper_model = None

# ---------- MODE / TONE ----------
SLANG_LEXICON = ["nah man","ye","dope","vibe check","bullshit","slice it","fartin' chaos","edgy truth","stylish flip","i don't flinch"]
DREAM_BANS = {w.lower() for w in SLANG_LEXICON}

MODE_SEEDS = {
    "dream":     "You are DREAM: gentle, precise, grounded. Speak in 1–3 short lines. Do not include metrics, role tags, or internal logs.",
    "astrofuck": "You are ASTROFUCK: direct, punchy, kind. 1–3 punchy lines. No therapy metrics, no role tags.",
    "sentinel":  "You are SENTINEL: calm, specific, de-escalating. 1–3 lines. Avoid poetic language; be concrete.",
}

# ---------- APP ----------
app = FastAPI(default_response_class=ORJSONResponse)

# CORS: allow renderer locally or packed
if os.getenv("BLUR_PACKAGED") == "1":
    app.add_middleware(CORSMiddleware, allow_origin_regex=".*", allow_credentials=True, allow_methods=["*"], allow_headers=["*"], expose_headers=["X-Session-ID"])
else:
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:6969","http://127.0.0.1:6969"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"], expose_headers=["X-Session-ID"])

# ---------- CFG HELPERS ----------
def get_cfg(path: str, default=None):
    node = manifest
    for key in path.split('.'):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node

def resolve_path(path_str: str, homes_dict: dict):
    if not isinstance(path_str, str):
        return path_str
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

# ---------- EMBEDDING (singleton) ----------
def _ensure_embedder():
    """Idempotent embedder init (embedding=True, conservative n_ctx=512)."""
    global _embed_llm
    if _embed_llm is not None:
        return
    model_key = (get_cfg("memory.vector_store.embed_model", "snowflake_arctic_embed") or "snowflake_arctic_embed")
    model_cfg = (manifest.get("models", {}) or {}).get(model_key, {})
    model_path = resolve_path(model_cfg.get("path", ""), homes)
    if not model_path or not os.path.exists(model_path):
        raise RuntimeError(f"Embed model not found: key={model_key} path={model_path}")
    _embed_llm = Llama(
        model_path=model_path,
        embedding=True,
        n_ctx=512,  # embed models don't need large ctx
        n_gpu_layers=int(get_cfg("engines.llama_cpp.n_gpu_layers", 0)),
        logits_all=False,
        verbose=False,
    )
    logging.info(f"✅ Embedder loaded: {os.path.basename(model_path)}")

def _embedding_dim() -> int:
    global _embed_dim
    if _embed_dim is not None:
        return _embed_dim
    _ensure_embedder()
    probe = _embed_llm.create_embedding("dim?")['data'][0]['embedding']  # type: ignore
    _embed_dim = len(probe)
    return _embed_dim

def _encode(texts: List[str]) -> np.ndarray:
    _ensure_embedder()
    out = _embed_llm.create_embedding(texts)['data']  # type: ignore
    arr = np.asarray([d['embedding'] for d in out], dtype="float32")
    faiss.normalize_L2(arr)
    return arr

# ---------- PERSISTENT RAG ----------
class PersistentRAG:
    """
    JSONL rows require an 'id' that *matches* FAISS ids.
    On load:
      - If an index exists and ids mismatch, rebuild from JSONL so future ids are aligned.
    """
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
        if not self.chunks_path.exists():
            return []
        out = []
        with self.chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out

    def _write_jsonl(self, rows: List[Dict[str, Any]]):
        self.chunks_path.parent.mkdir(parents=True, exist_ok=True)
        with self.chunks_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _new_index(self) -> faiss.Index:
        dim = _embedding_dim()
        base = faiss.IndexFlatIP(dim)
        return faiss.IndexIDMap2(base)

    def _compact_by_ttl(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.ttl_days or self.ttl_days <= 0:
            return rows
        cutoff = time.time() - self.ttl_days * 86400
        kept = [r for r in rows if int(r.get("ts", 0)) >= cutoff]
        return kept or rows

    def _assign_sequential_ids_if_missing(self, rows: List[Dict[str, Any]]) -> None:
        any_missing = False
        for i, r in enumerate(rows):
            if "id" not in r:
                r["id"] = i
                any_missing = True
        if any_missing:
            self._write_jsonl(rows)

    def _ids_from_index(self, idx: faiss.Index) -> Optional[np.ndarray]:
        try:
            return faiss.vector_to_array(idx.id_map)  # type: ignore[attr-defined]
        except Exception:
            return None

    def _save_index(self):
        if self.index is None: return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

    def _rebuild_from_rows(self, rows: List[Dict[str, Any]]):
        self.index = self._new_index()
        texts = [r.get("content","") for r in rows]
        vecs = _encode(texts)
        ids_arr = np.asarray([int(r["id"]) for r in rows], dtype="int64")
        self.index.add_with_ids(vecs, ids_arr)
        self._save_index()

    def load(self):
        with self.lock:
            rows = self._read_jsonl()
            if self.auto_compact:
                rows = self._compact_by_ttl(rows)

            self._assign_sequential_ids_if_missing(rows)
            self.max_id = max([int(r["id"]) for r in rows], default=-1)

            need_rebuild = True
            try:
                if self.index_path.exists():
                    idx = faiss.read_index(str(self.index_path))
                    if isinstance(idx, faiss.IndexIDMap2):
                        faiss_ids = self._ids_from_index(idx)
                        if faiss_ids is not None and len(faiss_ids) == len(rows):
                            order = list(map(int, faiss_ids))
                            by_id = {int(r["id"]): r for r in rows}
                            rows = [by_id[i] for i in order if i in by_id]
                            self.max_id = int(max(faiss_ids)) if len(faiss_ids) else -1
                            self.index = idx
                            need_rebuild = False
                            logging.info(f"✅ Persistent RAG index loaded: {self.index.ntotal} vectors")
                        else:
                            logging.warning("[persistent-rag] id/count mismatch; will rebuild")
                    else:
                        logging.warning("[persistent-rag] index not IDMap2; will rebuild")
            except Exception as e:
                logging.warning(f"[persistent-rag] index read failed ({e}); will rebuild")

            self.chunks = rows
            if need_rebuild:
                if not self.chunks:
                    self.index = self._new_index()
                    self._save_index()
                    logging.info("✅ Persistent RAG created (empty).")
                else:
                    self._rebuild_from_rows(self.chunks)
                    logging.info(f"✅ Persistent RAG rebuilt: {len(self.chunks)} vectors")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        with self.lock:
            if self.index is None or self.index.ntotal == 0 or not self.chunks:
                return []
            qv = _encode([query]).astype("float32")
            k = min(top_k, self.index.ntotal)
            D, I = self.index.search(qv, k)
            out = []
            idx_by_id = {int(r["id"]): r for r in self.chunks}
            for id_ in I[0]:
                r = idx_by_id.get(int(id_))
                if r:
                    out.append(r)
            return out

    def add_rows(self, rows: List[Dict[str, Any]]) -> int:
        if not rows: return 0
        with self.lock:
            for r in rows:
                self.max_id += 1
                r["id"] = self.max_id
            vecs = _encode([r["content"] for r in rows])
            ids_arr = np.asarray([int(r["id"]) for r in rows], dtype="int64")
            if self.index is None:
                self.index = self._new_index()
            self.index.add_with_ids(vecs, ids_arr)
            self.chunks.extend(rows)
            self._write_jsonl(self.chunks)
            self._save_index()
            return len(rows)

persistent_rag: Optional[PersistentRAG] = None

# ---------- USER MEMORY ----------
def _chunk_memory_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    if not text: return []
    chunks, i = [], 0
    while i < len(text):
        j = min(len(text), i + chunk_size)
        cut = text[i:j]
        k = max(cut.rfind('. '), cut.rfind('。'), cut.rfind('! '), cut.rfind('? '))
        if k != -1 and (i + k + 1 - i) > chunk_size * 0.6:
            j = i + k + 1
        chunks.append(text[i:j].strip())
        i = max(j - overlap, i + 1)
    return [c for c in chunks if len(c) > 30]

def load_user_memory():
    global user_memory_chunks
    try:
        if os.path.exists(USER_MEMORY_FILE):
            with open(USER_MEMORY_FILE, 'r') as f:
                user_memory_chunks = json.load(f)
            logging.info(f"✅ Loaded user memory for {len(user_memory_chunks)} users")
        else:
            user_memory_chunks = {}
    except Exception as e:
        logging.error(f"🛑 Failed to load user memory: {e}")
        user_memory_chunks = {}

def save_user_memory():
    try:
        Path(USER_MEMORY_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(USER_MEMORY_FILE, 'w') as f:
            json.dump(user_memory_chunks, f, indent=2)
    except Exception as e:
        logging.error(f"🛑 Failed to save user memory: {e}")

def retrieve_user_memory(username: Optional[str], query: str, top_k: int = 3) -> List[str]:
    if not username or username not in user_memory_chunks:
        return []
    try:
        _ensure_embedder()
        qv = _encode([query])[0]
        scored = []
        for ch in user_memory_chunks[username]:
            cv = _encode([ch["content"]])[0]
            scored.append((float(np.dot(qv, cv)), ch["content"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for s, c in scored[:top_k] if s > 0.3]
    except Exception as e:
        logging.error(f"User memory retrieval error: {e}")
        return []

# ---------- FILE PARSE / CHUNK ----------
def _read_file_bytes(file_bytes: bytes, filename: str) -> str:
    name = (filename or '').lower()
    try:
        if name.endswith('.txt') or name.endswith('.md'):
            return file_bytes.decode('utf-8', errors='ignore')
        elif name.endswith('.pdf'):
            import fitz
            text = []
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                if doc.is_encrypted:
                    raise ValueError("PDF is encrypted")
                for page in doc:
                    text.append(page.get_text("text"))
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

# ---------- EPHEMERAL RAG MAINT ----------
def _rebuild_index_from_chunks():
    global rag_index
    dim = _embedding_dim()
    idx = faiss.IndexFlatIP(dim)
    if knowledge_chunks:
        vecs = _encode([m["content"] for m in knowledge_chunks])
        idx.add(vecs)
    rag_index = idx

def _evict_and_ttl_gc(session_id: Optional[str] = None):
    global knowledge_chunks
    now = int(time.time())
    knowledge_chunks = [m for m in knowledge_chunks if (now - int(m.get("ts", 0))) <= EPHEMERAL_TTL_SECONDS]
    if session_id:
        sess = [m for m in knowledge_chunks if m.get("session_id") == session_id]
        if len(sess) > MAX_RAG_CHUNKS_PER_SESSION:
            excess = len(sess) - MAX_RAG_CHUNKS_PER_SESSION
            to_drop = set(id(m) for m in sorted(sess, key=lambda x: x.get("ts", 0))[:excess])
            knowledge_chunks = [m for m in knowledge_chunks if id(m) not in to_drop]
    if len(knowledge_chunks) > MAX_RAG_CHUNKS_TOTAL:
        drop_n = len(knowledge_chunks) - MAX_RAG_CHUNKS_TOTAL
        knowledge_chunks = sorted(knowledge_chunks, key=lambda x: x.get("ts", 0))[drop_n:]
    _rebuild_index_from_chunks()

# ---------- LANGUAGE DETECTION ----------
try:
    from langdetect import detect as _ld_detect
    from langdetect import DetectorFactory as _LDFactory
    _LDFactory.seed = 42
    _HAS_LANGDETECT = True
except Exception:
    _HAS_LANGDETECT = False

_LANG_CODE_TO_NAME = {
    "en":"English","es":"Spanish","pt":"Portuguese","fr":"French","de":"German","it":"Italian","nl":"Dutch","sv":"Swedish",
    "pl":"Polish","cs":"Czech","ru":"Russian","uk":"Ukrainian","tr":"Turkish","ar":"Arabic","he":"Hebrew","fa":"Persian",
    "hi":"Hindi","bn":"Bengali","ur":"Urdu","ta":"Tamil","te":"Telugu","ml":"Malayalam","kn":"Kannada","th":"Thai",
    "vi":"Vietnamese","id":"Indonesian","ms":"Malay","ja":"Japanese","ko":"Korean","zh-cn":"Chinese","zh-tw":"Chinese (Traditional)"
}

_SLANG_EN_GREETINGS = re.compile(r"^(yo+|ye+|ya+|sup|wass?up|ayy+|hey+|hiya)\b", re.I)

def detect_language_name(text: str) -> str:
    t = (text or "").strip()
    if not t or len(t) <= 3 or _SLANG_EN_GREETINGS.match(t):
        return "English"
    if re.fullmatch(r"[ -~\s]+", t) and len(t.split()) <= 4:
        return "English"
    if _HAS_LANGDETECT:
        try:
            code = _ld_detect(t)
            if code == "zh": code = "zh-cn"
            return _LANG_CODE_TO_NAME.get(code, "English")
        except Exception:
            pass
    if re.search(r"[\u3040-\u30FF\u4E00-\u9FFF]", t): return "Japanese"
    if re.search(r"[\uAC00-\uD7A3]", t): return "Korean"
    if re.search(r"[\u0590-\u05FF]", t): return "Hebrew"
    if re.search(r"[\u0600-\u06FF]", t): return "Arabic"
    return "English"

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

# ---------- STYLE / SANITIZE ----------
def sanitize_for_audience(text: str, audience: str) -> str:
    if (get_cfg("release.enabled", False) is not True) or audience != "public":
        return text
    aliases = (get_cfg("release_aliases", {}) or {})
    for k, v in aliases.items():
        text = re.sub(rf"(?:^|(?<!\w)){re.escape(k)}(?:(?!\w)|$)", v, text)
    for t in (get_cfg("filters.internal_only", []) or []):
        text = re.sub(rf"(?:^|(?<!\w)){re.escape(t)}(?:(?!\w)|$)", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()

_ALLOWED_GLYPHS = set("↺✶⛧🜃🜂∴∵∞")
_EMOJI_BLOCK = re.compile(r"[\U0001F300-\U0001FAFF]")
def _strip_emoji_except_glyphs(text: str) -> str:
    return _EMOJI_BLOCK.sub(lambda m: m.group(0) if m.group(0) in _ALLOWED_GLYPHS else "", text)

def astrofuck_ensure_slang(text: str, active_lang: str) -> str:
    if (active_lang or "English").lower() != "english": return text
    low = text.lower()
    return text if any(tok in low for tok in SLANG_LEXICON) else ("Dope — let's slice it clean. " + text).strip()

def enforce_persona_ending(text: str, mode: str) -> str:
    endings = get_cfg(f"range.modes.{mode}.endings", []) or []
    return text if not endings or random.random() < 0.5 else text.rstrip() + "\n" + random.choice(endings)

# ---------- PROMPT ASSEMBLY ----------
def build_messages(mode: str, system_seed: str, history: List[Dict[str, str]], user_text: str, context: str, lang_name: str):
    sys = system_seed
    if lang_name and lang_name not in ("English", "Unknown"):
        sys = f"CRITICAL: The user is writing in {lang_name}. Respond ONLY in {lang_name}.\n\n{sys}"
    if context:
        sys += "\n\n--- Context ---\n" + context
    msgs = [{"role":"system","content":sys}]
    for turn in history:
        u = (turn.get("user") or "").strip(); a = (turn.get("assistant") or "").strip()
        if u: msgs.append({"role":"user","content":u})
        if a: msgs.append({"role":"assistant","content":a})
    msgs.append({"role":"user","content":user_text})
    return msgs

# ---------- RETRIEVAL (persistent ⊕ ephemeral ⊕ memory) ----------
def retrieve_context_blend(query: str, session_id: Optional[str], thread_id: Optional[str], username: Optional[str], top_k: int = 5) -> str:
    parts: List[str] = []

    persistent_parts = []
    if persistent_rag is not None:
        try:
            rows = persistent_rag.search(query, top_k=min(3, top_k))
            for r in rows:
                persistent_parts.append(r.get("content","").strip())
        except Exception as e:
            logging.error(f"Persistent RAG search error: {e}")
    if persistent_parts:
        parts.append("--- Persistent Knowledge ---\n" + "\n\n".join(persistent_parts))

    if rag_index is not None and knowledge_chunks and rag_index.ntotal > 0:
        try:
            qv = _encode([query]).astype('float32')
            k = min(max(2, top_k - len(persistent_parts)), rag_index.ntotal)
            D, I = rag_index.search(qv, k)
            eph = []
            for i in I[0]:
                if 0 <= i < len(knowledge_chunks):
                    meta = knowledge_chunks[i]
                    if session_id and meta.get("session_id") != session_id: continue
                    if thread_id and meta.get("thread_id") != thread_id: continue
                    eph.append(meta.get("content",""))
            if eph:
                parts.append("--- Ephemeral Context ---\n" + "\n\n".join(eph))
        except Exception as e:
            logging.error(f"Ephemeral RAG retrieval error: {e}")

    um = retrieve_user_memory(username, query, top_k=3)
    if um:
        parts.append("--- Personal Memory Fragments ---\n" + "\n\n".join(um))

    return "\n\n".join(parts).strip()

# ---------- SAFE LOAD LLM (auto ctx backoff) ----------
def _try_load_llama(model_path: str, desired_ctx: int, n_gpu_layers: int) -> Optional[Llama]:
    try:
        return Llama(
            model_path=model_path,
            n_ctx=int(desired_ctx),
            n_gpu_layers=int(n_gpu_layers),
            embedding=False,
            logits_all=False,
            verbose=False,
        )
    except Exception as e:
        msg = str(e)
        if "n_ctx_per_seq" in msg or "n_ctx_train" in msg or "context overflow" in msg:
            return None
        raise

def _load_llama_with_backoff(model_path: str, requested_ctx: int, n_gpu_layers: int) -> Llama:
    ladder = [requested_ctx, 8192, 4096, 2048, 1024, 512]
    tried = []
    for ctx in ladder:
        if ctx in tried: continue
        tried.append(ctx)
        llm = _try_load_llama(model_path, ctx, n_gpu_layers)
        if llm:
            logging.info(f"[models] Loaded with n_ctx={ctx} (requested={requested_ctx})")
            return llm
    raise RuntimeError(
        f"Failed to load model at {model_path} — context too large for training window. "
        f"Tried ctx={tried}. Lower engines.llama_cpp.python_n_ctx in config.yaml."
    )

# ---------- MODELS ----------
def load_llm_from_config(model_config: dict, model_key: str) -> bool:
    """Idempotent: avoids reloading."""
    if model_key in llm_vessels:
        return True
    if not isinstance(model_config, dict):
        logging.error("Config for model '%s' is not a dict.", model_key)
        return False
    if model_config.get('engine') != 'llama_cpp':
        return False

    family = (model_config.get('family') or '').lower()
    model_path = resolve_path(model_config.get('path', ''), homes)
    logging.info("[models] %s -> %s exists=%s", model_key, model_path, os.path.exists(model_path))

    if not model_path or not os.path.exists(model_path):
        logging.error("Model not found: %s", model_path)
        return False

    try:
        requested_n_ctx = int(get_cfg("engines.llama_cpp.python_n_ctx", 8192))
        n_gpu_layers = int(get_cfg("engines.llama_cpp.n_gpu_layers", 0))
        llm = _load_llama_with_backoff(model_path, requested_n_ctx, n_gpu_layers)
        llm_vessels[model_key] = llm
        vessel_details[model_key] = {"family": family, "params": model_config.get("params", {}), "n_ctx": requested_n_ctx}
        logging.info("Vessel '%s' online (%s)", model_key, family or "chat")
        return True
    except Exception as e:
        logging.error("Failed to load '%s': %s", model_key, e)
        return False

# ---------- ENDPOINT MODELS ----------
class RequestModel(BaseModel):
    prompt: str
    mode: str
    turn: Optional[int] = 0
    session_id: Optional[str] = None
    new_session: bool = False
    force_lang: Optional[str] = None
    username: Optional[str] = None

class MemoryPayload(BaseModel):
    user: str
    memory: str = ""
    instructions: str = ""
    audience: str = "internal"

# ---------- STREAM GEN ----------
async def generate_response_stream(session: Dict, request: RequestModel):
    user_text = (request.prompt or "").strip()
    req_mode = (request.mode or "dream").strip().lower()
    mode = req_mode if req_mode in MODE_SEEDS else "dream"

    lang = language_hysteresis_update_lang(session, user_text)

    chat_key = get_cfg("chat.vessel_key", "qwen3_4b_unified")
    chat_llm = llm_vessels.get(chat_key)
    if not chat_llm:
        yield "Model not loaded."
        return

    context = retrieve_context_blend(user_text, session.get("id"), None, session.get("username") or request.username, top_k=5)
    audience = (session.get("audience") or get_cfg("release.audience_default","internal")).lower()

    history_pairs = session.get("history", [])[-int(get_cfg("assembly.history_turns", DEFAULT_HISTORY_TURNS_FOR_PROMPT)) :]
    msgs = build_messages(mode, MODE_SEEDS.get(mode, MODE_SEEDS["dream"]), history_pairs, user_text, context, lang)

    mp = get_cfg(f"range.modes.{mode}.params", {}) or {}
    params = dict(
        messages=msgs,
        temperature=float(mp.get("temperature",0.6)),
        top_p=float(mp.get("top_p",0.9)),
        repeat_penalty=float(mp.get("repeat_penalty",1.12)),
        max_tokens=int(mp.get("n_predict",768)),
        stop=["</s>","<|im_end|>","[INST]","[/INST]"],
        stream=True
    )

    response_text = ""
    try:
        for chunk in chat_llm.create_chat_completion(**params):
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            piece = delta.get("content")
            if not piece: continue
            piece = piece.replace("\r\n","\n")
            safe_piece = sanitize_for_audience(piece, audience)
            response_text += safe_piece
            yield safe_piece
    except Exception as e:
        logging.error("Generation error: %s", e)
        yield "Internal error."
        return

    final_text = (response_text.strip() or "…")
    if mode == "dream":
        t = final_text
        for w in DREAM_BANS: t = re.sub(rf"(?i)\b{re.escape(w)}\b","", t)
        t = re.sub(r"[ \t]{2,}"," ", t); t = re.sub(r"([!?])\1{1,}", r"\1", t)
        final_text = t.strip()
    elif mode == "astrofuck":
        final_text = astrofuck_ensure_slang(final_text, lang)
    final_text = _strip_emoji_except_glyphs(final_text)
    final_text = enforce_persona_ending(final_text, mode)

    hist = session.setdefault("history", [])
    hist.append({"user": user_text, "assistant": final_text})
    session["history"] = hist[-int(get_cfg("assembly.keep_history", DEFAULT_KEEP_HISTORY)) :]
    session["turn"] = int(session.get("turn", 0)) + 1

    # autosave immediately
    try:
        save_sessions()
    except Exception:
        logging.error("autosave sessions failed", exc_info=True)

# ---------- INGEST ROUTES (ephemeral) ----------
if _HAS_MULTIPART:
    @app.post("/rag/ingest")
    async def rag_ingest(
        files: List[UploadFile] = File(...),
        source: Optional[str] = Form(None),
        thread_id: Optional[str] = Form(None),
        http: FastAPIRequest = None
    ):
        if not EPHEMERAL_RAG_ENABLED:
            return JSONResponse({"ok": False, "error": "Ephemeral RAG disabled"}, status_code=503)

        global rag_index
        try:
            _ensure_embedder()
            if rag_index is None:
                rag_index = faiss.IndexFlatIP(_embedding_dim())

            sid = (http.cookies.get("blur_sid") if http else None) or last_seen_session_id
            total_added, results = 0, []

            for uf in files:
                name = uf.filename or "unnamed"
                ext = os.path.splitext(name)[1].lower()

                if ext not in SUPPORTED_TYPES:
                    results.append({"file": name, "status": "skipped", "reason": f"unsupported {ext}"})
                    continue

                data = await uf.read()
                if not data:
                    results.append({"file": name, "status": "skipped", "reason": "empty"})
                    continue

                try:
                    text = _read_file_bytes(data, name)
                    chunks = _chunk_text(text)
                    if not chunks:
                        results.append({"file": name, "status": "skipped", "reason": "no text"})
                        continue

                    vecs = _encode(chunks)
                    rows = [{
                        "source": (source or name),
                        "content": c,
                        "ts": int(time.time()),
                        "session_id": sid,
                        "thread_id": thread_id,
                        "filename": name
                    } for c in chunks]

                    with rag_lock:
                        knowledge_chunks.extend(rows)
                        rag_index.add(vecs.astype('float32', copy=False))
                        _evict_and_ttl_gc(session_id=sid)

                    total_added += len(chunks)
                    results.append({"file": name, "status": "ok", "chunks": len(chunks)})
                except Exception as e:
                    logging.error("Ingest error for %s: %s", name, e)
                    results.append({"file": name, "status": "error", "reason": str(e)})

            return {"ok": True, "added": total_added, "files": results}

        except Exception as e:
            logging.error("RAG ingest error: %s", e)
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
else:
    @app.post("/rag/ingest")
    async def rag_ingest_unavailable():
        return JSONResponse({"ok": False,"error": "File upload endpoint unavailable: install python-multipart","pip": "pip install python-multipart"}, status_code=503)

# ---------- PERSISTENT RAG ROUTES ----------
@app.get("/persistent/status")
def persistent_status():
    pv = persistent_rag.index.ntotal if (persistent_rag and persistent_rag.index) else 0
    return {
        "ok": True,
        "vectors": pv,
        "index_path": getattr(persistent_rag, "index_path", None) and str(persistent_rag.index_path),
        "chunks_path": getattr(persistent_rag, "chunks_path", None) and str(persistent_rag.chunks_path),
    }

@app.post("/persistent/reload")
def persistent_reload():
    try:
        if persistent_rag:
            persistent_rag.load()
            return {"ok": True, "vectors": persistent_rag.index.ntotal if persistent_rag.index else 0}
        return {"ok": False, "error": "persistent_rag not configured"}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/persistent/search")
def persistent_search(q: str, k: int = 5):
    try:
        if not persistent_rag: return {"ok": True, "results": []}
        rows = persistent_rag.search(q, top_k=int(k))
        return {"ok": True, "results": rows}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# ---------- ASR ----------
def _init_whisper():
    global whisper_model
    if WhisperModel is None:
        logging.info("Whisper unavailable (faster-whisper not installed).")
        whisper_model = None
        return
    try:
        if os.path.isdir(WHISPER_MODEL_DIR) and os.path.exists(os.path.join(WHISPER_MODEL_DIR, "config.json")):
            logging.info(f"🎙️ Loading faster-whisper from local dir: {WHISPER_MODEL_DIR}")
            whisper_model = WhisperModel(WHISPER_MODEL_DIR, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
        else:
            logging.info(f"🎙️ Loading faster-whisper id '{WHISPER_MODEL_ID}' to root '{WHISPER_ROOT}'")
            whisper_model = WhisperModel(WHISPER_MODEL_ID, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE, download_root=WHISPER_ROOT)
        logging.info("✅ Whisper online.")
    except Exception as e:
        logging.error(f"🛑 Whisper failed: {e}"); whisper_model = None

def _to_wav16k_mono(webm_bytes: bytes) -> bytes:
    audio = AudioSegment.from_file(io.BytesIO(webm_bytes))
    audio = audio.set_channels(1).set_frame_rate(16000)
    buf = io.BytesIO(); audio.export(buf, format="wav")
    return buf.getvalue()

if _HAS_MULTIPART:
    @app.post("/transcribe")
    async def transcribe(file: UploadFile = File(...)):
        if whisper_model is None:
            return JSONResponse({"text": "", "error": "Whisper not loaded"}, status_code=503)
        try:
            raw = await file.read()
            fname = (file.filename or "").lower()
            wav_bytes = raw if fname.endswith(".wav") else _to_wav16k_mono(raw)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_bytes); tmp_path = tmp.name
            segments, info = whisper_model.transcribe(tmp_path, vad_filter=True, beam_size=5, word_timestamps=False)
            text = "".join(seg.text for seg in segments).strip()
            try: os.remove(tmp_path)
            except Exception: pass
            return {"text": text}
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return JSONResponse({"text": "", "error": str(e)}, status_code=500)
else:
    @app.post("/transcribe")
    async def transcribe_unavailable():
        return JSONResponse({"ok": False,"error": "File upload endpoint unavailable: install python-multipart","pip": "pip install python-multipart"}, status_code=503)

# ---------- HEALTH / SESSION ----------
@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "whisper": whisper_model is not None,
        "sessions": len(sessions),
        "persistent_vectors": (persistent_rag.index.ntotal if (persistent_rag and persistent_rag.index) else 0),
        "ephemeral_vectors": (rag_index.ntotal if rag_index is not None else 0),
        "vessels": list(llm_vessels.keys()),
        "chat_key": get_cfg("chat.vessel_key", "qwen3_4b_unified"),
        "paths": {
            "BLUR_HOME": BLUR_HOME,
            "STATE_DIR": STATE_DIR,
            "MANIFEST_PATH": MANIFEST_PATH,
            "WHISPER_ROOT": WHISPER_ROOT,
            "WHISPER_MODEL_DIR": WHISPER_MODEL_DIR,
        },
        "multipart": _HAS_MULTIPART,
    }

@app.get("/session")
async def get_new_session():
    global last_seen_session_id
    sid = str(uuid.uuid4())
    sessions[sid] = {
        "id": sid,
        "mode": "dream",
        "history": [],
        "turn": 0,
        "username": None,
        "wants": {"tone":"dream", "defaultMode":"dream", "ragAutoIngest":True},
        "audience": get_cfg("release.audience_default", "internal")
    }
    last_seen_session_id = sid
    resp = JSONResponse({"session_id": sid})
    resp.headers["X-Session-ID"] = sid
    resp.set_cookie("blur_sid", sid, path="/", httponly=True, samesite="lax")
    return resp

@app.post("/generate_response")
async def handle_generate_request(req: RequestModel, http: FastAPIRequest):
    global last_seen_session_id
    sid = (req.session_id or http.headers.get("X-Session-ID") or http.cookies.get("blur_sid") or last_seen_session_id or str(uuid.uuid4())).strip()
    if req.new_session: sid = str(uuid.uuid4())
    if sid not in sessions:
        sessions[sid] = {"id": sid, "mode": req.mode, "history": [], "turn": 0, "username": req.username,
                         "wants": {"tone":"dream","defaultMode":"dream","ragAutoIngest":True},
                         "audience": get_cfg("release.audience_default","internal")}
    sessions[sid]["mode"] = req.mode; last_seen_session_id = sid
    resp = StreamingResponse(generate_response_stream(sessions[sid], req), media_type="text/plain")
    resp.headers["X-Session-ID"] = sid
    resp.set_cookie("blur_sid", sid, path="/", httponly=True, samesite="lax")
    return resp

@app.get("/status")
async def get_status():
    chat_key = get_cfg("chat.vessel_key", "qwen3_4b_unified")
    is_core_ready = chat_key in llm_vessels and llm_vessels.get(chat_key) is not None
    pv = persistent_rag.index.ntotal if (persistent_rag and persistent_rag.index) else 0
    ev = rag_index.ntotal if rag_index is not None else 0
    return {
        "status": "ready" if is_core_ready else "loading",
        "persistent_vectors": pv,
        "ephemeral_vectors": ev,
        "users_with_memory": len(user_memory_chunks),
        "persistent_index_path": getattr(persistent_rag, "index_path", None) and str(persistent_rag.index_path),
        "persistent_chunks_path": getattr(persistent_rag, "chunks_path", None) and str(persistent_rag.chunks_path),
    }

# ---------- PERSISTENCE ----------
STATE_DIR = os.path.expanduser(os.getenv("BLUR_STATE_DIR", "~/.blur"))
SESSIONS_FILE = os.path.join(STATE_DIR, "sessions.json")
USER_MEMORY_FILE = os.path.join(STATE_DIR, "user_memory.json")

def _ensure_state_dir():
    try:
        Path(STATE_DIR).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to create state dir {STATE_DIR}: {e}")

def load_sessions():
    """Load sessions from disk and normalize fields."""
    global sessions
    _ensure_state_dir()
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r") as f:
                loaded = json.load(f)
            sessions = {}
            keep_n = int(get_cfg("assembly.keep_history", DEFAULT_KEEP_HISTORY))
            for sid, data in (loaded or {}).items():
                hist = data.get("history", [])
                if isinstance(hist, list):
                    data["history"] = hist[-keep_n:]
                else:
                    data["history"] = []
                data.setdefault("username", None)
                data.setdefault("wants", {"tone": "dream", "defaultMode": "dream", "ragAutoIngest": True})
                data.setdefault("audience", get_cfg("release.audience_default", "internal"))
                data.setdefault("turn", len(data["history"]))
                sessions[sid] = data
            logging.info("Loaded %d sessions from %s", len(sessions), SESSIONS_FILE)
        except Exception as e:
            logging.error("Failed to load sessions: %s", e)
            sessions = {}
    else:
        sessions = {}

def save_sessions():
    """Persist sessions atomically to avoid tearing."""
    try:
        _ensure_state_dir()
        tmp = SESSIONS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(sessions, f, indent=2)
        os.replace(tmp, SESSIONS_FILE)
    except Exception as e:
        logging.error("Failed to save sessions: %s", e)

# ---------- AS YOU STREAM, APPEND + AUTOSAVE ----------
# (keep this inside generate_response_stream — shown here for context)
# hist = session.setdefault("history", [])
# hist.append({"user": user_text, "assistant": final_text})
# session["history"] = hist[-int(get_cfg("assembly.keep_history", DEFAULT_KEEP_HISTORY)) :]
# session["turn"] = int(session.get("turn", 0)) + 1
# try:
#     save_sessions()
# except Exception:
#     logging.error("autosave sessions failed", exc_info=True)

# ---------- OPTIONAL: USER MEMORY ENDPOINTS ----------
class MemoryUpsert(BaseModel):
    user: str
    text: str

@app.post("/memory/upsert")
def memory_upsert(payload: MemoryUpsert):
    """Simple append to user memory; chunks automatically derived."""
    if not payload.user or not payload.text.strip():
        return JSONResponse({"ok": False, "error": "user and text required"}, status_code=400)
    chunks = _chunk_memory_text(payload.text.strip())
    if not chunks:
        return {"ok": True, "added": 0}
    with user_memory_lock:
        arr = user_memory_chunks.setdefault(payload.user, [])
        now = int(time.time())
        for ch in chunks:
            arr.append({"content": ch, "ts": now})
        # cap + TTL trim
        cutoff = now - USER_MEMORY_TTL_DAYS * 86400
        arr[:] = [r for r in arr[-MAX_USER_MEMORY_CHUNKS:] if r.get("ts", 0) >= cutoff]
    try:
        save_user_memory()
    except Exception:
        logging.error("Failed to save user memory", exc_info=True)
    return {"ok": True, "added": len(chunks)}

# ---------- DEBUG / DIAGNOSTICS ----------
@app.get("/debug/paths")
def debug_paths():
    chat_key = get_cfg("chat.vessel_key", "qwen3_4b_unified")
    return {
        "BLUR_HOME": BLUR_HOME,
        "MANIFEST_PATH": MANIFEST_PATH,
        "STATE_DIR": STATE_DIR,
        "SESSIONS_FILE": SESSIONS_FILE,
        "USER_MEMORY_FILE": USER_MEMORY_FILE,
        "chat_key": chat_key,
        "known_vessels": list(llm_vessels.keys()),
        "embed_model_path": resolve_path((manifest.get("models", {}) or {}).get(
            get_cfg("memory.vector_store.embed_model", "snowflake_arctic_embed"), {}
        ).get("path", ""), homes),
        "persistent": {
            "index": getattr(persistent_rag, "index_path", None) and str(persistent_rag.index_path),
            "chunks": getattr(persistent_rag, "chunks_path", None) and str(persistent_rag.chunks_path),
            "vectors": (persistent_rag.index.ntotal if (persistent_rag and persistent_rag.index) else 0),
        },
    }

@app.get("/config/current")
def config_current():
    """Return the resolved manifest (safe subset) + key params to confirm tone diffs."""
    return {
        "manifest_path": MANIFEST_PATH,
        "chat": manifest.get("chat", {}),
        "range_modes": {
            k: {"params": v.get("params", {}), "endings": v.get("endings", [])}
            for k, v in (manifest.get("range", {}).get("modes", {}) or {}).items()
        },
        "assembly": {
            "history_turns": get_cfg("assembly.history_turns", DEFAULT_HISTORY_TURNS_FOR_PROMPT),
            "keep_history": get_cfg("assembly.keep_history", DEFAULT_KEEP_HISTORY),
        },
        "language": {
            "hysteresis_consecutive": get_cfg("language.hysteresis_consecutive", 3),
            "default_if_unknown": get_cfg("language.default_if_unknown", "English"),
        },
    }

# ---------- STARTUP / SHUTDOWN ----------
@app.on_event("startup")
async def startup_event():
    global manifest, homes, rag_index, persistent_rag
    logging.info(f"Startup: loading manifest: {MANIFEST_PATH}")
    try:
        # ensure state dir exists early
        _ensure_state_dir()

        if not os.path.exists(MANIFEST_PATH):
            logging.error(f"🛑 FATAL: Manifest file not found at '{MANIFEST_PATH}'. Core will not start.")
            raise FileNotFoundError(f"Manifest file not found at the required path: {MANIFEST_PATH}")

        with open(MANIFEST_PATH, 'r') as f:
            manifest = yaml.safe_load(f) or {}

        # default chat key
        manifest.setdefault("chat", {}).setdefault("vessel_key", "qwen3_4b_unified")

        homes_local = (manifest.get('meta', {}) or {}).get('homes', {}) or {}
        homes_local.setdefault("blur_home", BLUR_HOME)
        homes_local.setdefault("models", os.path.join(homes_local["blur_home"], "models"))
        homes_local.setdefault("pipes",  os.path.join(homes_local["blur_home"], "run", "pipes"))
        homes_local.setdefault("data",   os.path.join(homes_local["blur_home"], "core"))
        homes.clear(); homes.update(resolve_homes_recursive(homes_local))

        # models (idempotent)
        for key, cfg in (manifest.get('models', {}) or {}).items():
            await _VESSEL_LOCK.acquire()
            try:
                load_llm_from_config(cfg, key)
            finally:
                _VESSEL_LOCK.release()

        # chat vessel check
        chat_key = get_cfg("chat.vessel_key", "qwen3_4b_unified")
        if chat_key not in llm_vessels:
            logging.error(f"🛑 Chat vessel '{chat_key}' not loaded. Known vessels: {list(llm_vessels.keys())}")
        else:
            logging.info(f"✅ Chat vessel ready: {chat_key}")

        # embedder + ephemeral idx
        await _embed_lock.acquire()
        try:
            _ensure_embedder()
        finally:
            _embed_lock.release()
        rag_index = faiss.IndexFlatIP(_embedding_dim())

        # Persistent RAG init (BLUR_HOME-first)
        idx_path = resolve_path(get_cfg("memory.vector_store.path", ""), homes) or os.path.join(BLUR_HOME, "core", "ouinet", "blurchive", "ecosystem", "blur_knowledge.index")
        ch_path  = resolve_path(get_cfg("memory.vector_store.chunks_path", ""), homes) or os.path.join(BLUR_HOME, "core", "ouinet", "blurchive", "ecosystem", "knowledge_chunks.jsonl")
        ttl_days = int(get_cfg("memory.vector_store.ttl_days_persistent", 0) or 0)
        auto_compact = bool(get_cfg("memory.vector_store.auto_compact_on_start", False))
        persistent_rag_local = PersistentRAG(index_path=idx_path, chunks_path=ch_path, ttl_days=ttl_days, auto_compact=auto_compact)
        try:
            persistent_rag_local.load()
        except Exception as e:
            logging.error(f"Persistent RAG load failed: {e}")
        persistent_rag = persistent_rag_local
        logging.info(f"Persistent store: index='{idx_path}', chunks='{ch_path}'")

        load_sessions()
        load_user_memory()
        _init_whisper()
        logging.info("Core ready (v7.1).")
    except Exception as e:
        logging.error(f"FATAL on startup: {e}")
        raise e

@app.on_event("shutdown")
def shutdown_event():
    save_sessions()
    save_user_memory()
    logging.info("State saved on shutdown.")

# ---------- MAIN ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("BLUR_CORE_PORT", "8000")))
