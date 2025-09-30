#!/usr/bin/env python3
# convo_chat_core.py — Reforged v9.2+ (Qwen-only, Strict RAG, Live Stream, Fixed Ephemeral Ingest)
# TONE-SPLIT PATCH: strict mode-scoped history + KV per (mode,lang) + hard tone masks
# - Manifest is the single source of truth, but BLUR_HOME fallbacks are restored for persistent RAG paths.
# - Chat vessel: chat.vessel_key (qwen3_4b_unified). No other models loaded.
# - Persistent RAG: memory.vector_store.{path,chunks_path,embed_model,ttl_days_persistent,auto_compact_on_start}
#   * Rebuilds index if: missing, id/count mismatch, or dim != embedder dim
#   * Enforces JSONL ids stable [0..N-1] (auto-fix missing ids)
# - Ephemeral RAG:
#   * /rag/ingest (multipart) re-enabled; caches vectors at ingest; respects session_id & thread_id
#   * GC uses rag.ephemeral.{max_total,max_per_session,ttl_seconds}; rebuild only when list size changes
# - Assembly sizes: assembly.history_turns / keep_history
# - Language hysteresis: language.hysteresis_consecutive
# - Mode params/endings from range.modes.{dream,astrofuck,sentinel}
# - Acheflip hook stubbed from philosophy.acheflip.* (no external runner required)
# - Streaming: SSE (live), sets X-Session-ID + X-TTFT-Last

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

# --- multipart upload (A) ---
try:
    from fastapi import UploadFile, File, Form
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

# --- llama.cpp kw guard ---
def _llama_accepts_kw(llm: Llama, kw: str) -> bool:
    try:
        sig = inspect.signature(llm.create_chat_completion)
        return kw in sig.parameters
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
        model_path=model_path,
        embedding=True,
        n_ctx=512,
        n_gpu_layers=int(get_cfg("engines.llama_cpp.n_gpu_layers", 0) or 0),
        n_threads=max(2, os.cpu_count() or 4),
        n_batch=int(get_cfg("engines.llama_cpp.n_batch", 512) or 512),
        use_mmap=True,
        logits_all=False,
        verbose=False,
    )
    logging.info(f"✅ Embedder online: {os.path.basename(model_path)}")

def _embedding_dim() -> int:
    global _embed_dim
    if _embed_dim is not None: return _embed_dim
    _ensure_embedder()
    probe = _embed_llm.create_embedding(input=["dim?"])['data'][0]['embedding']  # <- input=[...]
    _embed_dim = len(probe)
    return _embed_dim

def _encode(texts: List[str]) -> np.ndarray:
    _ensure_embedder()
    # accept str or list
    if isinstance(texts, str):
        texts = [texts]
    out = _embed_llm.create_embedding(input=texts)['data']  # <- input=...
    arr = np.asarray([d['embedding'] for d in out], dtype="float32")
    faiss.normalize_L2(arr)
    return arr

def _memvec_get(text: str) -> np.ndarray:
    key = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    v = _MEMVEC.get(key)
    if v is not None: return v
    v = _encode([text])[0]
    _MEMVEC[key] = v
    return v

_recent_qv_cache: Dict[str, np.ndarray] = {}
def _query_vec_cached(query: str, sid: Optional[str], tid: Optional[str]) -> np.ndarray:
    key = f"{sid}|{tid}|{hashlib.sha1((query or '').encode('utf-8','ignore')).hexdigest()}"
    v = _recent_qv_cache.get(key)
    if v is not None:
        return v
    v = _encode([query])[0]
    _recent_qv_cache[key] = v
    if len(_recent_qv_cache) > 128:
        for _ in range(len(_recent_qv_cache) - 128):
            _recent_qv_cache.pop(next(iter(_recent_qv_cache)))
    return v

# ---------- PERSISTENT RAG ----------
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
                try:
                    out.append(json.loads(line))
                except Exception:
                    m = re.search(r'\{.*\}', line)
                    if m:
                        try: out.append(json.loads(m.group(0)))
                        except Exception: pass
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
        if not self.ttl_days or self.ttl_days <= 0: return rows
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
                    if getattr(idx, 'd', None) != _embedding_dim():
                        logging.warning("[persistent-rag] dim mismatch; rebuild")
                    else:
                        faiss_ids = self._ids_from_index(idx) if isinstance(idx, faiss.IndexIDMap2) else None
                        if faiss_ids is not None and len(faiss_ids) == len(rows):
                            order = list(map(int, faiss_ids))
                            by_id = {int(r["id"]): r for r in rows}
                            rows = [by_id[i] for i in order if i in by_id]
                            self.max_id = int(max(faiss_ids)) if len(faiss_ids) else -1
                            self.index = idx
                            need_rebuild = False
                        else:
                            logging.warning("[persistent-rag] id/count mismatch or not IDMap2; rebuild")
            except Exception as e:
                logging.warning(f"[persistent-rag] index read failed: {e}; rebuild")

            try:
                ids_preview = [int(r["id"]) for r in rows[:3]] + (["…"] if len(rows) > 6 else []) + [int(r["id"]) for r in rows[-3:]]
                logging.info("[persistent-rag] chunks=%d ids(head/tail)=%s", len(rows), ids_preview)
            except Exception:
                logging.info("[persistent-rag] chunks=%d", len(rows))

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

    def search_vec(self, qv: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        with self.lock:
            if self.index is None or self.index.ntotal == 0 or not self.chunks:
                return []
            qv = qv.reshape(1, -1).astype("float32")
            k = min(top_k, self.index.ntotal)
            D, I = self.index.search(qv, k)
            by_id = {int(r["id"]): r for r in self.chunks}
            return [by_id.get(int(i)) for i in I[0] if by_id.get(int(i))]

persistent_rag: Optional[PersistentRAG] = None

def _build_or_update_user_index(username: str):
    """(Re)build a FAISS IP index for a user's memory."""
    global user_memory_indexes
    dim = _embedding_dim()
    idx = faiss.IndexFlatIP(dim)

    chunks = user_memory_chunks.get(username, []) or []
    if not chunks:
        # drop dead index if no mem
        if username in user_memory_indexes:
            del user_memory_indexes[username]
        return

    # pull/calc vectors
    vecs = []
    for ch in chunks:
        # cache vec on the row if missing to avoid re-encode later
        if "vec" not in ch or ch["vec"] is None:
            ch["vec"] = _memvec_get(ch["content"])
        vecs.append(ch["vec"])

    if vecs:
        mat = np.vstack(vecs).astype("float32")
        idx.add(mat)

    user_memory_indexes[username] = idx

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
    """Load user memory from disk and (re)build per-user FAISS indexes."""
    global user_memory_chunks, user_memory_indexes
    try:
        if os.path.exists(USER_MEMORY_FILE):
            with open(USER_MEMORY_FILE, "r") as f:
                user_memory_chunks = json.load(f) or {}
            logging.info("[user-mem] Building indexes...")
            # normalize structure: list of {content, ts, vec?}
            for uname, rows in list(user_memory_chunks.items()):
                # some legacy files may store plain list[str]; normalize
                norm: List[Dict[str, Any]] = []
                for r in (rows or []):
                    if isinstance(r, str):
                        norm.append({"content": r, "ts": int(time.time()), "vec": _memvec_get(r)})
                    else:
                        # ensure keys exist
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
            logging.info("[user-mem] no existing file, starting fresh")
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
    """Append a memory row for a user (bounded), then rebuild index."""
    if not user or not text:
        return
    row = {"content": text.strip(), "ts": int(time.time())}
    # pre-cache vec to avoid re-encode on next build
    row["vec"] = _memvec_get(row["content"])

    with user_memory_lock:
        lst = user_memory_chunks.setdefault(user, [])
        lst.append(row)
        # TTL by age (optional) + cap by length
        if USER_MEMORY_TTL_DAYS > 0:
            cutoff = int(time.time()) - USER_MEMORY_TTL_DAYS * 86400
            lst = [r for r in lst if int(r.get("ts", 0)) >= cutoff]
        user_memory_chunks[user] = lst[-MAX_USER_MEMORY_CHUNKS:]
        _build_or_update_user_index(user)

def retrieve_user_memory(username: Optional[str], query: str, top_k: int = 3) -> List[str]:
    """FAISS-indexed user memory search (fast)."""
    if not username:
        return []
    try:
        index = user_memory_indexes.get(username)
        chunks = user_memory_chunks.get(username, []) or []
        if not index or index.ntotal == 0 or not chunks:
            return []

        qv = _encode([query])[0].reshape(1, -1).astype("float32")
        k = min(int(top_k), index.ntotal)
        distances, indices = index.search(qv, k)

        out: List[str] = []
        for i, dist in zip(indices[0], distances[0]):
            if i < 0:
                continue
            # cosine-ish via IP — tune threshold; 0.3 is a sane starter
            if float(dist) > 0.3:
                out.append(chunks[int(i)]["content"])
        return out
    except Exception as e:
        logging.error(f"[user-mem] retrieval error for {username}: {e}")
        return []

# ---------- FILE PARSE / CHUNK (from v7.1) ----------
SUPPORTED_TYPES = {'.txt', '.md', '.pdf', '.docx', '.csv'}

def _read_file_bytes(file_bytes: bytes, filename: str) -> str:
    name = (filename or '').lower()
    try:
        if name.endswith('.txt') or name.endswith('.md'):
            return file_bytes.decode('utf-8', errors='ignore')
        elif name.endswith('.pdf'):
            import fitz  # PyMuPDF
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

# ---------- EPHEMERAL RAG ----------
def _rebuild_index_from_chunks():
    """(B) Rebuild from cached vectors; if a row lacks vec, encode once and cache it."""
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
    """(B) GC and only rebuild index if the list size changed."""
    global knowledge_chunks
    now = int(time.time())
    before = len(knowledge_chunks)

    # TTL filter
    knowledge_chunks = [m for m in knowledge_chunks if (now - int(m.get("ts", 0))) <= ttl_seconds]

    # Per-session cap
    if session_id:
        sess = [m for m in knowledge_chunks if m.get("session_id") == session_id]
        if len(sess) > max_per_session:
            excess = len(sess) - max_per_session
            to_drop = set(id(m) for m in sorted(sess, key=lambda x: x.get("ts", 0))[:excess])
            knowledge_chunks = [m for m in knowledge_chunks if id(m) not in to_drop]

    # Global cap
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

_SLANG_EN_GREETINGS = re.compile(r"^(yo+|ye+|ya+|sup|wass?up|ayy+|hey+|hiya)\b", re.I)
_LANG_CODE_TO_NAME = {"en":"English","es":"Spanish","pt":"Portuguese","fr":"French","de":"German","it":"Italian",
                      "nl":"Dutch","sv":"Swedish","pl":"Polish","cs":"Czech","ru":"Russian","uk":"Ukrainian",
                      "tr":"Turkish","ar":"Arabic","he":"Hebrew","fa":"Persian","hi":"Hindi","bn":"Bengali",
                      "ur":"Urdu","ta":"Tamil","te":"Telugu","ml":"Malayalam","kn":"Kannada","th":"Thai",
                      "vi":"Vietnamese","id":"Indonesian","ms":"Malay","ja":"Japanese","ko":"Korean",
                      "zh-cn":"Chinese","zh-tw":"Chinese (Traditional)"}

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

def get_slang_lexicon() -> set:
    return set(get_cfg("variety.slang_lexicon.words", []) or [])

def sanitize_for_audience(text: str, audience: str) -> str:
    return text

_ALLOWED_GLYPHS = set("↺✶⛧🜃🜂∴∵∞ø☾⊙🜫☿⟁∆⧁∃")
_EMOJI_BLOCK = re.compile(r"[\U0001F300-\U0001FAFF]")

def _strip_emoji_except_glyphs(text: str) -> str:
    return _EMOJI_BLOCK.sub(lambda m: m.group(0) if m.group(0) in _ALLOWED_GLYPHS else "", text)

def enforce_persona_ending(text: str, mode: str) -> str:
    endings = get_cfg(f"range.modes.{mode}.endings", []) or []
    if not endings: return text
    return text if np.random.rand() < 0.5 else (text.rstrip() + "\n" + np.random.choice(endings))

HEDGE_REPLACEMENTS = [
    (r"\bAs an (?:AI|assistant)[^.\n]*\.\s*", ""),
    (r"\bI(?:\s+personally)?\s*think\b", "I think"),
    (r"\bI\s*believe\b", "I think"),
    (r"\bperhaps\b", "maybe"),
    (r"\bIt seems\b", "Looks like"),
    (r"\bWe (?:can try to|could)\b", "We do"),
    (r"\bYou might want to\b", "Do this:"),
    (r"\bConsider\b", "Do"),
    (r"\bshould\b", "will"),
]
SOFTENER_TRIMS = [(r"\s+\(let me know if that helps\)\.?$", ""), (r"\s+Thanks!\s*$", "")]

def _apply_pairs(text: str, pairs):
    for pat, rep in pairs:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    return text

def punch_up_text(s: str, mode: str, lang: str) -> str:
    s = _apply_pairs(_apply_pairs(s, HEDGE_REPLACEMENTS), SOFTENER_TRIMS)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"([!?])\1{1,}", r"\1", s)
    lex = get_slang_lexicon()
    if (lang or "English").lower() == "english":
        if mode == "astrofuck":
            low = s.lower()
            if lex and not any(tok in low for tok in lex):
                s = f"Dope — let’s slice it clean. {s}"
        elif mode == "dream" and DREAM_NO_SLANG:
            for w in lex:
                s = re.sub(rf"(?i)\b{re.escape(w)}\b", "", s)
    return s.strip()

# --- HARD TONE MASKS ---
_ASTROFUCK_BLEED_PHRASES = [
    r"\b(let'?s\s+breathe|gentle\s+check-in|soft(ly)?|whisper|tender|pour\s+tea|exhale)\b",
    r"[✨🌙⭐️]\s*",
    r"\b(it'?s\s+ok(?:ay)?\s+to\s+feel|holding\s+you\s+in\s+this)\b",
]
_DREAM_BLEED_PHRASES = [
    r"\b(yo+|ye+|wass?up|dope|deadass|nah\s*man|giggle-?slap|bullshit)\b",
    r"\b(slice it clean|vibe check)\b",
]

def _hard_tone_mask(text: str, mode: str) -> str:
    if not text:
        return text
    if mode == "astrofuck":
        for pat in _ASTROFUCK_BLEED_PHRASES:
            text = re.sub(pat, "", text, flags=re.IGNORECASE)
        lines = [l for l in text.splitlines() if l.strip()]
        if lines and not re.search(r"\b(Dope|Cut|Direct|Clean)\b", lines[0], re.IGNORECASE):
            lines[0] = "Dope — let’s slice it clean. " + lines[0].lstrip()
        return "\n".join(lines).strip()
    if mode == "dream" and DREAM_NO_SLANG:
        for pat in _DREAM_BLEED_PHRASES:
            text = re.sub(pat, "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\s*Dope\s+—\s+let[’']s\s+slice\s+it\s+clean\.\s*", "", text, flags=re.IGNORECASE)
        return text.strip()
    return text

# ---------- PROMPT ASSEMBLY ----------
def _cap(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars: return s
    return s[:max_chars].rsplit("\n", 1)[0].strip() or s[:max_chars].strip()

def build_messages(mode: str, system_seed: str, history: List[Dict[str, str]], user_text: str, context: str, lang_name: str):
    sys = system_seed
    if lang_name and lang_name not in ("English", "Unknown"):
        sys = f"CRITICAL: The user is writing in {lang_name}. Respond ONLY in {lang_name}.\n\n" + sys
    if context:
        sys += "\n\n--- Context ---\n" + context
    msgs = [{"role":"system","content":sys}]
    for turn in history:
        u = (turn.get("user") or "").strip(); a = (turn.get("assistant") or "").strip()
        if u: msgs.append({"role":"user","content":u})
        if a: msgs.append({"role":"assistant","content":a})
    msgs.append({"role":"user","content":user_text})
    return msgs

# ---------- ACHEFLIP ----------
def maybe_inject_acheflip(text_in: str, mode: str) -> str:
    if not get_cfg("philosophy.acheflip.enabled", True): return text_in
    kws = set(map(str.lower, get_cfg("philosophy.acheflip.distress_keywords", []) or []))
    distress = any(k in (text_in.lower()) for k in kws)
    if not distress: return text_in
    if np.random.rand() > float(get_cfg("philosophy.acheflip.nudge_probability", 0.7) or 0.7):
        return text_in
    nudges = get_cfg("philosophy.acheflip.nudge_templates", []) or []
    if not nudges: return text_in
    nudge = np.random.choice(nudges)
    if mode == "astrofuck":
        nudge = re.sub(r"\s+$", "", nudge) + " Ye, tiny step now."
    return (text_in.rstrip() + "\n\n" + nudge).strip()

# ---------- RETRIEVAL BLEND ----------
def retrieve_context_blend(query: str, session_id: Optional[str], thread_id: Optional[str], username: Optional[str], top_k: int = 8) -> str:
    parts: List[str] = []
    qv = _query_vec_cached(query, session_id, thread_id)

    # Persistent
    persistent_parts = []
    if persistent_rag is not None:
        try:
            rows = persistent_rag.search_vec(qv, top_k=min(5, top_k))
            for r in rows:
                c = (r.get("content", "") or "").strip()
                if c:
                    persistent_parts.append(c)
        except Exception as e:
            logging.error(f"Persistent RAG search error: {e}")
    if persistent_parts:
        parts.append("--- Persistent Knowledge ---\n" + "\n\n".join(persistent_parts))

    # Ephemeral
    if rag_index is not None and knowledge_chunks and rag_index.ntotal > 0:
        try:
            k = min(max(2, top_k - len(persistent_parts)), rag_index.ntotal)
            D, I = rag_index.search(qv.reshape(1, -1).astype("float32"), k)
            eph = []
            for i in I[0]:
                if 0 <= i < len(knowledge_chunks):
                    meta = knowledge_chunks[i]
                    if session_id and meta.get("session_id") != session_id: continue
                    if thread_id and meta.get("thread_id") != thread_id: continue
                    cc = meta.get("content", "")
                    if cc: eph.append(cc)
            if eph:
                parts.append("--- Ephemeral Context ---\n" + "\n\n".join(eph))
        except Exception as e:
            logging.error(f"Ephemeral RAG retrieval error: {e}")

    # Personal memory (FAISS-indexed)
    try:
        if username:
            um = retrieve_user_memory(username, query, top_k=3)
            if um:
                parts.append("--- Personal Memory Fragments ---\n" + "\n\n".join(um))
    except Exception as e:
        logging.error(f"[user-mem] retrieval error: {e}")

    return "\n\n".join(parts).strip()

# ---------- SAFE LOAD LLM ----------
def _try_load_llama(model_path: str, desired_ctx: int, n_gpu_layers: int) -> Optional[Llama]:
    base_kwargs = dict(
        model_path=model_path,
        n_ctx=int(desired_ctx),
        n_gpu_layers=int(n_gpu_layers),
        embedding=False,
        logits_all=False,
        verbose=False,
        use_mmap=True,
        use_mlock=True,
        session_file=os.path.join(STATE_DIR, f"{os.path.basename(model_path)}.kv"),
    )
    probe = Llama(**base_kwargs)
    addable = {}
    for k, v in [("n_threads", max(2, os.cpu_count() or 8)),
                 ("n_batch", int(get_cfg("engines.llama_cpp.n_batch", 1024) or 1024))]:
        try:
            test = dict(base_kwargs); test.update(addable); test.update({k: v})
            _ = Llama(**test)
            addable[k] = v
        except Exception:
            pass
    return Llama(**(dict(base_kwargs) | addable))


def _load_llama_with_backoff(model_path: str, requested_ctx: int, n_gpu_layers: int) -> Llama:
    ladder = [requested_ctx, 8192, 4096, 2048, 1024, 512]
    tried = []
    for ctx in ladder:
        if ctx in tried:
            continue
        tried.append(ctx)
        llm = _try_load_llama(model_path, ctx, n_gpu_layers)
        if llm:
            logging.info(f"[models] Loaded with n_ctx={ctx} (requested={requested_ctx})")
            return llm
    raise RuntimeError(f"Failed to load model at {model_path}; tried ctx {tried}")


# ---------- MODELS ----------
def load_llm_from_config(model_key: str) -> bool:
    if model_key in llm_vessels:
        return True
    model_config = (manifest.get("models", {}) or {}).get(model_key, {})
    if not isinstance(model_config, dict) or model_config.get("engine") != "llama_cpp":
        logging.error("Model config invalid or engine != llama_cpp for key %s", model_key)
        return False
    model_path = resolve_path(model_config.get("path", ""), homes)
    if not model_path or not os.path.exists(model_path):
        logging.error("Model not found: %s", model_path)
        return False
    try:
        requested_n_ctx = int(get_cfg("engines.llama_cpp.python_n_ctx", 4096) or 4096)
        n_gpu_layers = int(get_cfg("engines.llama_cpp.n_gpu_layers", 0) or 0)
        llm = _load_llama_with_backoff(model_path, requested_n_ctx, n_gpu_layers)
        llm_vessels[model_key] = llm
        vessel_details[model_key] = {"family": (model_config.get("family") or ""), "n_ctx": requested_n_ctx}
        logging.info("Vessel '%s' online.", model_key)
        return True
    except Exception as e:
        logging.error("Failed to load '%s': %s", model_key, e)
        return False


# ---------- API MODELS ----------
class RequestModel(BaseModel):
    prompt: str
    mode: Optional[str] = None
    turn: Optional[int] = 0
    session_id: Optional[str] = None
    new_session: bool = False
    force_lang: Optional[str] = None
    username: Optional[str] = None
    thread_id: Optional[str] = None   # thread scoping


class MemoryUpsert(BaseModel):
    user: str
    text: str


# ---- thread helpers ----
def _thread_id_of(req) -> Optional[str]:
    tid = getattr(req, "thread_id", None)
    if not tid:
        return None
    s = str(tid).strip()
    return s or None


def _thread_history(session: Dict[str, Any], thread_id: Optional[str], limit: int) -> List[Dict[str, str]]:
    key = thread_id or "__default__"
    by_thread = session.setdefault("history_by_thread", {})
    hist = by_thread.get(key, [])
    return hist[-int(limit):]


def _append_history(session: Dict[str, Any], thread_id: Optional[str],
                    user_text: str, assistant_text: str, mode: str, keep_cap: int) -> None:
    key = thread_id or "__default__"
    by_thread = session.setdefault("history_by_thread", {})
    lst = by_thread.get(key, [])
    lst.append({"user": user_text, "assistant": assistant_text, "mode": mode})
    by_thread[key] = lst[-int(keep_cap):]
    session["history_by_thread"] = by_thread  # write back


# ---------- mode-scoped history filter ----------
def filter_history_by_mode(history: List[Dict[str, str]], mode: str, max_turns: int) -> List[Dict[str, str]]:
    same = [t for t in history if (t.get("mode") or "").lower() == mode.lower()]
    return same[-max_turns:]


# ---------- STREAM GEN (SSE) ----------
async def generate_response_stream(session: Dict, request: RequestModel):
    global TTFT_LAST
    user_text = (request.prompt or "").strip()
    req_mode = (request.mode or "astrofuck").strip().lower()
    t_start = time.time()

    available_modes = set((get_cfg("range.modes", {}) or {}).keys())
    mode = req_mode if req_mode in available_modes else "astrofuck"

    lang = request.force_lang or language_hysteresis_update_lang(session, user_text)

    chat_key = get_cfg("chat.vessel_key", "qwen3_4b_unified")
    chat_llm = llm_vessels.get(chat_key)
    if not chat_llm:
        yield "event: error\ndata: Model not loaded.\n\n"
        return

    # ----- SYSTEM PROMPT ASSEMBLY -----
    system_prompt_parts: List[str] = []
    system_core = get_cfg("prompts.system_core", "")
    if system_core:
        system_prompt_parts.append(system_core)

    style_contract = get_cfg(
        "prompts.style_contract_astrofuck" if mode == "astrofuck" else "prompts.style_contract_dream",
        ""
    )
    if style_contract:
        system_prompt_parts.append(style_contract)

    mode_inject = get_cfg(f"prompts.mode_tone_inject.{mode}", "")
    if mode_inject:
        system_prompt_parts.append(mode_inject)

    # tiny watermark to reduce drift
    system_prompt_parts.append(
        f"[MODE:{mode.upper()}] Respond strictly in the {mode.upper()} register. "
        f"Do not mirror stylistic elements from other modes."
    )

    final_system_prompt = "\n\n".join(system_prompt_parts)

    # keep ephemeral small per session/thread every turn
    try:
        eph_cfg = get_cfg("rag.ephemeral", {}) or {}
        _evict_and_ttl_gc(
            max_total=int(eph_cfg.get("max_total", 5000) or 5000),
            max_per_session=int(eph_cfg.get("max_per_session", 1500) or 1500),
            ttl_seconds=int(eph_cfg.get("ttl_seconds", 1800) or 1800),
            session_id=session.get("id"),
        )
    except Exception:
        pass

    thread_id = _thread_id_of(request)
    context = retrieve_context_blend(
        user_text,
        session.get("id"),
        thread_id,
        session.get("username") or request.username,
        top_k=8,
    )

    audience = (session.get("audience") or get_cfg("release.audience_default", "internal")).lower()

    final_system_prompt = _cap(final_system_prompt, 3500)
    context = _cap(context, 4000)

    # history window (mode-scoped within this thread)
    history_pairs_all = _thread_history(
        session,
        thread_id,
        int(get_cfg("assembly.history_turns", DEFAULT_HISTORY_TURNS_FOR_PROMPT))
    )
    history_pairs = filter_history_by_mode(
        history_pairs_all,
        mode,
        int(get_cfg("assembly.history_turns", DEFAULT_HISTORY_TURNS_FOR_PROMPT))
    )

    msgs = build_messages(mode, final_system_prompt, history_pairs, user_text, context, lang)

    # KV prefill per (mode,lang)
    kv_map: Dict[str, bool] = session.setdefault("kv_ready_map", {})
    ctx_hash = hashlib.sha1((context or "").encode("utf-8","ignore")).hexdigest()[:8]
    kv_key = f"{mode}::{lang}::{ctx_hash}"
    if not kv_map.get(kv_key, False):
        try:
            if hasattr(chat_llm, "reset"):
                chat_llm.reset()
        except Exception:
            pass
        try:
            stable_hist = history_pairs[-2:]
            prefill_msgs = build_messages(mode, final_system_prompt, stable_hist, "", context, lang)
            prefill_params = _prune_unsupported_params(chat_llm, {
                "messages": prefill_msgs,
                "max_tokens": 0,
                "cache_prompt": True,
                "stream": False,
            })
            _ = chat_llm.create_chat_completion(**prefill_params)
            kv_map[kv_key] = True
            session["kv_ready_map"] = kv_map
            logging.info("[kv] Prefilled system+stable-history for %s", kv_key)
        except Exception as e:
            logging.warning(f"[kv] Prefill failed (non-fatal) for {kv_key}: {e}")

    # model params (mode scoped)
    mp = get_cfg(f"range.modes.{mode}.params", {}) or {}
    mp.setdefault("temperature", float(get_cfg("params.temperature", 0.8) or 0.8))
    mp.setdefault("top_p", float(get_cfg("params.top_p", 0.95) or 0.95))
    mp.setdefault("repeat_penalty", float(get_cfg("params.repeat_penalty", 1.1) or 1.1))
    mp.setdefault("n_predict", int(get_cfg("params.n_predict", 512) or 512))

    raw_params = dict(
        messages=msgs,
        temperature=float(mp.get("temperature")),
        top_p=float(mp.get("top_p")),
        repeat_penalty=float(mp.get("repeat_penalty")),
        max_tokens=int(mp.get("n_predict")),
        stop=["</s>", "<|im_end|>", "[INST]", "[/INST]"],
        stream=True,
        cache_prompt=True,
    )
    call_params = _prune_unsupported_params(chat_llm, raw_params)

    response_text = ""
    stream_emitted_any = False
    last_error_msg = None
    first_piece_done = False

    def _sse(data: str) -> str:
        data = (data or "").replace("\r\n", "\n")
        return "data: " + data.replace("\n", "\ndata: ") + "\n\n"

    async def _run_stream(call_params: dict):
        nonlocal response_text, stream_emitted_any, last_error_msg, first_piece_done, t_start, audience
        try:
            yield ": ready\n\n"
            last_flush = time.time()
            for chunk in chat_llm.create_chat_completion(**call_params):
                choices = (chunk.get("choices") or [])
                delta = (choices[0].get("delta") if choices else {}) or {}
                piece = delta.get("content")
                if not piece:
                    now = time.time()
                    if now - last_flush > 1.0:
                        yield ": keep-alive\n\n"
                        last_flush = now
                    continue

                if not first_piece_done:
                    TTFT_LAST = time.time() - t_start
                    logging.info(f"[ttft] {TTFT_LAST:.3f}s")
                    first_piece_done = True

                safe_piece = sanitize_for_audience(piece, audience)
                response_text += safe_piece
                stream_emitted_any = True
                yield _sse(safe_piece)
                await asyncio.sleep(0)
        except Exception as e:
            logging.error("Generation error: %s", e, exc_info=True)
            last_error_msg = f"[core-error] {type(e).__name__}: {e}"
            yield f"event: error\ndata: {last_error_msg}\n\n"

    # Attempt stream
    async for sse_chunk in _run_stream(call_params):
        yield sse_chunk

    # Retry sans cache_prompt
    if not stream_emitted_any and "cache_prompt" in call_params:
        call_params = {k: v for k, v in call_params.items() if k != "cache_prompt"}
        async for sse_chunk in _run_stream(call_params):
            yield sse_chunk

    # Minimal
    if not stream_emitted_any:
        minimal = _prune_unsupported_params(chat_llm, {"messages": msgs, "stream": True})
        async for sse_chunk in _run_stream(minimal):
            yield sse_chunk

    if not stream_emitted_any and not response_text.strip():
        msg = last_error_msg or "[core-error] No content generated. Check model load/params."
        yield f"event: error\ndata: {msg}\n\n"
        return

    # finalize text (mode post)
    final_text = (response_text.strip() or "…")
    if mode == "dream" and DREAM_NO_SLANG:
        for w in get_slang_lexicon():
            final_text = re.sub(rf"(?i)\b{re.escape(w)}\b", "", final_text)
    if mode == "astrofuck":
        lex = get_slang_lexicon()
        if lex and not any(tok in final_text.lower() for tok in lex):
            final_text = "Dope — let’s slice it clean. " + final_text

    final_text = punch_up_text(final_text, mode, lang)

    # --- hard tone mask (bleed scrub) ---
    # (place earlier in file near punch_up_text if you prefer; safe to call here)
    # NOTE: assumes _ASTROFUCK_BLEED_PHRASES / _DREAM_BLEED_PHRASES / _hard_tone_mask already defined above.
    final_text = _hard_tone_mask(final_text, mode)

    final_text = maybe_inject_acheflip(final_text, mode)
    final_text = _strip_emoji_except_glyphs(final_text)
    final_text = enforce_persona_ending(final_text, mode)

    keep_cap = int(get_cfg("assembly.keep_history", DEFAULT_KEEP_HISTORY))
    _append_history(session, thread_id, user_text, final_text, mode, keep_cap)
    session["turn"] = int(session.get("turn", 0)) + 1
    try:
        save_session(session.get("id"))
        save_sessions()
    except Exception:
        logging.error("autosave sessions failed", exc_info=True)

    yield ": done\n\n"

# ---------- INGEST (A) — re-enabled multipart endpoint ----------
if _HAS_MULTIPART:
    @app.post("/rag/ingest")
    async def rag_ingest(
        files: List[UploadFile] = File(...),
        source: Optional[str] = Form(None),
        thread_id: Optional[str] = Form(None),
        http: FastAPIRequest = None
    ):
        try:
            _ensure_embedder()
            global rag_index
            if rag_index is None:
                rag_index = faiss.IndexFlatIP(_embedding_dim())

            sid = (http.cookies.get("blur_sid") if http else None) or last_seen_session_id
            total_added, results = 0, []

            eph_cfg = get_cfg("rag.ephemeral", {}) or {}
            max_total = int(eph_cfg.get("max_total", 5000) or 5000)
            max_per_session = int(eph_cfg.get("max_per_session", 1500) or 1500)
            ttl_seconds = int(eph_cfg.get("ttl_seconds", 1800) or 1800)

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
                    now = int(time.time())
                    rows = [{
                        "source": (source or name),
                        "content": c,
                        "ts": now,
                        "session_id": sid,
                        "thread_id": thread_id,
                        "filename": name,
                        "vec": v  # (B) cache vector at ingest
                    } for c, v in zip(chunks, vecs)]

                    with rag_lock:
                        knowledge_chunks.extend(rows)
                        rag_index.add(np.vstack([r["vec"] for r in rows]).astype("float32"))
                        _evict_and_ttl_gc(max_total=max_total, max_per_session=max_per_session, ttl_seconds=ttl_seconds, session_id=sid)

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
        return JSONResponse({"ok": False, "error": "install python-multipart"}, status_code=503)

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
def persistent_search(q: str, k: int = 8):
    try:
        if not persistent_rag: return {"ok": True, "results": []}
        rows = persistent_rag.search(q, top_k=int(k))
        return {"ok": True, "results": rows}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    
# ---------- USER MEMORY ROUTES ----------
@app.post("/memory/upsert")
def memory_upsert(m: MemoryUpsert):
    try:
        upsert_user_memory(m.user, m.text)
        save_user_memory()
        return {"ok": True, "count": len(user_memory_chunks.get(m.user, []))}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# ---------- ASR (optional) ----------
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

# ---------- HEALTH / SESSION ----------
@app.get("/healthz")
def healthz():
    eph_cfg = get_cfg("rag.ephemeral", {}) or {}
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
        },
        "rag_ephemeral": {
            "max_total": int(eph_cfg.get("max_total", 5000) or 5000),
            "max_per_session": int(eph_cfg.get("max_per_session", 1500) or 1500),
            "ttl_seconds": int(eph_cfg.get("ttl_seconds", 1800) or 1800),
        },
        "multipart": _HAS_MULTIPART,
    }

@app.get("/session")
async def get_new_session():
    global last_seen_session_id
    sid = str(uuid.uuid4())
    sessions[sid] = {
        "id": sid,
        "mode": "astrofuck",  # default
        "history": [],
        "turn": 0,
        "username": None,
        "wants": {"tone": "astrofuck", "defaultMode": "astrofuck", "ragAutoIngest": True},
        "audience": "internal",
        "kv_ready": False,
        "kv_ready_map": {},          # per-thread kv readiness
        "history_by_thread": {},     # per-thread history
        "active_lang": get_cfg("language.default_if_unknown", "English"),
    }
    last_seen_session_id = sid
    resp = JSONResponse({"session_id": sid})
    resp.headers["X-Session-ID"] = sid
    resp.set_cookie("blur_sid", sid, path="/", httponly=True, samesite="lax")
    return resp

@app.post("/generate_response")
async def handle_generate_request(req: RequestModel, http: FastAPIRequest):
    global last_seen_session_id, TTFT_LAST
    sid = (req.session_id or http.headers.get("X-Session-ID") or http.cookies.get("blur_sid") or last_seen_session_id or str(uuid.uuid4())).strip()
    if req.new_session:
        sid = str(uuid.uuid4())

    if sid not in sessions:
        sessions[sid] = {
            "id": sid,
            "mode": "astrofuck",
            "history": [],
            "turn": 0,
            "username": req.username,
            "wants": {"tone": "astrofuck", "defaultMode": "astrofuck", "ragAutoIngest": True},
            "audience": "internal",
            "kv_ready": False,
            "kv_ready_map": {},
            "history_by_thread": {},
            "active_lang": get_cfg("language.default_if_unknown", "English"),
        }

    if req.mode:
        sessions[sid]["mode"] = (req.mode or "astrofuck").lower()

    last_seen_session_id = sid

    resp = StreamingResponse(
        generate_response_stream(sessions[sid], req),
        media_type="text/event-stream; charset=utf-8",
    )
    resp.headers["Cache-Control"] = "no-cache, no-transform"
    resp.headers["X-Accel-Buffering"] = "no"
    resp.headers["Connection"] = "keep-alive"
    resp.headers["X-Session-ID"] = sid
    if TTFT_LAST is not None:
        resp.headers["X-TTFT-Last"] = f"{TTFT_LAST:.3f}"
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

@app.get("/debug/paths")
def debug_paths():
    chat_key = get_cfg("chat.vessel_key", "qwen3_4b_unified")
    return {
        "BLUR_HOME": BLUR_HOME,
        "MANIFEST_PATH": MANIFEST_PATH,
        "STATE_DIR": STATE_DIR,
        "SESSIONS_DIR": SESSIONS_DIR,
        "LAST_SESSION_FILE": LAST_SESSION_FILE,
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

# ---------- SESSIONS I/O ----------
def _ensure_state_dir():
    try:
        Path(STATE_DIR).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to create state dir {STATE_DIR}: {e}")

def _session_path(sid: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{sid}.json")

def load_session(sid: str) -> Dict[str, Any]:
    """Lazy-load a single session file; clamp history in-memory."""
    path = _session_path(sid)
    data: Dict[str, Any] = {}
    keep_n = int(get_cfg("assembly.keep_history", DEFAULT_KEEP_HISTORY))
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f) or {}
        except Exception as e:
            logging.error("Failed to read session %s: %s", sid, e)
            data = {}
    data.setdefault("id", sid)
    hist = data.get("history", [])
    data["history"] = hist[-keep_n:] if isinstance(hist, list) else []
    data.setdefault("mode", "astrofuck")
    data.setdefault("turn", len(data["history"]))
    data.setdefault("username", data.get("username"))
    data.setdefault("wants", {"tone": "astrofuck", "defaultMode": "astrofuck", "ragAutoIngest": True})
    data.setdefault("audience", "internal")
    data.setdefault("kv_ready", False)
    data.setdefault("kv_ready_map", {})       # thread_id -> bool
    data.setdefault("history_by_thread", {})  # thread_id -> [{user,assistant}]
    data.setdefault("active_lang", get_cfg("language.default_if_unknown","English"))
    sessions[sid] = data
    return data

def save_session(sid: str):
    """Persist only this session; no global write."""
    if sid not in sessions: return
    path = _session_path(sid)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(sessions[sid], f, indent=2)
        os.replace(tmp, path)
        # remember last active
        try:
            with open(LAST_SESSION_FILE, "w") as g:
                g.write(sid)
        except Exception:
            pass
    except Exception as e:
        logging.error("Failed to save session %s: %s", sid, e)

def load_sessions():
    # noop: try to restore only last seen for warmth
    global last_seen_session_id
    try:
        if os.path.exists(LAST_SESSION_FILE):
            with open(LAST_SESSION_FILE, "r") as f:
                sid = (f.read() or "").strip()
                if sid:
                    load_session(sid)
                    last_seen_session_id = sid
    except Exception:
        pass

def save_sessions():
    # noop: we persist per-session; nothing to do globally
    pass


# ---------- STARTUP / SHUTDOWN ----------
@app.on_event("startup")
async def startup_event():
    global manifest, homes, rag_index, persistent_rag
    logging.info(f"Startup: loading manifest: {MANIFEST_PATH}")
    try:
        _ensure_state_dir()

        if not os.path.exists(MANIFEST_PATH):
            logging.error(f"🛑 Manifest file not found: {MANIFEST_PATH}")
            raise FileNotFoundError(MANIFEST_PATH)

        with open(MANIFEST_PATH, 'r') as f:
            manifest = yaml.safe_load(f) or {}

        # ensure chat vessel in manifest (qwen-only)
        manifest.setdefault("chat", {}).setdefault("vessel_key", "qwen3_4b_unified")

        # resolve homes with safe BLUR_HOME fallbacks (v7.1 behavior restored)
        homes_local = (manifest.get('meta', {}) or {}).get('homes', {}) or {}
        homes_local.setdefault("blur_home", BLUR_HOME)
        homes_local.setdefault("models", os.path.join(homes_local["blur_home"], "models"))
        homes_local.setdefault("pipes",  os.path.join(homes_local["blur_home"], "run", "pipes"))
        homes_local.setdefault("data",   os.path.join(homes_local["blur_home"], "core"))
        homes.clear(); homes.update(resolve_homes_recursive(homes_local))

        # Load ONLY the chat vessel
        chat_key = get_cfg("chat.vessel_key", "qwen3_4b_unified")
        async with _VESSEL_LOCK:
            if not load_llm_from_config(chat_key):
                logging.error(f"🛑 Chat vessel '{chat_key}' failed to load.")
            else:
                logging.info(f"✅ Chat vessel ready: {chat_key}")

        # embedder + ephemeral idx
        async with _embed_lock:
            _ensure_embedder()
        rag_index = faiss.IndexFlatIP(_embedding_dim())

        # Persistent RAG init — use manifest if provided, otherwise BLUR_HOME fallbacks (v7.1 style)
        idx_path = resolve_path(get_cfg("memory.vector_store.path", ""), homes)
        ch_path  = resolve_path(get_cfg("memory.vector_store.chunks_path", ""), homes)
        if not idx_path:
            idx_path = os.path.join(BLUR_HOME, "core", "ouinet", "blurchive", "ecosystem", "blur_knowledge.index")
        if not ch_path:
            ch_path  = os.path.join(BLUR_HOME, "core", "ouinet", "blurchive", "ecosystem", "knowledge_chunks.jsonl")
        # ensure embed model key has a sane default
        manifest.setdefault("memory", {}).setdefault("vector_store", {}).setdefault("embed_model", "snowflake_arctic_embed")
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
        logging.info("Core ready (v9.2).")
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
