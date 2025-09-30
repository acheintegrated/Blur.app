#!/usr/bin/env python3
# convo_chat_core.py — Reforged v9.1 (Manifest-True, Qwen-only, Strict RAG)
# - Manifest is the single source of truth (no path guessing). ${meta.homes.*}, ${BLUR_HOME} resolved.
# - Chat vessel: chat.vessel_key (qwen3_4b_unified). No other models loaded.
# - Persistent RAG: uses memory.vector_store.{path,chunks_path,embed_model,ttl_days_persistent,auto_compact_on_start}
#   * Rebuilds index if: missing, id/count mismatch, or dim != embedder dim
#   * Enforces JSONL ids stable [0..N-1] (auto-fix missing ids)
# - Ephemeral RAG: uses rag.ephemeral.{max_total,max_per_session,ttl_seconds}
# - Assembly sizes: assembly.history_turns / keep_history
# - Language hysteresis: language.hysteresis_consecutive
# - Mode params/endings from range.modes.{dream,astrofuck,sentinel}
# - Acheflip hook stubbed from philosophy.acheflip.* (no external runner required)
# - Streaming: plain text chunks; exposes X-Session-ID + X-TTFT-Last

import sys, os, logging, asyncio, yaml, faiss, json, uuid, re, time, threading, io, tempfile, hashlib, inspect, base64
from typing import Optional, Dict, List, Any
from pathlib import Path
import numpy as np

# Quiet spam
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
SESSIONS_FILE = os.path.join(STATE_DIR, "sessions.json")
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
    # recursive expansion for ${meta.homes.*}, ${BLUR_HOME}, $ENV, ~
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
    probe = _embed_llm.create_embedding("dim?")['data'][0]['embedding']  # type: ignore
    _embed_dim = len(probe)
    return _embed_dim

def _encode(texts: List[str]) -> np.ndarray:
    _ensure_embedder()
    out = _embed_llm.create_embedding(texts)['data']  # type: ignore
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

persistent_rag: Optional[PersistentRAG] = None

# ---------- USER MEMORY ----------
user_memory_chunks: Dict[str, List[Dict[str, Any]]] = {}
user_memory_lock = threading.Lock()
MAX_USER_MEMORY_CHUNKS = 50
USER_MEMORY_TTL_DAYS = 90

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
            cv = _memvec_get(ch["content"])
            scored.append((float(np.dot(qv, cv)), ch["content"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for s, c in scored[:top_k] if s > 0.3]
    except Exception as e:
        logging.error(f"User memory retrieval error: {e}")
        return []

# ---------- EPHEMERAL RAG ----------
def _rebuild_index_from_chunks():
    global rag_index
    dim = _embedding_dim()
    idx = faiss.IndexFlatIP(dim)
    if knowledge_chunks:
        vecs = _encode([m["content"] for m in knowledge_chunks])
        idx.add(vecs)
    rag_index = idx

def _evict_and_ttl_gc(max_total: int, max_per_session: int, ttl_seconds: int, session_id: Optional[str] = None):
    global knowledge_chunks
    now = int(time.time())
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
    # load from manifest at call-time (manifest loads after import)
    return set(get_cfg("variety.slang_lexicon.words", []) or [])

def sanitize_for_audience(text: str, audience: str) -> str:
    # moderation off in manifest; passthrough
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

# ---------- PROMPT ASSEMBLY ----------
def _cap(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars: return s
    return s[:max_chars].rsplit("\n", 1)[0].strip() or s[:max_chars].strip()

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

# ---------- ACHEFLIP (stubbed based on YAML) ----------
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

    persistent_parts = []
    if persistent_rag is not None:
        try:
            rows = persistent_rag.search(query, top_k=min(5, top_k))
            for r in rows:
                c = (r.get("content","") or "").strip()
                if c: persistent_parts.append(c)
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
                    cc = meta.get("content","")
                    if cc: eph.append(cc)
            if eph:
                parts.append("--- Ephemeral Context ---\n" + "\n\n".join(eph))
        except Exception as e:
            logging.error(f"Ephemeral RAG retrieval error: {e}")

    um = retrieve_user_memory(username, query, top_k=3)
    if um:
        parts.append("--- Personal Memory Fragments ---\n" + "\n\n".join(um))

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
        if ctx in tried: continue
        tried.append(ctx)
        llm = _try_load_llama(model_path, ctx, n_gpu_layers)
        if llm:
            logging.info(f"[models] Loaded with n_ctx={ctx} (requested={requested_ctx})")
            return llm
    raise RuntimeError(f"Failed to load model at {model_path}; tried ctx {tried}")

# ---------- MODELS ----------
def load_llm_from_config(model_key: str) -> bool:
    if model_key in llm_vessels: return True
    model_config = (manifest.get('models', {}) or {}).get(model_key, {})
    if not isinstance(model_config, dict) or model_config.get('engine') != 'llama_cpp':
        logging.error("Model config invalid or engine != llama_cpp for key %s", model_key); return False
    model_path = resolve_path(model_config.get('path', ''), homes)
    if not model_path or not os.path.exists(model_path):
        logging.error("Model not found: %s", model_path); return False
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

class MemoryUpsert(BaseModel):
    user: str
    text: str

# ---------- STREAM GEN ----------
async def generate_response_stream(session: Dict, request: RequestModel):
    global TTFT_LAST
    user_text = (request.prompt or "").strip()
    req_mode = (request.mode or "astrofuck").strip().lower()
    t_start = time.time()

    # mode gate → default astrofuck, no policy fallback
    available_modes = set((get_cfg("range.modes", {}) or {}).keys())
    mode = req_mode if req_mode in available_modes else "astrofuck"

    # language hysteresis
    lang = request.force_lang or language_hysteresis_update_lang(session, user_text)

    # chat vessel (qwen3_4b_unified)
    chat_key = get_cfg("chat.vessel_key", "qwen3_4b_unified")
    chat_llm = llm_vessels.get(chat_key)
    if not chat_llm:
        yield "Model not loaded."
        return

    # ----- SYSTEM PROMPT ASSEMBLY (mode-scoped contracts + inject) -----
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

    final_system_prompt = "\n\n".join(system_prompt_parts)

    # retrieval blend (top_k = 8)
    context = retrieve_context_blend(
        user_text,
        session.get("id"),
        None,
        session.get("username") or request.username,
        top_k=8,
    )
    audience = (session.get("audience") or get_cfg("release.audience_default", "internal")).lower()

    # caps to keep TTFT sharp
    final_system_prompt = _cap(final_system_prompt, 3500)
    context = _cap(context, 4000)

    # history window
    history_pairs = session.get("history", [])[-int(get_cfg("assembly.history_turns", DEFAULT_HISTORY_TURNS_FOR_PROMPT)) :]

    msgs = build_messages(mode, final_system_prompt, history_pairs, user_text, context, lang)

    # one-time KV prefill
    if not session.get("kv_ready"):
        try:
            prefill_params = _prune_unsupported_params(chat_llm, {
                "messages": build_messages(mode, final_system_prompt, history_pairs, "", context, lang),
                "max_tokens": 0, "cache_prompt": True, "stream": False,
            })
            _ = chat_llm.create_chat_completion(**prefill_params)
            session["kv_ready"] = True
            logging.info("[kv] Prefilled system+history.")
        except Exception as e:
            logging.warning(f"[kv] Prefill failed (non-fatal): {e}")

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

    async def _run_stream(call_params: dict):
        nonlocal response_text, stream_emitted_any, last_error_msg, first_piece_done, t_start, audience
        try:
            for chunk in chat_llm.create_chat_completion(**call_params):
                choices = chunk.get("choices") or []
                if not choices: continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if not piece: continue
                if not first_piece_done:
                    TTFT_LAST = time.time() - t_start
                    logging.info(f"[ttft] {TTFT_LAST:.3f}s")
                    first_piece_done = True
                piece = piece.replace("\r\n", "\n")
                safe_piece = sanitize_for_audience(piece, audience)
                response_text += safe_piece
                stream_emitted_any = True
                yield safe_piece
                await asyncio.sleep(0)
        except Exception as e:
            logging.error("Generation error: %s", e, exc_info=True)
            last_error_msg = f"[core-error] {type(e).__name__}: {e}\n"
            yield "\n\n" + last_error_msg

    # Attempt stream
    async for chunk in _run_stream(call_params):
        yield chunk

    # Retry without cache_prompt if dry
    if not stream_emitted_any and "cache_prompt" in call_params:
        call_params = {k: v for k, v in call_params.items() if k != "cache_prompt"}
        async for chunk in _run_stream(call_params):
            yield chunk

    # Minimal
    if not stream_emitted_any:
        minimal = _prune_unsupported_params(chat_llm, {"messages": msgs, "stream": True})
        async for chunk in _run_stream(minimal):
            yield chunk

    if not stream_emitted_any and not response_text.strip():
        msg = last_error_msg or "[core-error] No content generated. Check model load/params.\n"
        yield "\n\n" + msg
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
    final_text = maybe_inject_acheflip(final_text, mode)
    final_text = _strip_emoji_except_glyphs(final_text)
    final_text = enforce_persona_ending(final_text, mode)

    # persist history within cap
    hist = session.setdefault("history", [])
    hist.append({"user": user_text, "assistant": final_text})
    session["history"] = hist[-int(get_cfg("assembly.keep_history", DEFAULT_KEEP_HISTORY)) :]
    session["turn"] = int(session.get("turn", 0)) + 1
    try:
        save_sessions()
    except Exception:
        logging.error("autosave sessions failed", exc_info=True)

# ---------- INGEST (ephemeral; disabled multipart here to keep deps minimal) ----------
@app.post("/rag/ingest")
async def rag_ingest_unavailable():
    return JSONResponse({"ok": False,"error": "File upload endpoint unavailable in this build. Enable python-multipart if needed."}, status_code=503)

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
        }
    }

@app.get("/session")
async def get_new_session():
    global last_seen_session_id
    sid = str(uuid.uuid4())
    sessions[sid] = {
        "id": sid,
        "mode": "astrofuck",  # 🔥 hard default
        "history": [],
        "turn": 0,
        "username": None,
        "wants": {"tone": "astrofuck", "defaultMode": "astrofuck", "ragAutoIngest": True},
        "audience": "internal",
        "kv_ready": False,
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
            "mode": "astrofuck",  # 🔥 cold-start astrofuck
            "history": [],
            "turn": 0,
            "username": req.username,
            "wants": {"tone": "astrofuck", "defaultMode": "astrofuck", "ragAutoIngest": True},
            "audience": "internal",
            "kv_ready": False,
            "active_lang": get_cfg("language.default_if_unknown", "English"),
        }

    # If request omitted mode, keep astrofuck
    sessions[sid]["mode"] = (req.mode or "astrofuck").lower()

    last_seen_session_id = sid
    resp = StreamingResponse(generate_response_stream(sessions[sid], req), media_type="text/plain")
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

def load_sessions():
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
                data.setdefault("wants", {"tone": "astrofuck", "defaultMode": "astrofuck", "ragAutoIngest": True})
                data.setdefault("audience", "internal")
                data.setdefault("turn", len(data["history"]))
                data.setdefault("kv_ready", False)
                data.setdefault("active_lang", get_cfg("language.default_if_unknown","English"))
                sessions[sid] = data
            logging.info("Loaded %d sessions from %s", len(sessions), SESSIONS_FILE)
        except Exception as e:
            logging.error("Failed to load sessions: %s", e)
            sessions = {}
    else:
        sessions = {}

def save_sessions():
    try:
        _ensure_state_dir()
        tmp = SESSIONS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(sessions, f, indent=2)
        os.replace(tmp, SESSIONS_FILE)
    except Exception as e:
        logging.error("Failed to save sessions: %s", e)

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

        # resolve homes strictly from manifest
        homes_local = (manifest.get('meta', {}) or {}).get('homes', {}) or {}
        homes_local.setdefault("blur_home", BLUR_HOME)
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

        # Persistent RAG init (strict manifest paths)
        idx_path = resolve_path(get_cfg("memory.vector_store.path", ""), homes)
        ch_path  = resolve_path(get_cfg("memory.vector_store.chunks_path", ""), homes)
        if not idx_path or not ch_path:
            raise RuntimeError("memory.vector_store.{path,chunks_path} must be set in manifest")
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
        logging.info("Core ready (v9.1).")
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
