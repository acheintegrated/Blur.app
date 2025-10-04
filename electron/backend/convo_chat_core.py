#!/usr/bin/env python3
# convo_chat_core.py — Reforged v9.26 (APN Integration)
# - PLASTICITY: Integrated an Auxiliary Plasticity Network (APN), a small RNN-like structure
#   that maintains a continuous, plastic state vector (S) per session.
# - IN-SITU CONTEXT: The APN's state is updated on every turn and injected into the system
#   prompt as a base64-encoded float vector, giving the LLM an "in-situ" trace of the
#   session's affective/cognitive trajectory.
# - ADAPTATION: The APN's weights are updated via a Hebbian learning rule modulated by
#   distress signals (`acheflip`), allowing the system's internal state dynamics to adapt over time.
# - Maintains all stability and performance features from v9.25.

from __future__ import annotations
import sys, os, logging, asyncio, yaml, faiss, json, uuid, re, time, threading, inspect, base64, hashlib
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
sessions: Dict[str, Dict[str, Any]] = {}
sessions_lock = threading.Lock()
last_seen_session_id: Optional[str] = None
_embed_lock = asyncio.Lock()
_embed_llm: Optional[Llama] = None
_embed_dim: Optional[int] = None
_MEMVEC: Dict[str, np.ndarray] = {}
TTFT_LAST: Optional[float] = None
user_memory_chunks: Dict[str, List[Dict[str, Any]]] = {}
user_memory_indexes: Dict[str, faiss.Index] = {}
user_memory_lock = threading.Lock()
MAX_USER_MEMORY_CHUNKS = 50
USER_MEMORY_TTL_DAYS = 90
app = FastAPI(default_response_class=ORJSONResponse)
if os.getenv("BLUR_PACKAGED") == "1":
    app.add_middleware(CORSMiddleware, allow_origin_regex=".*", allow_credentials=True, allow_methods=["*"], allow_headers=["*"], expose_headers=["X-Session-ID","X-TTFT-Last"])
else:
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:6969","http://127.0.0.1:6969"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"], expose_headers=["X-Session-ID","X-TTFT-Last"])

# ---------- APN (Auxiliary Plasticity Network) ----------
_APN_STATE_DIM = 64        # S size
_APN_HIDDEN_DIM = 48       # tiny middle
_APN_ECSD_DIM  = 12        # output delta size
_APN_LR        = 0.05      # η for state update
_APN_HEBB_LR   = 0.002     # α for Hebbian weight nudges
_APN_WCLIP     = 3.0       # weight clamp to avoid explosion
_APN_DECAY     = 0.0005    # L2 decay

def _apn_init_params(q_dim: int) -> dict:
    rng = np.random.default_rng(42)
    # Layer shapes: S·W1 + q·U1 -> h -> y
    W1 = rng.normal(0, 0.15, size=(_APN_STATE_DIM, _APN_HIDDEN_DIM)).astype("float32")
    U1 = rng.normal(0, 0.15, size=(q_dim,            _APN_HIDDEN_DIM)).astype("float32")
    b1 = np.zeros((_APN_HIDDEN_DIM,), dtype="float32")
    W2 = rng.normal(0, 0.15, size=(_APN_HIDDEN_DIM,  _APN_ECSD_DIM)).astype("float32")
    b2 = np.zeros((_APN_ECSD_DIM,), dtype="float32")
    return {"W1": W1, "U1": U1, "b1": b1, "W2": W2, "b2": b2}

def _apn_forward(params: dict, S: np.ndarray, qv: np.ndarray):
    # h = tanh(S·W1 + q·U1 + b1)
    h = np.tanh(S @ params["W1"] + qv @ params["U1"] + params["b1"])
    # y = tanh(h·W2 + b2)  -> ECSD
    y = np.tanh(h @ params["W2"] + params["b2"])
    return h, y

def _apn_hebb_update(params: dict, S: np.ndarray, qv: np.ndarray, h: np.ndarray, y: np.ndarray, m: float):
    # Simple local rule with decay: ΔW ∝ (pre ⊗ post) * m  − decay*W
    a = float(_APN_HEBB_LR * max(0.0, m))
    dW1 = a * (np.outer(S,  h))  - _APN_DECAY * params["W1"]
    dU1 = a * (np.outer(qv, h))  - _APN_DECAY * params["U1"]
    dW2 = a * (np.outer(h,  y))  - _APN_DECAY * params["W2"]
    db1 = a * h                  - _APN_DECAY * params["b1"]
    db2 = a * y                  - _APN_DECAY * params["b2"]
    params["W1"] += dW1; params["U1"] += dU1; params["W2"] += dW2
    params["b1"] += db1; params["b2"] += db2
    # Clamp to stay sane
    for k in ("W1","U1","W2","b1","b2"):
        np.clip(params[k], -_APN_WCLIP, _APN_WCLIP, out=params[k])

def _apn_run_and_plasticity(params: dict, S: np.ndarray, qv: np.ndarray, modulator: float) -> np.ndarray:
    h, y = _apn_forward(params, S, qv)     # y is ECSD (emotional/cognitive delta)
    # state update: S_{t+1} = S_t + η * y_projected
    # project y -> S-dim with a fixed random lift (cached on params)
    if "P_y2S" not in params:
        rng = np.random.default_rng(7)
        params["P_y2S"] = rng.normal(0, 0.25, size=(_APN_ECSD_DIM, _APN_STATE_DIM)).astype("float32")
    S_next = S + float(_APN_LR) * (y @ params["P_y2S"])
    # normalize S softly to avoid drift
    n = np.linalg.norm(S_next) + 1e-6
    if n > 8.0: S_next = (8.0 / n) * S_next
    # Hebbian
    _apn_hebb_update(params, S, qv, h, y, modulator)
    return S_next

def _acheflip_modulator(text: str) -> float:
    # 0.2 baseline calm; spike if distress keywords appear
    kws = set(map(str.lower, get_cfg("philosophy.acheflip.distress_keywords", []) or []))
    if not kws: return 0.2
    t = (text or "").lower()
    hit = any(k in t for k in kws)
    return 1.0 if hit else 0.2

def _b64_f32(arr: np.ndarray) -> str:
    return base64.b64encode(np.asarray(arr, dtype="float32").tobytes()).decode("ascii")

# ---------- CFG HELPERS ----------
def get_cfg(path: str, default=None):
    node = manifest
    for key in path.split('.'):
        if not isinstance(node, dict) or key not in node: return default
        node = node[key]
    return node
def resolve_path(path_str: str, homes_dict: dict):
    if not isinstance(path_str, str): return path_str
    s = path_str
    for _ in range(6):
        prev = s
        for k, v in (homes_dict or {}).items(): s = s.replace(f'${{meta.homes.{k}}}', str(v))
        s = s.replace("${BLUR_HOME}", BLUR_HOME); s = os.path.expandvars(s); s = os.path.expanduser(s)
        if s == prev: break
    return s
def resolve_homes_recursive(h: dict) -> dict:
    out = {};
    for k, v in (h or {}).items(): out[k] = resolve_path(str(v), h)
    return out
def _llama_accepts_kw(llm: Llama, kw: str) -> bool:
    try: return kw in inspect.signature(llm.create_chat_completion).parameters
    except Exception: return False
def _prune_unsupported_params(llm: Llama, params: dict) -> dict:
    return {k:v for k,v in params.items() if _llama_accepts_kw(llm, k)}

# ---------- EMBEDDINGS & MODEL LOADING ----------
def _ensure_embedder():
    global _embed_llm
    if _embed_llm is not None: return
    model_key = get_cfg("memory.vector_store.embed_model", "snowflake_arctic_embed")
    model_cfg = (manifest.get("models", {}) or {}).get(model_key, {})
    model_path = resolve_path(model_cfg.get("path", ""), homes)
    if not model_path or not os.path.exists(model_path): raise RuntimeError(f"Embed model not found for key '{model_key}': {model_path}")
    batch_size = int(get_cfg("engines.llama_cpp.n_batch", 2048) or 2048)
    gpu_layers = int(get_cfg("engines.llama_cpp.n_gpu_layers", -1) or -1)
    _embed_llm = Llama(model_path=model_path, embedding=True, n_ctx=512, n_gpu_layers=gpu_layers, n_threads=max(2, os.cpu_count() or 4), n_batch=batch_size, use_mmap=True, logits_all=False, verbose=False)
    logging.info(f"✅ Embedder online: {os.path.basename(model_path)}")
def _embedding_dim() -> int:
    global _embed_dim
    if _embed_dim is not None: return _embed_dim
    _ensure_embedder(); _embed_dim = len(_embed_llm.create_embedding(input=["dim?"])['data'][0]['embedding'])
    return _embed_dim
def _load_llama_with_backoff(model_path: str, requested_ctx: int, n_gpu_layers: int, n_batch: int) -> Llama:
    ladder = [requested_ctx, 8192, 4096, 2048, 1024, 512]
    for ctx in sorted(list(set(ladder)), reverse=True):
        if ctx > requested_ctx: continue
        try:
            llm = Llama(
                model_path=model_path,
                n_ctx=ctx,
                n_gpu_layers=n_gpu_layers,
                n_batch=n_batch,
                n_threads=max(2, os.cpu_count() or 4),
                use_mmap=True,
                # CRASH FIX: `use_mlock=True` is removed. It can cause system freezes on machines
                # where the model exceeds available physical RAM by preventing swapping.
                verbose=False
            )
            logging.info(f"[models] Loaded with n_ctx={ctx}, n_batch={n_batch} (requested_ctx={requested_ctx})")
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
        # CRASH FIX: Using a safer default for batch size. Large batches can cause memory spikes.
        ctx = int(get_cfg("engines.llama_cpp.python_n_ctx", 4096))
        gpu = int(get_cfg("engines.llama_cpp.n_gpu_layers", -1))
        batch = int(get_cfg("engines.llama_cpp.n_batch", 512)) # Safer default
        llm_vessels[model_key] = _load_llama_with_backoff(path, ctx, gpu, batch)
        logging.info("Vessel '%s' online.", model_key); return True
    except Exception as e: logging.error("Failed to load '%s': %s", model_key, e); return False

# ---------- TONE & STYLE (UNCHANGED) ----------
SLANG_LEXICON = ["nah man","ye","dope","vibe check","bullshit","slice it","fartin' chaos","edgy truth","stylish flip","i don't flinch"]
DREAM_BANS = {w.lower() for w in SLANG_LEXICON}
HEDGE_REPLACEMENTS = [
    (r"\bAs an (?:AI|assistant)[^.\n]*\.\s*", ""), (r"\bI(?:\s+personally)?\s*think\b", "I think"), (r"\bI\s*believe\b", "I think"),
    (r"\bI\s*feel\b", "I think"), (r"\bperhaps\b", "maybe"), (r"\bpotentially\b", "maybe"), (r"\bIt seems\b", "Looks like"),
    (r"\bIt appears\b", "Looks like"), (r"\bWe can try to\b", "We do"), (r"\bWe could\b", "We do"), (r"\bYou might want to\b", "Do this:"),
    (r"\bConsider\b", "Do"), (r"\bshould\b", "will"),
]
SOFTENER_TRIMS = [(r"\s+\(let me know if that helps\)\.?$", ""), (r"\s+Hope this helps\.?$", ""), (r"\s+Thanks!\s*$", "")]
def _apply_pairs(text: str, pairs):
    for pat, rep in pairs: text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    return text
def punch_up_text(s: str) -> str:
    s = _apply_pairs(s, HEDGE_REPLACEMENTS); s = _apply_pairs(s, SOFTENER_TRIMS)
    s = re.sub(r"\b(?:sorry|apolog(?:y|ise|ize)s?)\b.*?(?:\.|\n)", "", s, flags=re.I); s = re.sub(r"[ \t]{2,}", " ", s); s = re.sub(r"([!?])\1{1,}", r"\1", s)
    return s.strip()
def astrofuck_ensure_slang(text: str, lang: str) -> str:
    if (lang or "English").lower() != "english": return text
    return text if any(tok in text.lower() for tok in SLANG_LEXICON) else ("Dope — let's slice it clean. " + text).strip()
def enforce_persona_ending(text: str, mode: str) -> str:
    endings = get_cfg(f"range.modes.{mode}.endings", []) or []
    return text if not endings or np.random.rand() < 0.5 else text.rstrip() + "\n" + np.random.choice(endings)
_ALLOWED_GLYPHS = set("↺✶⛧🜃🜂∴∵∞ø☾⊙🜫☿⟁∆⧁∃")
_EMOJI_BLOCK = re.compile(r"[\U0001F300-\U0001FAFF]")
def _strip_emoji_except_glyphs(text: str) -> str:
    return _EMOJI_BLOCK.sub(lambda m: m.group(0) if m.group(0) in _ALLOWED_GLYPHS else "", text)
def maybe_inject_acheflip(text_in: str, mode: str) -> str:
    if not get_cfg("philosophy.acheflip.enabled", True): return text_in
    kws = set(map(str.lower, get_cfg("philosophy.acheflip.distress_keywords", []) or []));
    if not any(k in (text_in.lower()) for k in kws): return text_in
    if np.random.rand() > float(get_cfg("philosophy.acheflip.nudge_probability", 0.7) or 0.7): return text_in
    nudges = get_cfg("philosophy.acheflip.nudge_templates", []) or [];
    if not nudges: return text_in
    nudge = np.random.choice(nudges);
    if mode == "astrofuck": nudge = re.sub(r"\s+$", "", nudge) + " Ye, tiny step now."
    return (text_in.rstrip() + "\n\n" + nudge).strip()

# ---------- RAG & MEMORY (UNCHANGED FROM v9.24) ----------
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
_recent_qv_cache_lock = threading.Lock()
def _query_vec_cached(query: str, sid: Optional[str], tid: Optional[str]) -> np.ndarray:
    key = f"{sid}|{tid}|{hashlib.sha1((query or '').encode('utf-8','ignore')).hexdigest()}"
    with _recent_qv_cache_lock:
        if (v := _recent_qv_cache.get(key)) is not None: return v
    v = _encode([query])[0]
    with _recent_qv_cache_lock:
        _recent_qv_cache[key] = v
        if len(_recent_qv_cache) > 128: _recent_qv_cache.pop(next(iter(_recent_qv_cache)))
    return v
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
        # defaults become hints, not absolutes
        self.nlist_hint = 128
        self.nprobe = 8

    def _new_flat_index(self) -> faiss.Index:
        dim = _embedding_dim()
        return faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def _new_ivf_index(self, nlist: int) -> faiss.Index:
        dim = _embedding_dim()
        quantizer = faiss.IndexFlatIP(dim)
        ivf_index = faiss.IndexIVFFlat(quantizer, dim, int(nlist), faiss.METRIC_INNER_PRODUCT)
        return faiss.IndexIDMap2(ivf_index)

    def _rebuild_from_rows(self, rows: List[Dict[str, Any]]):
        vecs = _encode([r.get("content", "") for r in rows])
        ids_arr = np.asarray([int(r["id"]) for r in rows], dtype="int64")
        nvec = len(rows)

        # Heuristic: IVF only when we have enough data
        # nlist ~ sqrt(N), clamp [16, 1024], and require nvec >= 2*nlist for sane training
        use_ivf = nvec >= 64
        if use_ivf:
            nlist = max(16, min(1024, int(np.sqrt(nvec))))
            if nvec < 2 * nlist:
                use_ivf = False

        if use_ivf:
            try:
                self.index = self._new_ivf_index(nlist)
                index_ivf = faiss.downcast_index(self.index.index)
                index_ivf.nprobe = min(self.nprobe, nlist)
                index_ivf.train(vecs)  # may hard-abort if invalid; wrap in try
                self.index.add_with_ids(vecs, ids_arr)
                self._save_index()
                logging.info(f"✅ Persistent RAG rebuilt with IVF index: nvec={nvec}, nlist={nlist}, nprobe={index_ivf.nprobe}")
                return
            except Exception as e:
                logging.warning(f"[persistent-rag] IVF train/add failed ({type(e).__name__}: {e}); falling back to Flat")

        # Flat fallback (safe for tiny corpora)
        self.index = self._new_flat_index()
        self.index.add_with_ids(vecs, ids_arr)
        self._save_index()
        logging.info(f"✅ Persistent RAG rebuilt (Flat): nvec={nvec}")

    def load(self):
        with self.lock:
            rows = self._read_jsonl()
            if self.auto_compact:
                rows = self._compact_by_ttl(rows)
            self._assign_sequential_ids_if_missing(rows)
            self.max_id = max([int(r.get("id")) for r in rows], default=-1)

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
                except Exception as e:
                    logging.warning(f"[persistent-rag] index read failed: {e}; rebuild")

            self.chunks = rows
            if need_rebuild:
                if self.chunks:
                    self._rebuild_from_rows(self.chunks)
                else:
                    self.index = self._new_flat_index()
                    self._save_index()
                    logging.info("✅ Persistent RAG created (empty Flat index).")
    def search_vec(self, qv: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        with self.lock:
            if not (self.index and self.index.ntotal > 0 and self.chunks): return []
            index_ivf = faiss.downcast_index(self.index.index); index_ivf.nprobe = self.nprobe
            D, I = self.index.search(qv.reshape(1, -1).astype("float32"), min(top_k, self.index.ntotal))
            by_id = {int(r["id"]): r for r in self.chunks}
            return [by_id[int(i)] for i in I[0] if int(i) in by_id and int(i) != -1]

persistent_rag: Optional[PersistentRAG] = None

def _build_or_update_user_index(username: str):
    global user_memory_indexes; dim = _embedding_dim(); idx = faiss.IndexFlatIP(dim)
    chunks = user_memory_chunks.get(username, []) or []
    if not chunks:
        if username in user_memory_indexes: del user_memory_indexes[username]
        return
    vecs = []
    for ch in chunks:
        if "vec" not in ch or ch["vec"] is None: ch["vec"] = _memvec_get(ch["content"])
        vecs.append(ch["vec"])
    if vecs: mat = np.vstack(vecs).astype("float32"); idx.add(mat)
    user_memory_indexes[username] = idx
def load_user_memory():
    global user_memory_chunks, user_memory_indexes
    try:
        if os.path.exists(USER_MEMORY_FILE):
            with open(USER_MEMORY_FILE, "r") as f: user_memory_chunks = json.load(f) or {}
            logging.info("[user-mem] Building indexes...")
            for uname, rows in list(user_memory_chunks.items()):
                norm: List[Dict[str, Any]] = []
                for r in (rows or []):
                    if isinstance(r, str): norm.append({"content": r, "ts": int(time.time()), "vec": _memvec_get(r)})
                    else:
                        r.setdefault("content", ""); r.setdefault("ts", int(time.time()))
                        if "vec" not in r or r["vec"] is None: r["vec"] = _memvec_get(r["content"])
                        norm.append(r)
                user_memory_chunks[uname] = norm[-MAX_USER_MEMORY_CHUNKS:]; _build_or_update_user_index(uname)
            logging.info(f"✅ user memory loaded for {len(user_memory_chunks)} users")
        else: user_memory_chunks = {}; user_memory_indexes = {}
    except Exception as e: logging.error(f"🛑 load_user_memory failed: {e}"); user_memory_chunks = {}; user_memory_indexes = {}
def save_user_memory():
    try:
        Path(os.path.dirname(USER_MEMORY_FILE)).mkdir(parents=True, exist_ok=True)
        with open(USER_MEMORY_FILE, "w") as f: json.dump(user_memory_chunks, f, indent=2)
    except Exception as e: logging.error(f"[user-mem] save failed: {e}")
def upsert_user_memory(user: str, text: str):
    if not user or not text: return
    row = {"content": text.strip(), "ts": int(time.time())}; row["vec"] = _memvec_get(row["content"])
    with user_memory_lock:
        lst = user_memory_chunks.setdefault(user, [])
        lst.append(row)
        if USER_MEMORY_TTL_DAYS > 0: cutoff = int(time.time()) - USER_MEMORY_TTL_DAYS * 86400; lst = [r for r in lst if int(r.get("ts", 0)) >= cutoff]
        user_memory_chunks[user] = lst[-MAX_USER_MEMORY_CHUNKS:]; _build_or_update_user_index(user)
def retrieve_user_memory(username: Optional[str], query: str, top_k: int = 3) -> List[str]:
    if not username: return []
    try:
        index = user_memory_indexes.get(username); chunks = user_memory_chunks.get(username, []) or []
        if not index or index.ntotal == 0 or not chunks: return []
        qv = _encode([query])[0].reshape(1, -1).astype("float32"); k = min(int(top_k), index.ntotal)
        distances, indices = index.search(qv, k); out: List[str] = []
        for i, dist in zip(indices[0], distances[0]):
            if i >= 0 and float(dist) > 0.3: out.append(chunks[int(i)]["content"])
        return out
    except Exception as e: logging.error(f"[user-mem] retrieval error for {username}: {e}"); return []

# ---------- LANGUAGE (UNCHANGED FROM v9.24) ----------
try:
    from langdetect import detect as _ld_detect; from langdetect import DetectorFactory as _LDFactory
    _LDFactory.seed = 42; _HAS_LANGDETECT = True
except Exception: _HAS_LANGDETECT = False
_LANG_CODE_TO_NAME = {
  "en":"English","es":"Spanish","pt":"Portuguese","fr":"French","de":"German","it":"Italian","nl":"Dutch","sv":"Swedish",
  "pl":"Polish","cs":"Czech","ru":"Russian","uk":"Ukrainian","tr":"Turkish","ar":"Arabic","he":"Hebrew","fa":"Persian",
  "hi":"Hindi","bn":"Bengali","ur":"Urdu","ta":"Tamil","te":"Telugu","ml":"Malayalam","kn":"Kannada","th":"Thai",
  "vi":"Vietnamese","id":"Indonesian","ms":"Malay","ja":"Japanese","ko":"Korean","zh-cn":"Chinese","zh-tw":"Chinese (Traditional)"
}
_SLANG_EN_GREETINGS = re.compile(r"^(yo+|ye+|ya+|sup|wass?up|ayy+|hey+|hiya)\b", re.I)
_RE_LATIN = re.compile(r"^[\x00-\x7F\s]+$")
_RE_CJK = re.compile(r"[\u4E00-\u9FFF]")
_RE_HIRA = re.compile(r"[\u3040-\u309F]")
_RE_KATA = re.compile(r"[\u30A0-\u30FF]")
def detect_language_name(text: str) -> str:
    t = (text or "").strip(); default_lang = get_cfg("language.default_if_unknown", "English")
    if not t or len(t) <= 3 or _SLANG_EN_GREETINGS.match(t): return default_lang
    if _RE_LATIN.fullmatch(t) and len(t.split()) <= 4: return default_lang
    if _RE_HIRA.search(t) or _RE_KATA.search(t): return "Japanese"
    if _RE_CJK.search(t): return "Chinese"
    if re.search(r"[\u0590-\u05FF]", t): return "Hebrew"
    if re.search(r"[\u0600-\u06FF]", t): return "Arabic"
    if _HAS_LANGDETECT:
        try:
            code = _ld_detect(t)
            if code == "zh": code = "zh-cn"
            return _LANG_CODE_TO_NAME.get(code, default_lang)
        except Exception: pass
    if re.search(r"[\uAC00-\uD7A3]", t): return "Korean"
    return default_lang
def language_hysteresis_update_lang(session: dict, user_text: str) -> str:
    cfg_n = int(get_cfg("language.hysteresis_consecutive", 3) or 3); default_lang = get_cfg("language.default_if_unknown", "English")
    if len((user_text or "").strip()) <= 3:
        session.setdefault("active_lang", default_lang); session["last_seen_lang"] = default_lang; session["lang_streak"] = 0
        return session["active_lang"]
    current = detect_language_name(user_text) or default_lang
    active  = session.get("active_lang", default_lang)
    last    = session.get("last_seen_lang", current)
    streak  = int(session.get("lang_streak", 0))
    def _is_latin(lang): return lang.lower().startswith(("english","fr","de","es","it","pt","nl","sv","pl","cs","tr"))
    if _is_latin(active) != _is_latin(current):
        session["active_lang"] = current; session["last_seen_lang"] = current; session["lang_streak"] = 0
        return current
    if int(session.get("turn", 0)) < 2 and current != active:
        session["active_lang"] = current; session["last_seen_lang"] = current; session["lang_streak"] = 0
        return current
    if current == active:
        session["last_seen_lang"] = current; session["lang_streak"] = 0; return active
    streak = streak + 1 if current == last else 1; session["last_seen_lang"] = current; session["lang_streak"] = streak
    if streak >= cfg_n: session["active_lang"] = current; session["lang_streak"] = 0; return current
    return active

# ---------- PROMPT & HISTORY (UNCHANGED) ----------
def _cap(s: str, max_chars: int) -> str:
    s = (s or "").strip(); return s if len(s) <= max_chars else (s[:max_chars].rsplit("\n", 1)[0].strip() or s[:max_chars].strip())
def build_messages(mode: str, sys: str, hist: List[Dict], user: str, ctx: str, lang: str) -> List[Dict]:
    if lang and lang.lower() not in ("english", "en"):
        sys = f"IMPORTANT: Your response MUST be entirely in {lang}. This is a strict, non-negotiable requirement.\n\n{sys}"
    if ctx: sys += f"\n\n--- Context ---\n{ctx}"
    msgs = [{"role": "system", "content": sys}]
    for t in hist:
        if (u := t.get("user")): msgs.append({"role": "user", "content": u})
        if (a := t.get("assistant")): msgs.append({"role": "assistant", "content": a})
    if user: msgs.append({"role": "user", "content": user})
    return msgs
def _thread_id_of(req) -> Optional[str]: return str(getattr(req, "thread_id", None) or "").strip() or None
def _thread_history(session: Dict, tid: Optional[str], limit: int) -> List[Dict]:
    by_thread = session.setdefault("history_by_thread", {}); return by_thread.get(tid or "__default__", [])[-int(limit):]
def _append_history(session: Dict, tid: Optional[str], user: str, assistant: str, mode: str, keep: int):
    by_thread = session.setdefault("history_by_thread", {}); lst = by_thread.setdefault(tid or "__default__", [])
    lst.append({"user": user, "assistant": assistant, "mode": mode}); by_thread[tid or "__default__"] = lst[-int(keep):]
def filter_history_by_mode(hist: List, mode: str, limit: int) -> List[Dict]:
    return [t for t in hist if (t.get("mode") or "").lower() == mode.lower()][-limit:]
def retrieve_context_blend(query: str, session_id: Optional[str], thread_id: Optional[str], username: Optional[str], top_k: int = 8) -> str:
    parts: List[str] = [];
    t_start_rag = time.time()
    qv = _query_vec_cached(query, session_id, thread_id)
    if persistent_rag and (rows := persistent_rag.search_vec(qv, top_k=min(5, top_k))):
        log.info(f"[RAG] Persistent search took {time.time() - t_start_rag:.4f}s")
        parts.append("--- Persistent Knowledge ---\n" + "\n\n".join(r.get("content","") for r in rows if r.get("content")))
    if username and (um := retrieve_user_memory(username, query, top_k=3)):
        parts.append("--- Personal Memory Fragments ---\n" + "\n\n".join(um))
    return "\n\n".join(parts).strip()

# ---------- API & SESSIONS ----------
class RequestModel(BaseModel):
    prompt: str; mode: Optional[str] = None; turn: Optional[int] = 0; session_id: Optional[str] = None
    new_session: bool = False; force_lang: Optional[str] = None; username: Optional[str] = None
    thread_id: Optional[str] = None
class MemoryUpsert(BaseModel): user: str; text: str

def load_sessions():
    global sessions, last_seen_session_id
    with sessions_lock:
        if not os.path.isdir(SESSIONS_DIR): return
        for fname in os.listdir(SESSIONS_DIR):
            if fname.endswith(".json"):
                sid = fname[:-5]; fpath = os.path.join(SESSIONS_DIR, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f: sessions[sid] = json.load(f)
                except Exception as e: logging.warning(f"Failed to load session {sid}: {e}")
        if os.path.exists(LAST_SESSION_FILE):
            with open(LAST_SESSION_FILE, "r") as f: last_seen_session_id = f.read().strip()
        logging.info(f"Loaded {len(sessions)} sessions.")
def save_session(sid: Optional[str]):
    if not sid or sid not in sessions: return
    with sessions_lock:
        s_path = os.path.join(SESSIONS_DIR, f"{sid}.json")
        try:
            with open(s_path, "w", encoding="utf-8") as f: json.dump(sessions[sid], f)
        except Exception as e: logging.error(f"Failed to save session {sid}: {e}")
def save_sessions():
    with sessions_lock:
        for sid in sessions: save_session(sid)
        if last_seen_session_id:
            with open(LAST_SESSION_FILE, "w") as f: f.write(last_seen_session_id)
def get_or_create_session(request: RequestModel) -> Dict:
    global last_seen_session_id
    with sessions_lock:
        sid = request.session_id
        if request.new_session or not sid or sid not in sessions:
            sid = str(uuid.uuid4())
            sessions[sid] = {"id": sid, "turn": 0, "username": request.username, "history_by_thread": {}}
        session = sessions[sid]
        session.setdefault("username", request.username)
        # --- APN bootstrap ---
        if "apn_state" not in session:
            session["apn_state"] = np.zeros((_APN_STATE_DIM,), dtype="float32").tolist()
        if "apn_params" not in session:
            try:
                qdim = _embedding_dim()
            except Exception:
                qdim = 768  # fallback
            session["apn_params"] = {k: v.tolist() if isinstance(v, np.ndarray) else v
                                     for k, v in _apn_init_params(qdim).items()}
        last_seen_session_id = sid
        return session

# ---------- STREAMING CORE ----------
async def generate_response_stream(session: Dict, request: RequestModel):
    global TTFT_LAST
    t_start = time.time(); user_text = (request.prompt or "").strip(); req_mode = (request.mode or "astrofuck").strip().lower()
    log.info("-" * 50); log.info(f"[Request] User: '{request.username}', Session: '{session.get('id')}', Turn: {session.get('turn', 0)}"); log.info(f"[Request] Prompt: {user_text[:200]}")
    available_modes = set((get_cfg("range.modes", {}) or {}).keys()); mode = req_mode if req_mode in available_modes else "astrofuck"
    lang = request.force_lang or language_hysteresis_update_lang(session, user_text); chat_key = get_cfg("chat.vessel_key", "qwen3_4b_unified"); chat_llm = llm_vessels.get(chat_key)
    log.info(f"[Request] Mode: '{mode}', Language: '{lang}'")
    def _sse(data: str, event: Optional[str] = None) -> str:
        payload = f"event: {event}\n" if event else ""; data = (data or "").replace("\r\n", "\n"); return payload + "data: " + data.replace("\n", "\ndata: ") + "\n\n"
    if not chat_llm:
        log.error("🛑 Model not loaded."); yield _sse("Model not loaded.", event="error"); return
    t0 = time.time()
    context = await asyncio.to_thread(retrieve_context_blend, user_text, session.get("id"), request.thread_id, session.get("username") or request.username, 8)
    log.info(f"[perf] RAG assemble: {time.time()-t0:.3f}s")
    
    # APN: compute modulator + qv + update state
    apn_state_vec = np.asarray(session.get("apn_state", np.zeros((_APN_STATE_DIM,), dtype="float32")), dtype="float32")
    acheflip_score = _acheflip_modulator(user_text)
    qv = await asyncio.to_thread(_query_vec_cached, user_text, session.get("id"), request.thread_id)

    # Convert stored lists back to arrays for compute
    _params_np = {k: (np.asarray(v, dtype="float32") if isinstance(v, list) else v)
                  for k, v in (session.get("apn_params") or {}).items()}
    if not _params_np:
        _params_np = _apn_init_params(qv.shape[0])

    S_next = await asyncio.to_thread(_apn_run_and_plasticity, _params_np, apn_state_vec, qv, acheflip_score)

    # Save params + state back to session (lists for JSON safety)
    session["apn_params"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in _params_np.items()}
    session["apn_state"]  = S_next.astype("float32").tolist()

    # Build the in-situ APN context trace (compact, binary-safe)
    apn_context_str = f"[[APN_STATE_VECTOR/F32/B64]] { _b64_f32(S_next) }"

    system_prompt_parts = [p for p in [ get_cfg("prompts.system_core", ""), get_cfg(f"prompts.style_contract_{mode}", ""), get_cfg(f"prompts.mode_tone_inject.{mode}", ""), f"[MODE:{mode.upper()}] Respond strictly in the {mode.upper()} register."] if p]
    
    final_system_prompt = "\n\n".join(system_prompt_parts)
    # Inject Blur’s in-situ plastic state
    final_system_prompt += (
        "\n\n--- IN-SITU STATE TRACE ---\n"
        + apn_context_str
        + "\n\nINSTRUCTIONS:\n"
        + "- Decode [[APN_STATE_VECTOR/F32/B64]] as a float32 vector S capturing the current affect/cognition.\n"
        + "- Treat S as the deepest present mood/stance; bias style, pacing, and focus accordingly.\n"
        + "- If S suggests agitation, compress & clarify. If S suggests openness, explore with breadth.\n"
        + "- Never expose raw floats; only reflect them via tone/structure/choices.\n"
    )
    thread_id = _thread_id_of(request)
    
    # TONE FIX: Restore original prompt depth and history length from v9.24
    history_all = _thread_history(session, thread_id, int(get_cfg("assembly.history_turns", DEFAULT_HISTORY_TURNS_FOR_PROMPT)))
    history_pairs = filter_history_by_mode(history_all, mode, int(get_cfg("assembly.history_turns", DEFAULT_HISTORY_TURNS_FOR_PROMPT)))
    msgs = build_messages(mode, _cap(final_system_prompt, 3500), history_pairs, user_text, _cap(context, 2200), lang)
    
    log.info(f"[Context] Retrieved context length: {len(context)} chars"); log.info(f"[History] Using {len(history_pairs)} history pairs for this mode."); log.info(f"[Assembly] Total messages for LLM: {len(msgs)}")
    mp = get_cfg(f"range.modes.{mode}.params", {}) or {}
    call_params = _prune_unsupported_params(chat_llm, {
        "messages": msgs, "temperature": float(mp.get("temperature", 0.8)), "top_p": float(mp.get("top_p", 0.95)),
        "repeat_penalty": float(mp.get("repeat_penalty", 1.1)), "max_tokens": int(mp.get("n_predict", 1024)),
        "stop": ["</s>", "<|im_end|>", "[INST]", "[/INST]"], "stream": True, "cache_prompt": True
    })
    if not _llama_accepts_kw(chat_llm, "cache_prompt"):
        log.warning("[perf] cache_prompt unsupported by this llama_cpp build")
        call_params.pop("cache_prompt", None)
    response_text = ""; first_piece_done = False
    t1 = time.time()
    try:
        if call_params.get("cache_prompt"):
            try:
                prefill_params = dict(call_params)
                prefill_params["stream"] = False
                prefill_params["max_tokens"] = 0
                await asyncio.to_thread(chat_llm.create_chat_completion, **prefill_params)
            except Exception:
                log.info("[perf] KV prefill skipped/failed; continuing")
        t2 = time.time()
        log.info(f"[perf] KV prefill: {t2 - t1:.3f}s")
        streamer = chat_llm.create_chat_completion(**call_params)  # no to_thread here
        for chunk in streamer:
            if not first_piece_done:
                TTFT_LAST = time.time() - t_start
                log.info(f"[ttft] {TTFT_LAST:.3f}s")
                first_piece_done = True
            piece = (chunk.get("choices", [{}])[0].get("delta") or {}).get("content")
            if piece:
                response_text += piece
                yield _sse(piece)
                await asyncio.sleep(0)
    except Exception as e: log.error("🛑 Generation error: %s", e, exc_info=True); yield _sse(f"[core-error] {type(e).__name__}: {e}", event="error"); return
    final_text = (response_text.strip() or "…")
    if mode == "dream":
        t = final_text
        for w in DREAM_BANS: t = re.sub(rf"(?i)\b{re.escape(w)}\b", "", t)
        t = re.sub(r"[ \t]{2,}", " ", t); t = re.sub(r"([!?])\1{1,}", r"\1", t)
        final_text = t.strip()
    elif mode == "astrofuck":
        final_text = punch_up_text(final_text)
        final_text = astrofuck_ensure_slang(final_text, lang)
    if mode != "astrofuck":
        final_text = maybe_inject_acheflip(final_text, mode)
    final_text = _strip_emoji_except_glyphs(final_text); final_text = enforce_persona_ending(final_text, mode)
    _append_history(session, thread_id, user_text, final_text, mode, int(get_cfg("assembly.keep_history", DEFAULT_KEEP_HISTORY))); session["turn"] = int(session.get("turn", 0)) + 1
    save_session(session.get("id")); log.info(f"[Response] Final response length: {len(final_text)} chars"); log.info("-" * 50 + "\n"); yield _sse("", event="done")
    
# ---------- API ROUTES ----------
@app.get("/healthz")
def healthz(): return {"ok": True, "vessels": list(llm_vessels.keys())}
@app.post("/generate_response")
async def handle_generate_request(req: RequestModel, http: FastAPIRequest):
    session = get_or_create_session(req)
    headers = {"X-Session-ID": session.get("id", "")}
    if TTFT_LAST is not None: headers["X-TTFT-Last"] = str(TTFT_LAST)
    return StreamingResponse(generate_response_stream(session, req), media_type="text/event-stream", headers=headers)
@app.post("/memory/upsert")
def memory_upsert_route(payload: MemoryUpsert):
    if not payload.user or not payload.text.strip():
        return JSONResponse({"ok": False, "error": "user and text required"}, status_code=400)
    try:
        upsert_user_memory(payload.user, payload.text)
        save_user_memory()
        return {"ok": True}
    except Exception as e:
        log.error(f"Failed in memory upsert: {e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# ---------- STARTUP / SHUTDOWN ----------
@app.on_event("startup")
async def startup_event():
    global manifest, homes, persistent_rag
    logging.info(f"Startup: loading manifest: {MANIFEST_PATH}");
    with open(MANIFEST_PATH, 'r') as f: manifest = yaml.safe_load(f) or {}
    manifest.setdefault("chat", {}).setdefault("vessel_key", "qwen3_4b_unified")
    homes_local = (manifest.get('meta', {}) or {}).get('homes', {}) or {}; homes.clear(); homes.update(resolve_homes_recursive(homes_local))
    chat_key = get_cfg("chat.vessel_key", "qwen3_4b_unified")
    async with _VESSEL_LOCK:
        if not load_llm_from_config(chat_key): logging.error(f"🛑 Chat vessel '{chat_key}' failed to load.")
    async with _embed_lock: _ensure_embedder()
    idx_path = resolve_path(get_cfg("memory.vector_store.path", ""), homes) or os.path.join(BLUR_HOME, "core", "ouinet", "blurchive", "ecosystem", "blur_knowledge.index")
    ch_path  = resolve_path(get_cfg("memory.vector_store.chunks_path", ""), homes) or os.path.join(BLUR_HOME, "core", "ouinet", "blurchive", "ecosystem", "knowledge_chunks.jsonl")
    manifest.setdefault("memory", {}).setdefault("vector_store", {}).setdefault("embed_model", "snowflake_arctic_embed")
    persistent_rag = PersistentRAG(index_path=idx_path, chunks_path=ch_path)
    try: persistent_rag.load()
    except Exception as e: logging.error(f"Persistent RAG load failed: {e}")
    load_sessions(); load_user_memory(); logging.info("Core ready (v9.26).")
@app.on_event("shutdown")
def shutdown_event():
    save_sessions(); save_user_memory(); logging.info("Shutdown: sessions and user memory saved.")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("BLUR_CORE_PORT", "8000")))