#!/usr/bin/env python3
# convo_chat_core.py — Reforged v10.4 (RAG revival + persona isolation + bundle-first paths)
# - Bundle-first path resolution: CONFIG_DIR/Resources/BLUR_HOME with env overrides
# - Supports ${CONFIG_DIR}, ${RESOURCES_DIR}, ${BLUR_MODELS}, ${BLUR_HOME}, meta.homes.*
# - Robust model discovery for both embedder and chat LLM
# - FIX: RAG path defaults + validation; reuse FAISS index when dims match; auto-rebuild otherwise
# - FIX: Robust row content extraction (content|text|body|…)
# - FIX: Softer AF blocklist (avoid wiping context)
# - ADD: /rag/status and /rag/reload debug routes
# - RESTORE: load_sessions(), load_user_memory(), CORE_IS_READY=True on startup
# - KEEP: stream/TTFT, persona-isolated AF, user memory

from __future__ import annotations

import sys, os, logging, asyncio, yaml, json, uuid, re, time, threading, inspect, base64, hashlib, subprocess
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
import numpy as np

# ---------- Optional deps ----------
try:
    import faiss
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 4)//2))
except Exception as e:
    print("🛑 faiss not found: pip install faiss-cpu", file=sys.stderr); sys.exit(1)

try:
    from llama_cpp import Llama  # native, may be absent on x64 build
except Exception:
    Llama = None
    print("⚠️ llama_cpp not present — running in degraded mode (no local GGUF inference).", file=sys.stderr)

from fastapi import FastAPI, Request as FastAPIRequest, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# --------- JSON response selection (orjson -> ujson -> stdlib) ---------
BestJSONResponse = JSONResponse
try:
    import orjson  # fastest, but optional
    from fastapi.responses import ORJSONResponse as BestJSONResponse
except Exception:
    try:
        import ujson  # fast + easy wheels
        class UJSONResponse(JSONResponse):
            media_type = "application/json"
            def render(self, content) -> bytes:
                # compact utf-8, mirrors ORJSON behavior enough for APIs
                return ujson.dumps(content, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        BestJSONResponse = UJSONResponse
    except Exception:
        pass

# ---------- Logging ----------
os.environ.setdefault("GGML_LOG_LEVEL", "WARN")
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='INFO:     %(message)s')
log = logging.getLogger("core")

# ---------- Globals / Defaults ----------
DEFAULT_HOME = "~/blur"
BLUR_HOME = os.path.expanduser(os.getenv("BLUR_HOME", DEFAULT_HOME))

# sessions live independently at ~/.blur/sessions (or override via BLUR_SESSIONS_DIR)
DEFAULT_SESS_HOME = "~/.blur/sessions"
SESSIONS_DIR = os.path.expanduser(os.getenv("BLUR_SESSIONS_DIR", DEFAULT_SESS_HOME))

# default state dir inside BLUR_HOME unless overridden
STATE_DIR = os.path.expanduser(os.getenv("BLUR_STATE_DIR", os.path.join(BLUR_HOME, "state")))

# ensure dirs exist (no manual mkdir needed)
Path(STATE_DIR).mkdir(parents=True, exist_ok=True)
Path(SESSIONS_DIR).mkdir(parents=True, exist_ok=True)

# --- Bundle/package detection + path resolvers ---
def _is_packaged() -> bool:
    try:
        # afterPack writes a marker file into Resources
        marker = Path(__file__).resolve().parents[1] / "BLUR_PACKAGED"
        if marker.exists():
            return True
    except Exception:
        pass
    return os.getenv("BLUR_PACKAGED", "0").lower() in ("1", "true", "yes", "y")

def RESOURCES_DIR() -> Path:
    # this file lives in .../Contents/Resources/core/  → parent is Resources
    return Path(__file__).resolve().parents[1]

def _env_first(*keys: str) -> Optional[str]:
    for k in keys:
        v = os.getenv(k)
        if v and str(v).strip():
            return v
    return None

def resolve_config_path() -> Path:
    # 1) env overrides (support all common names)
    envp = _env_first("BLUR_CONFIG", "CONFIG_PATH", "BLUR_CONFIG_PATH")
    if envp:
        p = Path(os.path.expanduser(os.path.expandvars(envp)))
        if p.exists():
            return p
    # 2) bundled default
    for name in ("config.yaml", "config.yml"):
        q = RESOURCES_DIR() / name
        if q.exists():
            return q
    # 3) legacy home fallback
    home_cfg = Path(BLUR_HOME).expanduser() / "config.yaml"
    if home_cfg.exists():
        return home_cfg
    raise FileNotFoundError(f"config not found (tried env, Resources, {home_cfg})")

# compute manifest path and config dir early
try:
    MANIFEST_PATH = str(resolve_config_path())
except Exception as e:
    # still allow server to start; populate minimal placeholders
    MANIFEST_PATH = os.path.join(BLUR_HOME, "config.yaml")
CONFIG_DIR = Path(MANIFEST_PATH).parent

def resolve_models_dir(config_dir: Optional[Path] = None) -> Path:
    # 1) env
    envp = _env_first("BLUR_MODELS", "MODELS_DIR")
    if envp:
        p = Path(os.path.expanduser(os.path.expandvars(envp)))
        if p.exists():
            return p
    # 2) alongside config
    if config_dir:
        q = Path(config_dir) / "models"
        if q.exists():
            return q
    # 3) bundled Resources/models
    r = RESOURCES_DIR() / "models"
    if r.exists():
        return r
    # 4) legacy home
    return Path(BLUR_HOME).expanduser() / "models"

manifest: Dict[str, Any] = {}
homes: Dict[str, str] = {}
sessions: Dict[str, Dict[str, Any]] = {}
llm_vessels: Dict[str, Llama] = {}
persistent_rag: Optional['PersistentRAG'] = None
user_memory_chunks: Dict[str, List[Dict[str, Any]]] = {}
user_memory_indexes: Dict[str, faiss.Index] = {}

tone_tags_db: Dict[str, List[Dict[str, Any]]] = {}  # {session_id: [{turn, tone, text, ts}]}
ache_metrics_db: Dict[str, List[Dict[str, Any]]] = {}  # {session_id: [{turn, ache_score, ...}]}
tone_tags_lock = threading.Lock()
ache_metrics_lock = threading.Lock()

# ---------- Load manifest early ----------
try:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as _cf:
        manifest = yaml.safe_load(_cf) or {}
    log.info(f"[CFG] manifest: {MANIFEST_PATH}")
except Exception as e:
    log.error(f"🛑 Failed to load manifest: {e}")
    manifest = {}

# Make these available to templating/interpolation
homes.setdefault("resources", str(RESOURCES_DIR()))
homes.setdefault("config_dir", str(CONFIG_DIR))
homes.setdefault("models", str(resolve_models_dir(CONFIG_DIR)))
homes.setdefault("blur_home", BLUR_HOME)

# ---------- TONE DETECTION SYSTEM ----------
_TONE_KEYWORDS = {
    "playful": ["lol", "haha", "😂", "fun", "play", "goofy", "silly", "yo", "dope", "vibe"],
    "serious": ["important", "critical", "urgent", "must", "need", "serious", "real talk"],
    "protective": ["safe", "protect", "careful", "watch", "guard", "shield", "okay?"],
    "tender": ["gentle", "soft", "care", "love", "warm", "hold", "embrace", "💜", "🪷"],
    "giddy": ["excited", "omg", "wow", "amazing", "awesome", "yes!", "✨", "🎉"],
    "flat": ["meh", "whatever", "idk", "dunno", "sure", "fine", "okay"]
}

def _detect_tone(text: str) -> str:
    """Detect primary tone from text using keyword matching."""
    if not text or len(text.strip()) < 5:
        return "neutral"
    tracked = get_cfg("memory.tone_tags.track", []) or []
    if not tracked:
        return "neutral"
    text_lower = text.lower()
    scores = {}
    for tone in tracked:
        keywords = _TONE_KEYWORDS.get(tone, [])
        if not keywords:
            continue
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > 0:
            scores[tone] = score
    if not scores:
        return "neutral"
    return max(scores.items(), key=lambda x: x[1])[0]

# ---------- ACHE SCORE ESTIMATION ----------
_ACHE_INDICATORS = {
    "high": ["hurt", "pain", "ache", "stuck", "lost", "can't", "help me", "scared", "alone", "empty"],
    "medium": ["confused", "unsure", "worried", "anxious", "stressed", "tired", "difficult"],
    "low": ["okay", "fine", "good", "better", "thanks", "appreciate", "helpful"]
}

def _estimate_ache_score(user_text: str, response_text: str) -> Dict[str, Any]:
    """Estimate ache metrics from conversation turn."""
    if not user_text:
        return {"ache_score": 0.0, "final_ache": 0.0, "healing": 0.0, "expansion": 0.0}
    user_lower = user_text.lower()
    high_count = sum(1 for ind in _ACHE_INDICATORS["high"] if ind in user_lower)
    med_count = sum(1 for ind in _ACHE_INDICATORS["medium"] if ind in user_lower)
    low_count = sum(1 for ind in _ACHE_INDICATORS["low"] if ind in user_lower)
    total = high_count + med_count + low_count
    if total == 0:
        ache_score = 0.3
    else:
        ache_score = ((high_count * 1.0) + (med_count * 0.5) + (low_count * 0.0)) / total
    resp_lower = (response_text or "").lower()
    healing_words = ["breathe", "step", "try", "together", "safe", "okay", "here"]
    healing_count = sum(1 for w in healing_words if w in resp_lower)
    healing = min(1.0, healing_count * 0.15)
    expansion = min(1.0, len(response_text) / 500.0) if response_text else 0.0
    final_ache = max(0.0, ache_score - healing)
    flip_detected = (ache_score - final_ache) > 0.3
    return {
        "ache_score": round(ache_score, 3),
        "final_ache": round(final_ache, 3),
        "healing": round(healing, 3),
        "expansion": round(expansion, 3),
        "flip_detected": flip_detected,
        "delta": round(ache_score - final_ache, 3)
    }

# ---------- TONE TAGS DATABASE ----------
def _tone_tags_file() -> str:
    path = get_cfg("memory.tone_tags.persist_path", "")
    if path:
        return resolve_path(path, homes)
    return os.path.join(STATE_DIR, "tone_tags.json")

def load_tone_tags():
    global tone_tags_db
    try:
        p = _tone_tags_file()
        if os.path.exists(p):
            with open(p, "r") as f:
                tone_tags_db = json.load(f) or {}
        log.info(f"✅ Tone tags loaded: {len(tone_tags_db)} sessions")
    except Exception as e:
        log.error(f"tone tags load fail: {e}")
        tone_tags_db = {}

def save_tone_tags():
    try:
        p = _tone_tags_file()
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(tone_tags_db, f)
    except Exception as e:
        log.error(f"tone tags save fail: {e}")

def record_tone_tag(session_id: str, turn: int, tone: str, text: str):
    if not session_id or not tone:
        return
    with tone_tags_lock:
        if session_id not in tone_tags_db:
            tone_tags_db[session_id] = []
        tone_tags_db[session_id].append({
            "turn": turn,
            "tone": tone,
            "text": text[:200],
            "ts": int(time.time())
        })
        tone_tags_db[session_id] = tone_tags_db[session_id][-50:]

def get_recent_tone(session_id: str, window: int = 5) -> Optional[str]:
    if session_id not in tone_tags_db:
        return None
    recent = tone_tags_db[session_id][-window:]
    if not recent:
        return None
    from collections import Counter
    tones = [t["tone"] for t in recent if t.get("tone")]
    if not tones:
        return None
    most_common = Counter(tones).most_common(1)
    return most_common[0][0] if most_common else None

# ---------- ACHE METRICS DATABASE ----------
def _ache_metrics_file() -> str:
    path = get_cfg("memory.ache_metrics.persist_path", "") or get_cfg("philosophy.witness.metrics_persist_path", "")
    if path:
        return resolve_path(path, homes)
    return os.path.join(STATE_DIR, "ache_metrics.json")

def load_ache_metrics():
    global ache_metrics_db
    try:
        p = _ache_metrics_file()
        if os.path.exists(p):
            with open(p, "r") as f:
                ache_metrics_db = json.load(f) or {}
        log.info(f"✅ Ache metrics loaded: {len(ache_metrics_db)} sessions")
    except Exception as e:
        log.error(f"ache metrics load fail: {e}")
        ache_metrics_db = {}

def save_ache_metrics():
    try:
        p = _ache_metrics_file()
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(ache_metrics_db, f)
    except Exception as e:
        log.error(f"ache metrics save fail: {e}")

def record_ache_metrics(session_id: str, turn: int, metrics: Dict[str, Any]):
    if not session_id or not metrics:
        return
    with ache_metrics_lock:
        if session_id not in ache_metrics_db:
            ache_metrics_db[session_id] = []
        record = {"turn": turn, "ts": int(time.time())}
        record.update(metrics)
        ache_metrics_db[session_id].append(record)
        ache_metrics_db[session_id] = ache_metrics_db[session_id][-100:]

def get_ache_trend(session_id: str, window: int = 5) -> str:
    if session_id not in ache_metrics_db:
        return "stable"
    recent = ache_metrics_db[session_id][-window:]
    if len(recent) < 2:
        return "stable"
    first_ache = recent[0].get("ache_score", 0.5)
    last_ache = recent[-1].get("ache_score", 0.5)
    delta = last_ache - first_ache
    if delta < -0.2:
        return "improving"
    elif delta > 0.2:
        return "worsening"
    else:
        return "stable"

CORE_IS_READY = False
TTFT_LAST: Optional[float] = None

# ---------- Per-session locks ----------
sessions_lock = threading.Lock()
user_memory_lock = threading.Lock()
_session_locks: Dict[str, asyncio.Lock] = {}
_session_locks_lock = threading.Lock()
_embed_lock = asyncio.Lock()
_VESSEL_LOCK = asyncio.Lock()
_recent_qv_cache_lock = threading.Lock()

def _get_session_lock(session_id: str) -> asyncio.Lock:
    with _session_locks_lock:
        if session_id not in _session_locks:
            _session_locks[session_id] = asyncio.Lock()
        return _session_locks[session_id]

# Ephemeral embedding cache
_EMBED_LLM: Optional[Llama] = None
_EMBED_DIM: Optional[int] = None
_MEMVEC: Dict[str, np.ndarray] = {}
_RECENT_QV_CACHE: Dict[str, np.ndarray] = {}

# ---------- Helpers: config ----------
def _safe_load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_cfg(path: str, default=None):
    node = manifest
    for key in path.split('.'):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node

def resolve_path(s: Any, homes_dict: Dict[str, str]) -> Any:
    if not isinstance(s, str):
        return s
    out = s
    for _ in range(8):
        prev = out
        # meta.homes.* placeholders
        for k, v in (homes_dict or {}).items():
            out = out.replace(f"${{meta.homes.{k}}}", str(v))
        # explicit tokens
        out = out.replace("${BLUR_HOME}", BLUR_HOME)
        out = out.replace("${BLUR_MODELS}", str(resolve_models_dir(CONFIG_DIR)))
        out = out.replace("${RESOURCES_DIR}", str(RESOURCES_DIR()))
        out = out.replace("${CONFIG_DIR}", str(CONFIG_DIR))
        # expand env + ~
        out = os.path.expandvars(os.path.expanduser(out))
        if out == prev:
            break
    # If still relative, make it relative to the config file dir
    p = Path(out)
    if not p.is_absolute():
        p = (CONFIG_DIR / p).resolve()
    return str(p)

def resolve_homes_recursive(h: dict) -> dict:
    return {k: resolve_path(v, h) for k, v in (h or {}).items()}

# Apply any meta.homes overrides from config
homes.update(resolve_homes_recursive(get_cfg("meta.homes", {})))
homes.setdefault("blur_home", BLUR_HOME)

# ---------- LLM + Embedding ----------
def _safe_gpu_layers(req: Optional[int]) -> int:
    if str(os.getenv("BLUR_FORCE_CPU","0")).lower() in ("1","true","yes"):
        return 0
    if req is None or req < 0:
        return int(get_cfg("engines.llama_cpp.n_gpu_layers", 4) or 4)
    return max(0, int(req))

def _ensure_embedder():
    global _EMBED_LLM
    if _EMBED_LLM is not None:
        return
    if Llama is None:
        raise RuntimeError("llama_cpp unavailable on this build")

    embed_key = get_cfg("memory.vector_store.embed_model", "snowflake_arctic_embed")
    m = (manifest.get("models", {}) or {}).get(embed_key, {}) if manifest else {}
    raw = (m or {}).get("path", "")

    candidates: List[Path] = []

    # 1) explicit path (interpolated)
    if raw:
        candidates.append(Path(resolve_path(raw, homes)))

    # 2) raw relative to CONFIG_DIR
    if raw:
        candidates.append((CONFIG_DIR / raw))

    # 3) common filenames in known model roots
    common = [
        "snowflake-arctic-embed-m-Q4_K_M.gguf",
        "snowflake-arctic-embed-m.Q4_K_M.gguf",
        "snowflake-arctic-embed-m.gguf",
    ]
    for base in (resolve_models_dir(CONFIG_DIR), RESOURCES_DIR() / "models", Path(BLUR_HOME) / "models"):
        for name in ([raw] if raw else []) + common:
            candidates.append(Path(base) / name)

    mpath = next((c for c in candidates if c and c.exists()), None)
    if not mpath:
        tried = "\n  - " + "\n  - ".join(str(c) for c in candidates[:12])
        raise RuntimeError(f"Embed model missing: {embed_key}; tried:{tried}")

    _EMBED_LLM = Llama(
        model_path=str(mpath),
        embedding=True,
        n_ctx=512,
        n_batch=int(get_cfg("engines.llama_cpp.n_batch", 2048) or 2048),
        n_gpu_layers=_safe_gpu_layers(get_cfg("engines.llama_cpp.n_gpu_layers", -1)),
        n_threads=max(2, os.cpu_count() or 4),
        use_mmap=True,
        logits_all=False,
        verbose=False
    )
    log.info(f"✅ Embedder online: {mpath.name}")

def _embedding_dim() -> int:
    global _EMBED_DIM
    if _EMBED_DIM is None:
        _ensure_embedder()
    if _EMBED_LLM is None:
        return 0
    _EMBED_DIM = len(_EMBED_LLM.create_embedding(input=["dim?"])["data"][0]["embedding"])
    return _EMBED_DIM

def _check_memory_pressure() -> bool:
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.available < 500 * 1024 * 1024 or mem.percent > 90
    except ImportError:
        return False

def _encode(texts: List[str]) -> np.ndarray:
    _ensure_embedder()
    if _EMBED_LLM is None:
        raise RuntimeError("Embedder not available for encoding.")
    if isinstance(texts, str):
        texts = [texts]
    if len(texts) > 100 and _check_memory_pressure():
        log.warning(f"[MEMORY] High pressure, splitting {len(texts)} texts into batches")
        results = []
        for i in range(0, len(texts), 50):
            batch = texts[i:i+50]
            out = _EMBED_LLM.create_embedding(input=batch)["data"]
            arr = np.asarray([d["embedding"] for d in out], dtype="float32")
            results.append(arr)
        arr = np.vstack(results)
    else:
        out = _EMBED_LLM.create_embedding(input=texts)["data"]
        arr = np.asarray([d["embedding"] for d in out], dtype="float32")
    faiss.normalize_L2(arr)
    return arr

def _load_llama_with_backoff(model_path: str, requested_ctx: int, n_gpu_layers: int, n_batch: int) -> Llama:
    for ctx in sorted({requested_ctx, 8192, 4096, 2048, 1024}, reverse=True):
        if ctx > requested_ctx:
            continue
        try:
            llm = Llama(model_path=model_path, n_ctx=ctx, n_gpu_layers=n_gpu_layers,
                        n_batch=n_batch, n_threads=max(2, os.cpu_count() or 4),
                        use_mmap=True, logits_all=False, verbose=False)
            log.info(f"[models] Loaded {os.path.basename(model_path)} n_ctx={ctx} n_gpu_layers={n_gpu_layers} n_batch={n_batch}")
            return llm
        except Exception as e:
            log.warning(f"load fail n_ctx={ctx}: {e}")
    raise RuntimeError(f"cannot load model at {model_path}")

def load_llm_from_config(model_key: str) -> bool:
    if model_key in llm_vessels:
        return True
    cfg = (manifest.get("models", {}) or {}).get(model_key, {}) if manifest else {}
    if not cfg or cfg.get("engine") != "llama_cpp":
        log.error(f"model '{model_key}' not llama_cpp")
        return False

    raw = cfg.get("path", "")
    candidates: List[Path] = []

    # 1) explicit path (interpolated)
    if raw:
        candidates.append(Path(resolve_path(raw, homes)))
    # 2) raw relative to CONFIG_DIR
    if raw:
        candidates.append((CONFIG_DIR / raw))
    # 3) guesses across known model roots
    guesses = [
        f"{model_key}.gguf",
        "qwen-3-4b-instruct-q4_k_m.gguf",
        "qwen3-4b-instruct-q4_k_m.gguf",
    ]
    for base in (resolve_models_dir(CONFIG_DIR), RESOURCES_DIR() / "models", Path(BLUR_HOME) / "models"):
        for name in ([raw] if raw else []) + guesses:
            candidates.append(Path(base) / name)

    path = next((c for c in candidates if c and c.exists()), None)
    if not path:
        log.error("model path missing for '%s'; tried:\n  - %s", model_key, "\n  - ".join(str(c) for c in candidates[:12]))
        return False

    llm_vessels[model_key] = _load_llama_with_backoff(
        str(path),
        int(get_cfg("engines.llama_cpp.python_n_ctx", 8192) or 8192),
        _safe_gpu_layers(get_cfg("engines.llama_cpp.n_gpu_layers", -1)),
        int(get_cfg("engines.llama_cpp.n_batch", 512) or 512),
    )
    return True

def _llama_accepts_kw(llm: Llama, kw: str) -> bool:
    try:
        return kw in inspect.signature(llm.create_chat_completion).parameters
    except Exception:
        return False

def _prune_params(llm: Llama, params: dict) -> dict:
    # Keep only kwargs that the current llama_cpp build supports.
    return {k: v for k, v in params.items() if _llama_accepts_kw(llm, k)}

# ---------- RAG ----------
def _memvec_get(text: str) -> np.ndarray:
    key = hashlib.sha1(text.encode("utf-8","ignore")).hexdigest()
    if key in _MEMVEC: return _MEMVEC[key]
    if len(_MEMVEC) > 2048: _MEMVEC.pop(next(iter(_MEMVEC)))
    v = _encode([text])[0]
    _MEMVEC[key] = v
    return v

def _query_vec_cached(query: str, sid: Optional[str], tid: Optional[str]) -> np.ndarray:
    key = f"{sid}|{tid}|{hashlib.sha1((query or '').encode()).hexdigest()}"
    with _recent_qv_cache_lock:
        if key in _RECENT_QV_CACHE: return _RECENT_QV_CACHE[key]
    v = _encode([query])[0]
    with _recent_qv_cache_lock:
        if len(_RECENT_QV_CACHE)>128: _RECENT_QV_CACHE.pop(next(iter(_RECENT_QV_CACHE)))
        _RECENT_QV_CACHE[key]=v
    return v

_CONTENT_KEYS = ("content","text","chunk","body","data","payload","summary","desc","description","title")
def _extract_content(row: Dict[str, Any]) -> str:
    for k in _CONTENT_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict):
            for kk in ("content","text","body","value"):
                vv = v.get(kk)
                if isinstance(vv, str) and vv.strip():
                    return vv
    return ""

class PersistentRAG:
    def __init__(self, index_path: str, chunks_path: str, ttl_days: int=0, auto_compact: bool=False):
        self.index_path = Path(index_path); self.chunks_path = Path(chunks_path)
        self.ttl_days=int(ttl_days or 0); self.auto_compact=bool(auto_compact)
        self.lock = threading.Lock()
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict[str,Any]] = []

    def _read_jsonl(self) -> List[Dict[str,Any]]:
        if not self.chunks_path.exists(): return []
        rows=[]
        with self.chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try: rows.append(json.loads(line))
                except Exception:
                    m=re.search(r"\{.*\}", line)
                    if m:
                        try: rows.append(json.loads(m.group(0)))
                        except Exception: pass
        return rows

    def _write_jsonl(self, rows: List[Dict[str,Any]]):
        self.chunks_path.parent.mkdir(parents=True, exist_ok=True)
        with self.chunks_path.open("w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")

    def _compact_by_ttl(self, rows):
        if not self.ttl_days or self.ttl_days<=0: return rows
        cutoff=time.time()-self.ttl_days*86400
        kept=[r for r in rows if int(r.get("ts",0))>=cutoff]
        return kept or rows

    def _ensure_ids(self, rows):
        if any("id" not in r for r in rows):
            for i,r in enumerate(rows): r["id"]=i
            self._write_jsonl(rows)

    def _new_flat(self) -> faiss.Index:
        dim=_embedding_dim()
        if dim == 0:
            raise RuntimeError("Embedder dimension is 0. Cannot create FAISS index.")
        return faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def load(self):
        with self.lock:
            rows=self._read_jsonl()
            if self.auto_compact: rows=self._compact_by_ttl(rows)
            self._ensure_ids(rows)

            usable=[]
            for r in rows:
                c=_extract_content(r)
                if c.strip():
                    r["_resolved_content"]=c
                    usable.append(r)
            self.chunks=usable

            if _EMBED_LLM is None:
                log.error("[RAG] Cannot load index or chunks: Embedder is not online.")
                self.index = None
                return

            reused=False
            try:
                if self.index_path.exists():
                    idx=faiss.read_index(str(self.index_path))
                    if getattr(idx, 'd', -1)==_embedding_dim():
                        self.index=idx; reused=True
            except Exception as e:
                log.warning(f"[RAG] index read failed: {e}; rebuilding")
                self.index=None

            if not reused:
                try:
                    self.index=self._new_flat()
                except RuntimeError as e:
                    log.error(f"[RAG] Failed to create new index: {e}")
                    self.index = None
                    return

            if self.chunks and self.index:
                vecs=_encode([r["_resolved_content"] for r in self.chunks])
                ids=np.asarray([int(r["id"]) for r in self.chunks], dtype="int64")
                if reused and getattr(self.index, 'ntotal', 0)!=len(ids):
                    log.info("[RAG] index count mismatch → rebuilding")
                    self.index=self._new_flat()
                if getattr(self.index, 'ntotal', 0)==0:
                    self.index.add_with_ids(vecs, ids)
                else:
                    self.index=self._new_flat()
                    self.index.add_with_ids(vecs, ids)

            if self.index:
                self.index_path.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self.index, str(self.index_path))
            log.info(f"✅ RAG ready: nchunks={len(self.chunks)} ntotal={getattr(self.index,'ntotal',0)}")

    def search_vec(self, qv: np.ndarray, top_k: int=5) -> List[Dict[str,Any]]:
        with self.lock:
            if not (self.index and self.index.ntotal>0 and self.chunks): return []
            D,I=self.index.search(qv.reshape(1,-1).astype("float32"), min(top_k, self.index.ntotal))
            by_id={int(r["id"]): r for r in self.chunks}
            out=[]
            for i in I[0]:
                ii=int(i)
                if ii in by_id:
                    r=dict(by_id[ii])
                    r.setdefault("_resolved_content", _extract_content(r))
                    out.append(r)
            return out

# ---------- User memory ----------
def _user_mem_file() -> str: return os.path.join(STATE_DIR, "user_memory.json")
def load_user_memory():
    global user_memory_chunks, user_memory_indexes
    try:
        p=_user_mem_file()
        if os.path.exists(p):
            with open(p,"r") as f: user_memory_chunks=json.load(f) or {}
        # Only proceed if embedder is available
        if _EMBED_LLM is None:
            log.warning("[User Memory] Skipping vector indexing: Embedder is not online.")
            return

        for uname, rows in list(user_memory_chunks.items()):
            norm=[]
            for r in rows or []:
                if isinstance(r,str): r={"content":r,"ts":int(time.time())}
                r.setdefault("content",""); r.setdefault("ts",int(time.time()))
                r["vec"] = _memvec_get(r["content"])
                norm.append(r)
            user_memory_chunks[uname]=norm[-int(get_cfg("memory.user_memory.max_chunks",50) or 50):]
            if norm:
                dim=_embedding_dim()
                if dim > 0:
                    idx=faiss.IndexFlatIP(dim)
                    mat=np.vstack([r["vec"] for r in norm]).astype("float32")
                    idx.add(mat); user_memory_indexes[uname]=idx
                else:
                    log.warning(f"[User Memory] Index for {uname} skipped: Embed dim is 0.")
        log.info(f"✅ User memory for {len(user_memory_chunks)} users")
    except Exception as e:
        log.error(f"user memory load fail: {e}")
        user_memory_chunks={}; user_memory_indexes={}

def save_user_memory():
    try:
        with open(_user_mem_file(),"w") as f: json.dump(user_memory_chunks,f)
    except Exception as e:
        log.error(f"user memory save fail: {e}")

def upsert_user_memory(user: str, text: str):
    if not user or not text.strip(): return
    if _EMBED_LLM is None:
        log.error("[User Memory] Cannot upsert: Embedder is not online.")
        return
    
    row={"content":text.strip(),"ts":int(time.time())}
    try:
        row["vec"]=_memvec_get(row["content"])
    except RuntimeError as e:
        log.error(f"[User Memory] Encoding failed: {e}")
        return # Skip upsert if encoding fails

    with user_memory_lock:
        lst=user_memory_chunks.setdefault(user,[])
        lst.append(row)
        ttl_days=int(get_cfg("memory.user_memory.ttl_days",90) or 90)
        if ttl_days>0:
            cutoff=int(time.time())-ttl_days*86400
            lst[:]=[r for r in lst if int(r.get("ts",0))>=cutoff]
        user_memory_chunks[user]=lst[-int(get_cfg("memory.user_memory.max_chunks",50) or 50):]
        # rebuild simple index
        dim=_embedding_dim()
        if dim > 0:
            idx=faiss.IndexFlatIP(dim)
            mat=np.vstack([r["vec"] for r in user_memory_chunks[user]]).astype("float32")
            idx.add(mat); user_memory_indexes[user]=idx

def retrieve_user_memory(username: Optional[str], query: str, top_k: int=3) -> List[str]:
    if not username: return []
    if _EMBED_LLM is None: return [] # Cannot retrieve without embedder
    try:
        idx=user_memory_indexes.get(username); rows=user_memory_chunks.get(username,[])
        if not (idx and rows): return []
        qv=_encode([query])[0].reshape(1,-1).astype("float32")
        k=min(int(top_k), idx.ntotal); D,I=idx.search(qv,k)
        out=[]
        for i,dist in zip(I[0], D[0]):
            if i>=0 and float(dist)>0.3:
                out.append(rows[int(i)]["content"])
        return out
    except Exception as e:
        log.error(f"user mem retrieval fail: {e}"); return []

# ---------- Language detection (light) ----------
try:
    from langdetect import detect as _ld_detect, DetectorFactory as _LDFactory
    _LDFactory.seed=42; _HAS_LD=True
except Exception:
    _HAS_LD=False

_LANG_CODE_TO_NAME = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", 
    "it": "Italian", "pt": "Portuguese", "ja": "Japanese", "ko": "Korean", 
    "zh-cn": "Chinese", "zh-tw": "Chinese (Traditional)", "ru": "Russian",
    "ar": "Arabic", "hi": "Hindi", "th": "Thai", "vi": "Vietnamese",
    "nl": "Dutch", "pl": "Polish", "tr": "Turkish", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "no": "Norwegian", "cs": "Czech",
    "el": "Greek", "he": "Hebrew", "id": "Indonesian", "ms": "Malay",
}

# Quick patterns for common languages (before heavy langdetect)
_QUICK_PATTERNS = [
    # CJK scripts
    (re.compile(r'[\u4e00-\u9fff]'), "Chinese"),           # Chinese characters
    (re.compile(r'[\u3040-\u309f\u30a0-\u30ff]'), "Japanese"),  # Hiragana/Katakana
    (re.compile(r'[\uac00-\ud7af]'), "Korean"),             # Hangul
    
    # Other scripts
    (re.compile(r'[\u0400-\u04ff]'), "Russian"),            # Cyrillic
    (re.compile(r'[\u0600-\u06ff]'), "Arabic"),             # Arabic
    (re.compile(r'[\u0e00-\u0e7f]'), "Thai"),               # Thai
    (re.compile(r'[\u0590-\u05ff]'), "Hebrew"),             # Hebrew
    (re.compile(r'[\u0900-\u097f]'), "Hindi"),              # Devanagari
    (re.compile(r'[\u0370-\u03ff]'), "Greek"),
]

_ENGLISH_MARKERS = re.compile(
    r'^(yo+|ye+|ya+|sup|wass?up|ayy+|hey+|hiya|nah|yep|yup|ok|thx|thanks|pls|plz)\b',
    re.IGNORECASE
)

def detect_lang_name(text: str) -> str:
    default = get_cfg("language.default_if_unknown", "English")
    s = (text or "").strip()
    if not s or len(s) < 2: return default
    if _ENGLISH_MARKERS.match(s): return "English"
    for pattern, lang in _QUICK_PATTERNS:
        if pattern.search(s): return lang
    if _HAS_LD and len(s) >= 10:
        try:
            code = _ld_detect(s); code = "zh-cn" if code == "zh" else code
            detected = _LANG_CODE_TO_NAME.get(code, default)
            ascii_ratio = sum(1 for c in s if ord(c) < 128) / len(s)
            if ascii_ratio > 0.9 and detected not in ("English","Spanish","French","German","Italian","Portuguese","Dutch","Swedish","Danish","Norwegian"):
                return "English"
            return detected
        except Exception:
            pass
    return default

def detect_and_set_language(session: dict, user_text: str, force_instant: bool = True) -> str:
    default = get_cfg("language.default_if_unknown", "English")
    detected = detect_lang_name(user_text)
    prev_lang = session.get("active_lang", default)
    if detected != prev_lang:
        log.info(f"[LANG] Detected: {detected} (was: {prev_lang}) from: '{user_text[:50]}...'")
    if force_instant:
        session["active_lang"] = detected
        session["last_seen_lang"] = detected
        session["lang_streak"] = 0
        return detected
    n = int(get_cfg("language.hysteresis_consecutive", 2) or 2)
    active = session.get("active_lang", default)
    last = session.get("last_seen_lang", detected)
    streak = int(session.get("lang_streak", 0))
    if detected == active:
        session["last_seen_lang"] = detected
        session["lang_streak"] = 0
        return active
    streak = streak + 1 if detected == last else 1
    session["last_seen_lang"] = detected
    session["lang_streak"] = streak
    if streak >= n:
        session["active_lang"] = detected
        session["lang_streak"] = 0
        return detected
    return active

def detect_lang_with_confidence(text: str) -> Tuple[str, float]:
    default = get_cfg("language.default_if_unknown", "English")
    s = (text or "").strip()
    if not s or len(s) < 2: return default, 0.5
    if _ENGLISH_MARKERS.match(s): return "English", 0.95
    for pattern, lang in _QUICK_PATTERNS:
        if pattern.search(s): return lang, 0.98
    if _HAS_LD and len(s) >= 10:
        try:
            from langdetect import detect_langs
            results = detect_langs(s)
            if results:
                top = results[0]; code = "zh-cn" if top.lang == "zh" else top.lang
                detected = _LANG_CODE_TO_NAME.get(code, default)
                confidence = float(top.prob)
                ascii_ratio = sum(1 for c in s if ord(c) < 128) / len(s)
                if ascii_ratio > 0.9 and detected not in ("English","Spanish","French","German","Italian","Portuguese"):
                    return "English", 0.8
                return detected, confidence
        except Exception:
            pass
    return default, 0.3

# ---------- ASTRO blocklist (soft) ----------
def _rag_text_allowed_for_mode(text: str, mode: str) -> bool:
    if (mode or "").lower() != "astrofuck": return True
    bl = (get_cfg("rag.blocklist_words.astrofuck", []) or [])
    if not bl: return True
    t = (text or "").lower()
    hits = sum(1 for w in bl if w and w.lower() in t)
    return hits <= 1

# ---------- Context retrieval ----------
def retrieve_context_blend(query: str, session_id: Optional[str], thread_id: Optional[str], 
                           username: Optional[str], top_k: int = 8, mode: str = "astrofuck") -> str:
    parts = []
    if persistent_rag and _EMBED_LLM:
        try:
            qv = _query_vec_cached(query, session_id, thread_id)
            rows = persistent_rag.search_vec(qv, top_k=min(5, top_k))
        except Exception as e:
            log.error(f"[RAG] search failed: {e}, continuing without RAG")
            rows = []
        if rows:
            filt = []
            blocklist = (get_cfg(f"rag.blocklist_words.{mode.lower()}", []) or [])
            allowlist = (get_cfg(f"rag.allowlist_words.{mode.lower()}", []) or [])
            for r in rows:
                txt = (r.get("_resolved_content") or r.get("content", "") or "")
                if allowlist and any(w and w.lower() in txt.lower() for w in allowlist):
                    filt.append(txt); continue
                if _rag_text_allowed_for_mode(txt, mode):
                    filt.append(txt)
            if filt:
                parts.append("--- Persistent Knowledge ---\n" + "\n\n".join(filt))
    if username and _EMBED_LLM:
        pm = retrieve_user_memory(username, query, top_k=3)
        if pm:
            parts.append("--- Personal Memory Fragments ---\n" + "\n\n".join(pm))
    return "\n\n".join([p for p in parts if p.strip()])

# ---------- Prompt assembly (Persona isolation baked) ----------
def _cap(s: str, n: int) -> str:
    s=(s or "").strip()
    if len(s)<=n: return s
    cut=s[:n]
    return cut.rsplit("\n",1)[0].strip() if "\n" in cut else cut

def _history_for(session: Dict, thread_id: Optional[str], limit: int) -> List[Dict]:
    by=session.setdefault("history_by_thread",{})
    return by.get(thread_id or "__default__", [])[-int(limit):]

def _append_history(session: Dict, thread_id: Optional[str], user: str, assistant: str, mode: str, keep: int):
    by=session.setdefault("history_by_thread",{})
    lst=by.setdefault(thread_id or "__default__",[])
    lst.append({"user":user, "assistant":assistant, "mode":mode})
    by[thread_id or "__default__"]=lst[-int(keep):]

def _maybe_witness_line(session: Dict, turn: int, user_text: str) -> Optional[str]:
    if not bool(get_cfg("philosophy.witness.enabled", True)): return None
    every=int(get_cfg("assembly.include_witness_every_n", 2) or 2)
    if every<=0: return None
    if turn % every != 0: return None
    tpl=get_cfg("philosophy.witness.template", "⟪witness⟫ I’m noticing ${signal}. if i’m off, redirect me.")
    signal="hesitation" if len(user_text.strip())>0 and user_text.strip()[-1] in ".?!" else "open loop"
    return tpl.replace("${signal}", signal)

def _maybe_acheflip_nudge(mode: str) -> Optional[str]:
    if mode.lower()=="astrofuck" and not bool(get_cfg("style.post.acheflip_inject_non_astro", False)):
        return None
    if not bool(get_cfg("philosophy.witness.enabled", True)): return None
    prob=float(get_cfg("philosophy.witness.nudge_probability", 0.0) or 0.0)
    if prob<=0: return None
    import random
    if random.random()>prob: return None
    nudges=get_cfg("philosophy.witness.nudge_templates", []) or []
    if not nudges: return None
    return f"⟪nudge⟫ {random.choice(nudges)}"

def _mode_tone_inject(mode: str) -> str:
    return get_cfg(f"style.mode_tone_inject.{mode.lower()}", "")
    
def _mode_style_contract(mode: str) -> str:
    if mode.lower() == "astrofuck":
        return get_cfg("style.style_contract_astrofuck", "")
    return get_cfg("style.style_contract_dream", "")

def retrieve_tone_memory(session_id: str, current_tone: str, max_lines: int = 1) -> Optional[str]:
    """Retrieve past memory snippets matching current emotional tone."""
    if not session_id or session_id not in tone_tags_db:
        return None

    strategy = get_cfg("memory.recall.strategy", "tone-first")
    template = get_cfg(
        "memory.recall.template",
        'last time this felt ${prev_tone}, you smiled at: "${micro_moment}." want to try that beat again?'
    )
    if strategy != "tone-first":
        return None

    history = tone_tags_db[session_id]
    matches = [h for h in history if h.get("tone") == current_tone]
    if not matches:
        return None

    # Avoid super-recent repetition
    candidates = matches[:-3] if len(matches) > 3 else matches[:-1]
    if not candidates:
        return None

    memory = candidates[-1]
    recall_text = template.replace("${prev_tone}", current_tone)
    recall_text = recall_text.replace("${micro_moment}", (memory.get("text", "") or "")[:100])
    return recall_text if max_lines > 0 else None

def select_witness_nudge(session_id: Optional[str], ache_score: float) -> Optional[str]:
    """Select a gentle nudge line based on ache level and trend."""
    if not bool(get_cfg("philosophy.witness.enabled", True)):
        return None

    import random
    prob = float(get_cfg("philosophy.witness.nudge_probability", 0.7) or 0.7)
    if random.random() > prob:
        return None

    templates = get_cfg("philosophy.witness.nudge_templates", []) or []
    if not templates:
        return None

    trend = get_ache_trend(session_id, window=5) if session_id else "stable"
    if ache_score > 0.6 or trend == "worsening":
        grounding = [t for t in templates if any(w in t.lower() for w in ["breathe", "ground", "step"])]
        if grounding:
            return random.choice(grounding)

    return random.choice(templates)

def _build_system_prompt_with_recall(mode: str, context: str, session: Dict, user_text: str, tone: Optional[str] = None) -> str:
    # START WITH IDENTITY, not rules
    identity = {
        "astrofuck": "You are Blur: a sharp-tongued logician who cares enough to be brutally kind. Mix street-smart slang with academic rigor. When you see bullshit, call it - then explain why.",
        "dream": "You are Blur: a holistic professor-therapist who finds structure in feelings. Be warm yet incisive. Ground abstract pain in concrete steps."
    }.get(mode.lower(), "You are Blur.")
    
    parts = [identity]
    
    # Add context if available
    if context:
        parts.append("Relevant context:\n" + context)
    
    # Add memory recall if tone matches
    if tone and tone != "neutral":
        recall = retrieve_tone_memory(session.get("id"), tone, max_lines=1)
        if recall:
            parts.append("Memory echo: " + recall)
    
    # Add ONE clear mode instruction
    mode_guide = {
        "astrofuck": "Speak with stylish precision. Use slang naturally. Be a compassionate smartass.",
        "dream": "Be warm and grounded. Mix gentle presence with truth-telling."
    }.get(mode.lower(), "")
    
    if mode_guide:
        parts.append(mode_guide)
    
    return "\n\n".join(parts)

def _build_messages(lang: str, sys_prompt: str, hist: List[Dict], user_text: str) -> List[Dict]:
    msgs: List[Dict[str, str]] = []
    if lang and lang.lower() not in ("english", "en"):
        msgs.append({"role": "system", "content": f"IMPORTANT: Answer entirely in {lang}. No English."})
    msgs.append({"role": "system", "content": sys_prompt})
    for t in (hist or []):
        if t.get("user"):
            msgs.append({"role": "user", "content": t["user"]})
        if t.get("assistant"):
            msgs.append({"role": "assistant", "content": t["assistant"]})
    if user_text:
        msgs.append({"role": "user", "content": user_text})
    return msgs

# ---------- Output filtering / post-processing ----------
_BANNER_PATTERNS = [
    r"^\s*[⇌➤➡➔]\s*Thread:.*$", 
    r"^\s*[⇌➤➡➔]\s*Myth:.*$",
    r"^\s*[⇌➤➡➔]\s*Blessings:.*$",
    r"^\s*GNA CORE ONLINE.*$",
    r"^\s*\[StemPortal.*$",
]
_banner_re = re.compile("|".join(_BANNER_PATTERNS), re.IGNORECASE)

_GLYPHS = set("↺🜂☾⊙🜫⟁༄∃∞ø🜃🜉🜁🜊🜔☿🜠⛧↯⟴𓃰☥✦𓂀∴∵Σ∇△αΩ∿")
_EMOJI_RANGES = [
    (0x1F300, 0x1FAFF),
    (0x1F1E6, 0x1F1FF),
    (0x1F680, 0x1F6FF),
    (0x1F900, 0x1F9FF),
    (0x1FA70, 0x1FAFF),
    (0x2600,  0x26FF),
    (0x2700,  0x27BF),
    (0x1F100, 0x1F1FF),
    (0xFE00,  0xFE0F),
]
_ZWJ = 0x200D

def _is_emoji_char(ch: str) -> bool:
    cp = ord(ch)
    if cp == _ZWJ: return True
    for a, b in _EMOJI_RANGES:
        if a <= cp <= b: return True
    return False

def _allowed_emojis() -> set:
    return set(get_cfg("style.post.allowed_emojis", ["👾","🌐","🪷","🔮"]) or [])

def _strip_banners(text: str) -> str:
    lines = text.splitlines()
    kept = [ln for ln in lines if not _banner_re.search(ln)]
    return "\n".join(kept).strip()

def _strip_emoji_except_glyphs(text: str) -> str:
    if not bool(get_cfg("style.post.strip_emoji", False)): return text
    if not bool(get_cfg("style.post.strip_emoji_except_glyphs", False)): return text
    if bool(get_cfg("blur.keep_emoji", False)): return text
    allowed = _GLYPHS; white = _allowed_emojis(); out = []
    for ch in text:
        if ch in allowed or ch in white or not _is_emoji_char(ch):
            out.append(ch)
        elif ord(ch) in (0x200D, 0xFE0F):
            continue
    return "".join(out)

def _ensure_slang_if_astro(text: str, mode: str) -> str:
    if mode!="astrofuck": return text
    if not bool(get_cfg("style.post.ensure_slang", False)): return text
    lex=(get_cfg("variety.slang_lexicon.words",[]) or [])
    prob=float(get_cfg("variety.slang_lexicon.probability",0.0) or 0.0)
    import random
    if lex and random.random()<prob and not any(w in text.lower() for w in lex):
        text = text + "\n\n" + random.choice(lex)
    return text

def _persona_endings(text: str) -> str:
    if not bool(get_cfg("style.post.persona_endings", False)): return text
    banned=(get_cfg("variety.banned_endings",[]) or [])
    if any(text.strip().endswith(b) for b in banned):
        text=text.rstrip(".!") + "."
    return text

def _get_ngrams(text: str, n: int) -> set:
    words = text.lower().split()
    if len(words) < n: return set()
    return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

def _detect_echo(new_text: str, history: List[Dict], max_ngram: int = 4, window: int = 6) -> bool:
    if not history or not new_text.strip(): return False
    new_ngrams = _get_ngrams(new_text, max_ngram)
    if not new_ngrams: return False
    recent = [h.get("assistant", "") for h in history[-window:] if h.get("assistant")]
    for prev in recent:
        prev_ngrams = _get_ngrams(prev, max_ngram)
        if not prev_ngrams: continue
        overlap = len(new_ngrams & prev_ngrams)
        if overlap / len(new_ngrams) > 0.3: return True
    return False

def _has_banned_opener(text: str) -> bool:
    banned = get_cfg("variety.banned_openers", []) or []
    if not banned: return False
    clean = text.strip().lower()
    first_line = clean.split('\n')[0] if '\n' in clean else clean
    first_sentence = first_line.split('.')[0] if '.' in first_line else first_line
    for phrase in banned:
        if not phrase: continue
        if phrase.strip().lower() in first_sentence:
            return True
    return False

def _apply_release_aliases(text: str) -> str:
    aliases = get_cfg("release_aliases", {}) or {}
    if not aliases: return text
    result = text
    for internal, public in aliases.items():
        if internal and public:
            result = re.sub(re.escape(internal), public, result, flags=re.IGNORECASE)
    return result

def _apply_humor_spontaneity(base_params: dict, mode: str) -> dict:
    if not bool(get_cfg("philosophy.humor.enabled", True)): return base_params
    import random
    temp_range = get_cfg("philosophy.humor.spontaneity.temperature_delta_range", [0.0, 0.1])
    top_p_range = get_cfg("philosophy.humor.spontaneity.top_p_range", [0.88, 0.94])
    rep_penalty = float(get_cfg("philosophy.humor.spontaneity.repetition_penalty", 1.12))
    params = dict(base_params)
    if temp_range and len(temp_range) == 2:
        params["temperature"] = params.get("temperature", 0.8) + random.uniform(temp_range[0], temp_range[1])
    if top_p_range and len(top_p_range) == 2:
        params["top_p"] = random.uniform(top_p_range[0], top_p_range[1])
    if rep_penalty > 0:
        params["repeat_penalty"] = rep_penalty
    log.info(f"[HUMOR] Applied spontaneity: temp={params.get('temperature'):.2f}, top_p={params.get('top_p'):.2f}, repeat={params.get('repeat_penalty'):.2f}")
    return params

def _maybe_inject_aside(text: str, mode: str) -> str:
    if not text.strip(): return text
    prob = float(get_cfg("variety.stochastic_aside.probability", 0.1) or 0.1)
    if prob <= 0: return text
    import random
    if random.random() > prob: return text
    template = get_cfg("variety.stochastic_aside.template", "⟪aside⟫ ${tiny_thought} — okay, back.")
    thoughts = {
        "astrofuck": ["that's wild","hold up","nah wait","side note","quick thing"],
        "dream": ["one moment","pause here","side thread","gentle aside","brief touch"]
    }
    thought_pool = thoughts.get(mode.lower(), ["brief aside"])
    thought = random.choice(thought_pool)
    aside = template.replace("${tiny_thought}", thought)
    if '.' in text:
        parts = text.split('.', 1)
        return f"{parts[0]}. {aside} {parts[1]}" if len(parts) > 1 else f"{text} {aside}"
    elif '\n\n' in text:
        parts = text.split('\n\n', 1)
        return f"{parts[0]}\n\n{aside}\n\n{parts[1]}" if len(parts) > 1 else f"{text}\n\n{aside}"
    else:
        return f"{text}\n\n{aside}"

def _post_process(text: str, mode: str, history: List[Dict] = None) -> str:
    t = _strip_banners(text)
    if _has_banned_opener(t):
        log.warning(f"[FILTER] Banned opener detected: '{t[:50]}...'")
        t = {"astrofuck": "Yo—let's slice this.","dream": "I'm here. Let's explore this together."}.get(mode.lower(), "I'm listening.")
    if history and _detect_echo(t, history, 
                                max_ngram=int(get_cfg("variety.echo_guard.max_ngram", 4) or 4),
                                window=int(get_cfg("variety.echo_guard.window_turns", 6) or 6)):
        log.warning(f"[ECHO] Repetition detected, adding variety marker")
        t = f"[catching myself looping—] {t}"
    t = _strip_emoji_except_glyphs(t)
    t = _ensure_slang_if_astro(t, mode)
    t = _persona_endings(t)
    t = _apply_release_aliases(t)
    t = _maybe_inject_aside(t, mode)
    return t

# ---------- API models ----------
class RequestModel(BaseModel):
    prompt: str = Field(..., max_length=8192)
    mode: Optional[str] = Field("astrofuck", max_length=50)
    turn: Optional[int] = 0
    session_id: Optional[str] = Field(None, max_length=36)
    new_session: bool = False
    force_lang: Optional[str] = Field(None, max_length=50)
    username: Optional[str] = Field(None, max_length=128)
    thread_id: Optional[str] = Field(None, max_length=128)

class MemoryUpsert(BaseModel):
    user: str = Field(..., max_length=128)
    text: str = Field(..., max_length=4096)

# ---------- Sessions ----------
def _sessions_dir() -> str:
    return SESSIONS_DIR

def _last_session_file() -> str:
    return os.path.join(SESSIONS_DIR, "last_session.txt")

def load_sessions():
    d = _sessions_dir()
    if not os.path.isdir(d):
        return
    with sessions_lock:
        for fn in os.listdir(d):
            if not fn.endswith(".json"):
                continue
            sid = fn[:-5]
            try:
                with open(os.path.join(d, fn), "r", encoding="utf-8") as f:
                    sessions[sid] = json.load(f)
            except Exception as e:
                log.warning(f"session load fail {sid}: {e}")
        if os.path.exists(_last_session_file()):
            with open(_last_session_file(), "r") as f:
                sessions.setdefault("__meta__", {})["last_seen"] = f.read().strip()
        log.info(f"sessions loaded: {len([k for k in sessions.keys() if k!='__meta__'])}")

def save_session(sid: Optional[str]):
    if not sid or sid not in sessions:
        return
    Path(_sessions_dir()).mkdir(parents=True, exist_ok=True)
    try:
        with open(os.path.join(_sessions_dir(), f"{sid}.json"), "w") as f:
            json.dump(sessions[sid], f)
        with open(_last_session_file(), "w") as f:
            f.write(sid)
    except Exception as e:
        log.error(f"session save fail {sid}: {e}")

def get_or_create_session(req: RequestModel) -> Dict:
    with sessions_lock:
        sid = req.session_id
        if req.new_session or not sid or sid not in sessions:
            sid = str(uuid.uuid4())
            sessions[sid] = {"id": sid, "turn": 0, "username": req.username, "history_by_thread": {}}
        s = sessions[sid]
        s.setdefault("username", req.username)
        return s

# ---------- FastAPI ----------
app = FastAPI(default_response_class=BestJSONResponse)

# CORS
if os.getenv("BLUR_PACKAGED") == "1":
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=".*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Session-ID", "X-TTFT-Last"],
    )
else:
    allowed = get_cfg("server.cors_allowed_origins", ["http://localhost:25329", "http://127.0.0.1:25329"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Session-ID", "X-TTFT-Last"],
    )

def _sse_headers() -> Dict[str, str]:
    return {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}

# ---------- Streaming ----------
def _thread_id_of(req) -> Optional[str]:
    return (req.thread_id or None)

async def generate_stream(session: Dict, req: RequestModel):
    global TTFT_LAST
    t0 = time.time()

    # Get per-session lock to prevent concurrent modifications
    sess_lock = _get_session_lock(session.get('id', 'default'))
    async with sess_lock:
        user_text = (req.prompt or "").strip()
        req_mode = (req.mode or "astrofuck").strip().lower()
        modes = set((get_cfg("range.modes", {}) or {}).keys())
        mode = req_mode if req_mode in modes else "astrofuck"

        # Detect tone from user input
        user_tone = _detect_tone(user_text)

        # Use instant language detection (or force_lang if provided)
        if req.force_lang:
            lang = req.force_lang
            session["active_lang"] = lang
        else:
            instant_mode = bool(get_cfg("language.instant_detection", True))
            lang = detect_and_set_language(session, user_text, force_instant=instant_mode)

        chat_key = get_cfg("chat.vessel_key", "qwen3_4b_unified")
        chat_llm = llm_vessels.get(chat_key)
        thread_id = _thread_id_of(req)

        def sse(data: str, event: Optional[str] = None) -> str:
            ev = f"event: {event}\n" if event else "event: token\n"
            return ev + "data: " + (data or "").replace("\n", "\ndata: ") + "\n\n"

        # session id upfront
        yield sse(json.dumps({"session_id": session.get("id")}), event="session_info")

        if not chat_llm:
            yield sse("model not loaded", event="error")
            return

        # History (mode-filtered)
        hist_all = _history_for(session, thread_id, int(get_cfg("assembly.history_turns", 30) or 30))
        hist_mode = [t for t in hist_all if (t.get("mode") or "").lower() == mode][
            -int(get_cfg("assembly.history_turns", 30) or 30) :
        ]

        # Context with enhanced filtering
        context = await asyncio.to_thread(
            retrieve_context_blend,
            user_text,
            session.get("id"),
            thread_id,
            session.get("username"),
            8,
            mode,
        )

        # Estimate ache BEFORE generating response (for adaptive prompting)
        initial_ache = _estimate_ache_score(user_text, "")
        ache_score = initial_ache.get("ache_score", 0.3)

        # System prompt with tone-based recall and adaptive witness
        sys_prompt = _build_system_prompt_with_recall(
            mode,
            _cap(context, int(get_cfg("assembly.context_cap", 2200) or 2200)),
            session,
            user_text,
            tone=user_tone,
        )
        sys_prompt = _cap(sys_prompt, int(get_cfg("assembly.system_prompt_cap", 4096) or 4096))

        # Messages
        msgs = _build_messages(lang, sys_prompt, hist_mode, user_text)

        # Mode params with humor spontaneity
        mp = (get_cfg(f"range.modes.{mode}.params", {}) or {})
        global_params = (manifest.get("params", {}) or {})

        base_params = {
            "temperature": float(mp.get("temperature", global_params.get("temperature", 0.8))),
            "top_p": float(mp.get("top_p", global_params.get("top_p", 0.95))),
            "repeat_penalty": float(mp.get("repeat_penalty", global_params.get("repeat_penalty", 1.1))),
        }

        # Apply humor spontaneity adjustments
        params = _apply_humor_spontaneity(base_params, mode)

        max_tokens = int(mp.get("n_predict", global_params.get("n_predict", 512)))
        stop = get_cfg(f"range.modes.{mode}.stop_tokens", ["</s>", "<|im_end|>"])

        call = _prune_params(
            chat_llm,
            {
                "messages": msgs,
                "temperature": params["temperature"],
                "top_p": params["top_p"],
                "repeat_penalty": params["repeat_penalty"],
                "max_tokens": max_tokens,
                "stop": stop,
                "stream": True,
                "cache_prompt": True,
            },
        )

        # KV prefill (best-effort) with model lock
        if call.get("cache_prompt"):
            try:
                async with _VESSEL_LOCK:
                    pre = dict(call)
                    pre["stream"] = False
                    pre["max_tokens"] = 0
                    await asyncio.to_thread(chat_llm.create_chat_completion, **pre)
            except Exception as e:
                log.warning(f"KV prefill fail: {e}")

        # Stream with real-time emoji stripping
        first = False
        acc = []
        try:
            for chunk in chat_llm.create_chat_completion(**call):
                if not first:
                    TTFT_LAST = time.time() - t0
                    first = True
                piece = (chunk.get("choices", [{}])[0].get("delta") or {}).get("content")
                if piece:
                    # Strip emoji DURING streaming so UX is consistent
                    clean_piece = _strip_emoji_except_glyphs(piece)
                    acc.append(clean_piece)
                    yield sse(clean_piece, event="token")
                    await asyncio.sleep(0)
        except (BrokenPipeError, ConnectionResetError):
            log.warning("client disconnected during stream")
            return
        except Exception as e:
            log.error(f"gen error: {e}", exc_info=True)
            yield sse(f"[core-error] {type(e).__name__}", event="error")
            return

        final = "".join(acc).strip() or "…"

        # Enhanced post-processing with Phase 1 filters
        final = _post_process(final, mode, history=hist_mode)

        # Record metrics AFTER response generation
        full_ache_metrics = _estimate_ache_score(user_text, final)
        turn = int(session.get("turn", 0))

        # Record tone tag
        response_tone = _detect_tone(final)
        record_tone_tag(session.get("id"), turn, user_tone, user_text)
        record_tone_tag(session.get("id"), turn, response_tone, final)

        # Record ache metrics
        record_ache_metrics(session.get("id"), turn, full_ache_metrics)

        # Log insights
        if full_ache_metrics.get("flip_detected"):
            log.info(f"[ACHE] Flip detected: {full_ache_metrics['ache_score']} → {full_ache_metrics['final_ache']}")

        # Save history (still inside session lock)
        _append_history(
            session,
            thread_id,
            user_text,
            final,
            mode,
            int(get_cfg("assembly.keep_history", 200) or 200),
        )
        session["turn"] = turn + 1
        save_session(session.get("id"))

        # Persist metrics periodically (every 5 turns)
        if turn % 5 == 0:
            await asyncio.to_thread(save_tone_tags)
            await asyncio.to_thread(save_ache_metrics)

        yield sse(json.dumps({"final": final}), event="final")
        yield sse("", event="done")

# ---------- Healthchecks ----------
def _ensure_fifo(path: str) -> Tuple[bool, str]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        if p.exists() and not p.is_fifo():
            p.unlink()
        if not p.exists():
            os.mkfifo(str(p))
        return True, "ok"
    except Exception as e:
        return False, str(e)

def _exec_cmd_expand(cmd: str) -> Tuple[bool, str]:
    expanded = resolve_path(cmd, homes)
    try:
        out = subprocess.run(
            expanded, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=8
        )
        ok = out.returncode == 0
        return ok, (out.stdout.strip() or out.stderr.strip() or f"exit={out.returncode}")
    except Exception as e:
        return False, str(e)

def run_healthchecks() -> List[Dict[str, Any]]:
    results = []
    for hc in (manifest.get("healthchecks") or []):
        name = hc.get("name", "unnamed")
        kind = hc.get("kind")
        required = bool(hc.get("required", False))
        ok = False
        detail = ""
        if kind == "fifo_ensure":
            p = (
                resolve_path(get_cfg("io.pipes.main", ""), homes)
                if hc.get("path") == "io.pipes.main"
                else resolve_path(hc.get("path", ""), homes)
            )
            ok, detail = _ensure_fifo(p)
        elif "cmd" in hc:
            ok, detail = _exec_cmd_expand(hc["cmd"])
        else:
            detail = "skipped (unknown kind)"
        results.append({"name": name, "ok": ok, "required": required, "detail": detail})
    return results

# ---------- RAG debug routes ----------
@app.get("/rag/status")
def rag_status():
    return {
        "loaded": bool(persistent_rag),
        "index_path": (str(persistent_rag.index_path) if (persistent_rag and getattr(persistent_rag, "index_path", None)) else None),
        "chunks_path": (str(persistent_rag.chunks_path) if (persistent_rag and getattr(persistent_rag, "chunks_path", None)) else None),
        "ntotal": int(getattr(persistent_rag.index, "ntotal", 0)) if persistent_rag and persistent_rag.index else 0,
        "nchunks": len(getattr(persistent_rag, "chunks", []) or []) if persistent_rag else 0,
    }

@app.post("/rag/reload")
def rag_reload():
    try:
        if not persistent_rag:
            return BestJSONResponse({"ok": False, "error": "rag not initialized"}, status_code=500)
        persistent_rag.load()
        return {
            "ok": True,
            "ntotal": int(getattr(persistent_rag.index, "ntotal", 0)) if persistent_rag.index else 0,
            "nchunks": len(persistent_rag.chunks),
        }
    except Exception as e:
        log.error(f"rag reload fail: {e}", exc_info=True)
        return BestJSONResponse({"ok": False, "error": str(e)}, status_code=500)

# ---------- Routes ----------
@app.get("/healthz")
def healthz():
    if CORE_IS_READY:
        return {"ok": True, "vessels": list(llm_vessels.keys()), "ttft_last": TTFT_LAST}
    return BestJSONResponse(status_code=503, content={"ok": False, "status": "initializing"})

@app.get("/healthchecks")
def healthchecks_route():
    return {"checks": run_healthchecks()}

@app.post("/generate_response")
async def generate_post(req: RequestModel, http: FastAPIRequest):
    session = get_or_create_session(req)
    headers = {"X-Session-ID": session.get("id", "")}
    if TTFT_LAST is not None:
        headers["X-TTFT-Last"] = str(TTFT_LAST)
    return StreamingResponse(
        generate_stream(session, req),
        media_type="text/event-stream",
        headers={**_sse_headers(), **headers},
    )

@app.get("/generate_response_get")
async def generate_get(
    request: FastAPIRequest,
    prompt: str,
    mode: Optional[str] = "astrofuck",
    turn: Optional[int] = 0,
    session_id: Optional[str] = None,
    new_session: bool = False,
    force_lang: Optional[str] = None,
    username: Optional[str] = None,
    thread_id: Optional[str] = None,
):
    req = RequestModel(
        prompt=prompt,
        mode=mode,
        turn=turn,
        session_id=session_id,
        new_session=new_session,
        force_lang=force_lang,
        username=username,
        thread_id=thread_id,
    )
    session = get_or_create_session(req)
    headers = {"X-Session-ID": session.get("id", "")}
    if TTFT_LAST is not None:
        headers["X-TTFT-Last"] = str(TTFT_LAST)
    return StreamingResponse(
        generate_stream(session, req),
        media_type="text/event-stream",
        headers={**_sse_headers(), **headers},
    )

@app.post("/memory/upsert")
def memory_upsert(payload: MemoryUpsert):
    try:
        upsert_user_memory(payload.user, payload.text)
        save_user_memory()
        return {"ok": True}
    except Exception as e:
        log.error(f"mem upsert fail: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
def list_sessions():
    """List all active sessions (debug endpoint)."""
    with sessions_lock:
        return {
            "count": len([k for k in sessions.keys() if k != "__meta__"]),
            "sessions": [
                {
                    "id": sid,
                    "turn": s.get("turn", 0),
                    "username": s.get("username"),
                    "active_lang": s.get("active_lang", "English"),
                    "threads": list(s.get("history_by_thread", {}).keys()),
                }
                for sid, s in sessions.items()
                if sid != "__meta__"
            ],
        }

@app.get("/sessions/{sid}")
def get_session_detail(sid: str):
    """Get detailed session info."""
    with sessions_lock:
        if sid not in sessions:
            raise HTTPException(status_code=404, detail="session not found")
        s = sessions[sid]
        return {
            "id": sid,
            "turn": s.get("turn", 0),
            "username": s.get("username"),
            "active_lang": s.get("active_lang"),
            "threads": {tid: len(hist) for tid, hist in s.get("history_by_thread", {}).items()},
        }

# ---------- Startup/Shutdown ----------
@app.on_event("startup")
async def on_start():
    global CORE_IS_READY, persistent_rag
    log.info("startup… resolving models & stores")

    # embedder (PATCHED: Don't abort startup if missing)
    try:
        async with _embed_lock:
            _ensure_embedder()
    except Exception as e:
        log.error(f"embedder fail: {e}")
        # continue in degraded mode (RAG may be disabled)

    # chat model (PATCHED: Don't abort startup if missing)
    chat_key = get_cfg("chat.vessel_key", "qwen3_4b_unified")
    try:
        async with _VESSEL_LOCK:
            if not load_llm_from_config(chat_key):
                log.error(f"chat vessel '{chat_key}' failed")
                # keep going; the API will report not loaded
    except Exception as e:
        log.error(f"chat vessel load exception: {e}")
        # keep going

    # RAG (PATCHED: Default to STATE_DIR)
    try:
        idx_path = resolve_path(get_cfg("memory.vector_store.path", ""), homes)
        ch_path = resolve_path(get_cfg("memory.vector_store.chunks_path", ""), homes)

        if not idx_path:
            idx_path = os.path.join(STATE_DIR, "blur_knowledge.index")
            log.warning(f"[RAG] No index path in config; defaulting to {idx_path}")
        if not ch_path:
            ch_path = os.path.join(STATE_DIR, "knowledge_chunks.jsonl")
            log.warning(f"[RAG] No chunks path in config; defaulting to {ch_path}")

        if os.path.isdir(ch_path):
            log.error(f"[RAG] chunks path points to a directory, expected file: {ch_path}")

        ttl = int(get_cfg("memory.vector_store.ttl_days_persistent", 0) or 0)
        auto = bool(get_cfg("memory.vector_store.auto_compact_on_start", True))

        if _EMBED_LLM is not None:
            persistent_rag = PersistentRAG(index_path=idx_path, chunks_path=ch_path, ttl_days=ttl, auto_compact=auto)
            persistent_rag.load()
        else:
            log.warning("[RAG] Skipping RAG load: Embedder is not online.")
    except Exception as e:
        log.error(f"RAG init failed: {e}")

    # Sessions + user memory + tone/ache tracking
    load_sessions()
    load_user_memory()
    load_tone_tags()
    load_ache_metrics()

    CORE_IS_READY = True
    log.info("Core ready (v10.4 - with tone/ache tracking).")

@app.delete("/sessions/{sid}")
def delete_session_api(sid: str):
    with sessions_lock:
        sessions.pop(sid, None)
    p = Path(SESSIONS_DIR) / f"{sid}.json"
    try:
        if p.exists():
            p.unlink()
    except Exception as e:
        log.warning(f"failed to delete session file {p}: {e}")
    return {"ok": True}

@app.post("/sessions/reset")
def reset_all_sessions_api():
    with sessions_lock:
        sessions.clear()
        sessions["__meta__"] = {}
    try:
        for f in Path(SESSIONS_DIR).glob("*.json"):
            f.unlink()
        lp = Path(SESSIONS_DIR) / "last_session.txt"
        if lp.exists():
            lp.unlink()
    except Exception as e:
        log.warning(f"session reset partial: {e}")
    return {"ok": True}

@app.on_event("shutdown")
def on_shutdown():
    save_user_memory()
    save_tone_tags()
    save_ache_metrics()
    for sid in list(sessions.keys()):
        save_session(sid)
    log.info("shutdown complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("BLUR_CORE_PORT", "25421")))
