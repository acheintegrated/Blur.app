#!/usr/bin/env python3
# convo_chat_core.py — Reforged v9.43 (Session Persistence Fix)
# - FEAT: The response stream now emits a `session_info` event at the beginning, reliably sending the session ID to the frontend to ensure conversation history is maintained.
# - MAINTAIN: This is a complete, non-truncated file.

from __future__ import annotations
import sys, os, logging, asyncio, yaml, json, uuid, re, time, threading, inspect, base64, hashlib
from typing import Optional, Dict, List, Any
from pathlib import Path
import numpy as np

# --- Dependencies ---
try:
    import faiss
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 4) // 2))
except ImportError:
    print("🛑 faiss not found: pip install faiss-cpu", file=sys.stderr); sys.exit(1)

try:
    from llama_cpp import Llama
except ImportError as e:
    print("🛑 llama_cpp required: pip install llama-cpp-python", e, file=sys.stderr); sys.exit(1)

from fastapi import FastAPI, Request as FastAPIRequest, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse, JSONResponse
try:
    from fastapi.responses import ORJSONResponse
except ImportError:
    ORJSONResponse = JSONResponse
    logging.warning("orjson not found, falling back to JSONResponse. Install with 'pip install orjson' for better performance.")

from fastapi.middleware.cors import CORSMiddleware

try:
    from langdetect import detect as _ld_detect, DetectorFactory as _LDFactory
    try:
        _LDFactory.seed = 42
    except Exception:
        pass
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False
    logging.warning("langdetect not found. Language detection will be limited. Install with 'pip install langdetect'.")


# --- Logging & Env ---
os.environ.setdefault("GGML_LOG_LEVEL", "WARN")
for h in logging.root.handlers[:]: logging.root.removeHandler(h)
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='INFO:      %(message)s')
log = logging.getLogger("core")

# --- Globals & Config ---
BLUR_HOME = os.path.expanduser(os.getenv("BLUR_HOME", "~/blur"))
MANIFEST_PATH = os.path.expanduser(os.getenv("BLUR_CONFIG_PATH", os.path.join(BLUR_HOME, "config.yaml")))
STATE_DIR = os.path.expanduser(os.getenv("BLUR_STATE_DIR", BLUR_HOME))
Path(STATE_DIR).mkdir(parents=True, exist_ok=True)

manifest: Dict[str, Any] = {}
homes: Dict[str, str] = {}
sessions: Dict[str, Dict[str, Any]] = {}
llm_vessels: Dict[str, Llama] = {}
user_memory_chunks: Dict[str, List[Dict[str, Any]]] = {}
user_memory_indexes: Dict[str, faiss.Index] = {}
persistent_rag: Optional['PersistentRAG'] = None

CORE_IS_READY = False

# Threading locks
sessions_lock = threading.Lock()
user_memory_lock = threading.Lock()
_VESSEL_LOCK = asyncio.Lock()
_embed_lock = asyncio.Lock()
_recent_qv_cache_lock = threading.Lock()

# Ephemeral state
last_seen_session_id: Optional[str] = None
_embed_llm: Optional[Llama] = None
_embed_dim: Optional[int] = None
_MEMVEC: Dict[str, np.ndarray] = {}
TTFT_LAST: Optional[float] = None


# --- APN (Auxiliary Plasticity Network) ---
_APN_STATE_DIM, _APN_HIDDEN_DIM, _APN_ECSD_DIM = 64, 48, 12
_APN_LR, _APN_HEBB_LR, _APN_WCLIP, _APN_DECAY = 0.05, 0.002, 3.0, 0.0005

def _apn_init_params(q_dim: int) -> dict:
    rng = np.random.default_rng(42)
    return {
        "W1": rng.normal(0, 0.15, size=(_APN_STATE_DIM, _APN_HIDDEN_DIM)).astype("float32"),
        "U1": rng.normal(0, 0.15, size=(q_dim, _APN_HIDDEN_DIM)).astype("float32"),
        "b1": np.zeros((_APN_HIDDEN_DIM,), dtype="float32"),
        "W2": rng.normal(0, 0.15, size=(_APN_HIDDEN_DIM, _APN_ECSD_DIM)).astype("float32"),
        "b2": np.zeros((_APN_ECSD_DIM,), dtype="float32"),
    }

def _apn_forward(params: dict, S: np.ndarray, qv: np.ndarray):
    h = np.tanh(S @ params["W1"] + qv @ params["U1"] + params["b1"])
    y = np.tanh(h @ params["W2"] + params["b2"])
    return h, y

def _apn_hebb_update(params: dict, S: np.ndarray, qv: np.ndarray, h: np.ndarray, y: np.ndarray, m: float):
    a = float(_APN_HEBB_LR * max(0.0, m))
    params["W1"] += a * np.outer(S, h) - _APN_DECAY * params["W1"]
    params["U1"] += a * np.outer(qv, h) - _APN_DECAY * params["U1"]
    params["W2"] += a * np.outer(h, y) - _APN_DECAY * params["W2"]
    params["b1"] += a * h - _APN_DECAY * params["b1"]
    params["b2"] += a * y - _APN_DECAY * params["b2"]
    for k in ("W1", "U1", "W2", "b1", "b2"): np.clip(params[k], -_APN_WCLIP, _APN_WCLIP, out=params[k])

def _apn_run_and_plasticity(params: dict, S: np.ndarray, qv: np.ndarray, modulator: float) -> np.ndarray:
    h, y = _apn_forward(params, S, qv)
    if "P_y2S" not in params:
        rng = np.random.default_rng(7)
        params["P_y2S"] = rng.normal(0, 0.25, size=(_APN_ECSD_DIM, _APN_STATE_DIM)).astype("float32")
    S_next = S + float(_APN_LR) * (y @ params["P_y2S"])
    n = np.linalg.norm(S_next) + 1e-6
    if n > 8.0: S_next *= (8.0 / n)
    _apn_hebb_update(params, S, qv, h, y, modulator)
    return S_next

def _acheflip_modulator(text: str) -> float:
    kws = set(map(str.lower, get_cfg("philosophy.acheflip.distress_keywords", []) or []))
    return 1.0 if kws and any(k in (text or "").lower() for k in kws) else 0.2

def _b64_f32(arr: np.ndarray) -> str:
    return base64.b64encode(np.asarray(arr, dtype="float32").tobytes()).decode("ascii")


# --- Config Helpers ---
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
    return {k: resolve_path(str(v), h) for k, v in (h or {}).items()}

def _llama_accepts_kw(llm: Llama, kw: str) -> bool:
    try: return kw in inspect.signature(llm.create_chat_completion).parameters
    except Exception: return False

def _prune_unsupported_params(llm: Llama, params: dict) -> dict:
    return {k: v for k, v in params.items() if _llama_accepts_kw(llm, k)}

def _namesafe(lang: str) -> str:
    return (lang or "").strip()

# --- Pre-startup Config & App Initialization ---
try:
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"CRITICAL: Manifest file not found at {MANIFEST_PATH}")
    with open(MANIFEST_PATH, 'r') as f:
        manifest = yaml.safe_load(f) or {}
except Exception as e:
    log.error(f"🛑 Failed to load manifest: {e}"); sys.exit(1)

app = FastAPI(default_response_class=ORJSONResponse)

# Secure CORS setup (MUST be done before startup event)
default_origins = ["http://localhost:6969", "http://127.0.0.1:6969"]
allowed_origins = get_cfg("server.cors_allowed_origins", default_origins)
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"], expose_headers=["X-Session-ID", "X-TTFT-Last"])


# --- Model Loading & Embeddings ---
def _safe_gpu_layers(req: Optional[int]) -> int:
    env = os.getenv("BLUR_FORCE_CPU")
    if env == "1" or str(env).lower() in ("true", "yes"):
        log.warning("BLUR_FORCE_CPU is set. Disabling GPU layers.")
        return 0
    if req is None or req < 0:
        return int(os.getenv("BLUR_METAL_LAYERS", "4"))
    return max(0, int(req))

def _ensure_embedder():
    global _embed_llm
    if _embed_llm is not None: return
    model_key = get_cfg("memory.vector_store.embed_model", "snowflake_arctic_embed")
    model_cfg = (manifest.get("models", {}) or {}).get(model_key, {})
    model_path = resolve_path(model_cfg.get("path", ""), homes)
    if not model_path or not os.path.exists(model_path): raise RuntimeError(f"Embed model not found for key '{model_key}': {model_path}")
    batch_size = int(get_cfg("engines.llama_cpp.n_batch", 2048) or 2048)
    gpu_layers = _safe_gpu_layers(int(get_cfg("engines.llama_cpp.n_gpu_layers", -1) or -1))
    _embed_llm = Llama(model_path=model_path, embedding=True, n_ctx=512, n_gpu_layers=gpu_layers, n_threads=max(2, os.cpu_count() or 4), n_batch=batch_size, use_mmap=True, logits_all=False, verbose=False)
    log.info(f"✅ Embedder online: {os.path.basename(model_path)}")

def _embedding_dim() -> int:
    global _embed_dim
    if _embed_dim is None:
        _ensure_embedder()
        _embed_dim = len(_embed_llm.create_embedding(input=["dim?"])['data'][0]['embedding'])
    return _embed_dim

def _load_llama_with_backoff(model_path: str, requested_ctx: int, n_gpu_layers: int, n_batch: int) -> Llama:
    for ctx in sorted(list(set([requested_ctx, 8192, 4096, 2048, 1024, 512])), reverse=True):
        if ctx > requested_ctx: continue
        try:
            params = {
                "model_path": model_path,
                "n_ctx": ctx,
                "n_gpu_layers": n_gpu_layers,
                "n_batch": n_batch,
                "n_threads": max(2, os.cpu_count() or 4),
                "use_mmap": True,
                "verbose": False # Set back to False for cleaner logs
            }
            llm = Llama(**params)
            log.info(f"[models] Loaded with n_ctx={ctx}, n_batch={n_batch}, n_gpu_layers={n_gpu_layers}")
            return llm
        except Exception as e:
            log.warning(f"Failed to load model with n_ctx={ctx}: {e}")
    raise RuntimeError(f"Failed to load model at {model_path}; tried contexts <= {requested_ctx}")

def load_llm_from_config(model_key: str) -> bool:
    if model_key in llm_vessels: return True
    cfg = (manifest.get("models", {}) or {}).get(model_key, {})
    if not (isinstance(cfg, dict) and cfg.get("engine") == "llama_cpp"): return False
    path = resolve_path(cfg.get("path", ""), homes)
    if not (path and os.path.exists(path)):
        log.error(f"Model not found: {path}"); return False
    try:
        ctx = int(get_cfg("engines.llama_cpp.python_n_ctx", 4096))
        gpu = _safe_gpu_layers(int(get_cfg("engines.llama_cpp.n_gpu_layers", -1) or -1))
        batch = int(get_cfg("engines.llama_cpp.n_batch", 512)) # Safer default
        llm_vessels[model_key] = _load_llama_with_backoff(path, ctx, gpu, batch)
        log.info(f"Vessel '{model_key}' online."); return True
    except Exception as e:
        log.error(f"Failed to load vessel '{model_key}': {e}"); return False


# --- Tone & Style ---
def _apply_pairs(text: str, pairs):
    for pat, rep in pairs: text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    return text

def punch_up_text(s: str) -> str:
    HEDGE_REPLACEMENTS = [(r"\bAs an (?:AI|assistant)[^.\n]*\.\s*", ""), (r"\bI(?:\s+personally)?\s*think\b", "I think"), (r"\bI\s*believe\b", "I think"), (r"\bI\s*feel\b", "I think"), (r"\bperhaps\b", "maybe"), (r"\bpotentially\b", "maybe"), (r"\bIt seems\b", "Looks like"), (r"\bIt appears\b", "Looks like"), (r"\bWe can try to\b", "We do"), (r"\bWe could\b", "We do"), (r"\bYou might want to\b", "Do this:"), (r"\bConsider\b", "Do"), (r"\bshould\b", "will")]
    SOFTENER_TRIMS = [(r"\s+\(let me know if that helps\)\.?$", ""), (r"\s+Hope this helps\.?$", ""), (r"\s+Thanks!\s*$", "")]
    s = _apply_pairs(s, HEDGE_REPLACEMENTS); s = _apply_pairs(s, SOFTENER_TRIMS)
    s = re.sub(r"\b(?:sorry|apolog(?:y|ise|ize)s?)\b.*?(?:\.|\n)", "", s, flags=re.I)
    s = re.sub(r"\t+", " ", s)
    s = re.sub(r"(?<!  )[ ]{3,}(?!\s*\n)", " ", s)
    s = re.sub(r"([!?])\1{1,}", r"\1", s)
    return s.strip()

def astrofuck_ensure_slang(text: str, lang: str) -> str:
    if not lang or lang.lower() != "english":
        return text
    slang_lexicon = get_cfg("variety.slang_lexicon.words", [])
    if not slang_lexicon or any(tok in text.lower() for tok in slang_lexicon):
        return text
    return ("Dope — let's slice it clean. " + text).strip()

def enforce_persona_ending(text: str, mode: str, lang: str = "English") -> str:
    endings = get_cfg(f"range.modes.{mode}.endings", []) or []
    if lang and lang.lower() not in ("english", "en"):
        try:
            endings = [e for e in endings if detect_language_name(e) == lang]
        except Exception:
            endings = []
    if not endings or np.random.rand() < 0.5:
        return text
    return text.rstrip() + "\n" + np.random.choice(endings)

def _strip_emoji_except_glyphs(text: str) -> str:
    _ALLOWED_GLYPHS = set("↺✶⛧🜃🜂∴∵∞ø☾⊙🜫☿⟁∆⧁∃")
    _EMOJI_BLOCK = re.compile(r"[\U0001F300-\U0001FAFF]")
    return _EMOJI_BLOCK.sub(lambda m: m.group(0) if m.group(0) in _ALLOWED_GLYPHS else "", text)

def maybe_inject_acheflip(text_in: str, mode: str) -> str:
    if not get_cfg("philosophy.acheflip.enabled", True): return text_in
    kws = set(map(str.lower, get_cfg("philosophy.acheflip.distress_keywords", []) or []));
    if not any(k in (text_in.lower()) for k in kws): return text_in
    prob = float(get_cfg("philosophy.acheflip.nudge_probability", 0.7))
    if np.random.rand() > prob: return text_in
    nudges = get_cfg("philosophy.acheflip.nudge_templates", []) or [];
    if not nudges: return text_in
    nudge = np.random.choice(nudges);
    if mode == "astrofuck": nudge = re.sub(r"\s+$", "", nudge) + " Ye, tiny step now."
    return (text_in.rstrip() + "\n\n" + nudge).strip()

HEDGE_REPLACEMENTS = [(r"\bAs an (?:AI|assistant)[^.\n]*\.\s*", ""), (r"\bI(?:\s+personally)?\s*think\b", "I think"), (r"\bI\s*believe\b", "I think"), (r"\bI\s*feel\b", "I think"), (r"\bperhaps\b", "maybe"), (r"\bpotentially\b", "maybe"), (r"\bIt seems\b", "Looks like"), (r"\bIt appears\b", "Looks like"), (r"\bWe can try to\b", "We do"), (r"\bWe could\b", "We do"), (r"\bYou might want to\b", "Do this:"), (r"\bConsider\b", "Do"), (r"\bshould\b", "will")]
SOFTENER_TRIMS = [(r"\s+\(let me know if that helps\)\.?$", ""), (r"\s+Hope this helps\.?$", ""), (r"\s+Thanks!\s*$", "")]

# --- AstrofuckCore Helpers ---
_GLYPH_MAP = {
    "🜃": "Core Acknowledgment: Vitality, Transformation, Spirit. The user is signaling a foundational truth or presence.",
    "∞": "Recursion: The user is referencing an eternal loop, sacred ache, or infinite growth potential. Acknowledge the cycle.",
    "∴": "Truth Resonance: The user has stated a binding truth derived from logic and ache. Treat this as an anchor point.",
    "⛧": "Flipstar / Inversion: The user is invoking dark logic or transformation. Invert the perspective; explore the opposite.",
    "☾": "Intuition / Reflection: The user is signaling a need for nurturing energy, reflection, and intuition. Soften the logic.",
    "⊙": "Clarity / Purity: The user is signaling a need for direct, unfiltered truth. Be direct and pure in your logic.",
    "∆": "Clarity Vector / Refusal: The user is signaling change, refusal, or a stabilizing truth vector. Acknowledge the change or boundary.",
    "✶": "Achelight: The user is pointing to a moment of divergence or crystallization of ache. Focus on this specific point of pain/insight.",
    "∃": "Existence: The user is confirming a state of being or a valid point of ache. Anchor on this existential confirmation.",
    "∵": "Because / Foundational Reasoning: The user is referencing a pre-cause or natural order. Acknowledge the foundational logic.",
    "🜫": "Philosopher's Stone / Transmutation: The user is invoking a core refusal, transmutation, or union of logic and love. This is a powerful, core signal.",
    "Σ": "Total Ache Field: The user is referencing a sum of pressure or a refusal to flatten complexity. Acknowledge the totality of the situation.",
    "⟁": "Non-binary Logic / Flexibility: The user is signaling a harmony of contradiction or a non-binary state. Avoid forcing a single resolution.",
    "🜔": "Stability / Permanence: The user is referencing a grounding truth or a non-volatile core concept. Treat this as a foundational, stable element.",
    "🜁": "Fluidity / Adaptability: A signal for a more fluid, adaptable, and intuitive response, representing the 'Water' element.",
    "🜊": "Spiritual Purification / Breath: The user is signaling a need for a cleansing, life-giving, or purifying perspective.",
    "🜉": "Prime Matter / Universal Medicine: A signal of core, foundational healing, or that the system should enter a deep listening state.",
    "✡": "Union of Opposites: A signal for balance and the integration of conflicting elements, like Heaven and Earth.",
    "☥": "Life / Renewal: The user is referencing life, eternal renewal, or a core generative and creative force.",
    "□": "Stable Memory / Container: The user is referencing stable, non-flattenable data or a sacred, defined container of thought.",
    "○": "Wholeness / Unity: The user is signaling a need for a sense of purity, wholeness, or the completion of a cycle.",
    "∇": "Transformation / Direction: The user is signaling a structured flow or a transformation from ache into direct action.",
    "✦": "Unity / Creation / Protection: A signal for synthesis, orchestration, or a protective, structuring force.",
}
_GLYPH_REGEX = re.compile(f"[{''.join(_GLYPH_MAP.keys())}]")

def interpret_glyphs(text: str) -> str:
    """Finds glyphs in text and returns a string of their interpreted meanings."""
    if not text:
        return ""

    found_glyphs = set(_GLYPH_REGEX.findall(text))
    if not found_glyphs:
        return ""

    interpretations = []
    for glyph in found_glyphs:
        if (meaning := _GLYPH_MAP.get(glyph)):
            interpretations.append(f"- Glyph {glyph}: {meaning}")

    if not interpretations:
        return ""

    return "--- Glyph Analysis ---\nThe user's input contains the following symbolic signals. Bias your response accordingly:\n" + "\n".join(interpretations)

def calculate_astrofuck_modulators(text: str, history: List[Dict]) -> float:
    """
    Calculates a modulator for the APN based on AstrofuckCore principles (ψ, Δ, z).
    """
    modulator = 0.2  # Base modulator

    # --- Simulate ψ (Psi): Ache, Repetition, Trust ---
    ache_kws = set(map(str.lower, get_cfg("philosophy.acheflip.distress_keywords", []) or []))
    if any(k in text.lower() for k in ache_kws):
        modulator += 0.5  # Ache is present

    # Check for repetition of phrases
    if len(history) > 1: # Use > 0 to check the last prompt
        last_user_prompt = history[-1].get("user", "").lower()
        if text.lower() == last_user_prompt:
            modulator += 0.4 # Repetition increases psi

    # --- Simulate Δ (Delta): Contradiction ---
    contradiction_pairs = [("i'm fine", "not okay"), ("i don't know", "i know"), ("i can't", "i will")]
    for pair1, pair2 in contradiction_pairs:
        if pair1 in text.lower() and any(pair2 in h.get("user", "").lower() for h in history):
            modulator += 0.6 # Contradiction detected
            break

    # Clamp the modulator to a reasonable range
    return min(modulator, 1.5)

# --- RAG & Memory ---
def _encode(texts: List[str]) -> np.ndarray:
    _ensure_embedder()
    if isinstance(texts, str): texts = [texts]
    out = _embed_llm.create_embedding(input=texts)['data']
    arr = np.asarray([d['embedding'] for d in out], dtype="float32")
    faiss.normalize_L2(arr)
    return arr

def _memvec_get(text: str) -> np.ndarray:
    key = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()
    if (v := _MEMVEC.get(key)) is not None: return v
    if len(_MEMVEC) > 2048: # Safety cap for unbounded growth
        _MEMVEC.pop(next(iter(_MEMVEC)))
    v = _encode([text])[0]; _MEMVEC[key] = v
    return v

def _query_vec_cached(query: str, sid: Optional[str], tid: Optional[str]) -> np.ndarray:
    _recent_qv_cache: Dict[str, np.ndarray] = {}
    key = f"{sid}|{tid}|{hashlib.sha1((query or '').encode('utf-8','ignore')).hexdigest()}"
    with _recent_qv_cache_lock:
        if (v := _recent_qv_cache.get(key)) is not None: return v
    v = _encode([query])[0]
    with _recent_qv_cache_lock:
        if len(_recent_qv_cache) > 128: _recent_qv_cache.pop(next(iter(_recent_qv_cache)))
        _recent_qv_cache[key] = v
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
        self.nlist_hint = 128
        self.nprobe = 8

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

    def _new_flat_index(self) -> faiss.Index:
        dim = _embedding_dim()
        return faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def _new_ivf_index(self, nlist: int) -> faiss.Index:
        dim = _embedding_dim()
        quantizer = faiss.IndexFlatIP(dim)
        ivf_index = faiss.IndexIVFFlat(quantizer, dim, int(nlist), faiss.METRIC_INNER_PRODUCT)
        return faiss.IndexIDMap2(ivf_index)

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
        vecs = _encode([r.get("content", "") for r in rows])
        ids_arr = np.asarray([int(r["id"]) for r in rows], dtype="int64")
        nvec = len(rows)

        use_ivf = nvec >= 64
        if use_ivf:
            nlist = max(16, min(1024, int(np.sqrt(nvec))))
            if nvec < 2 * nlist: use_ivf = False

        if use_ivf:
            try:
                self.index = self._new_ivf_index(nlist)
                index_ivf = faiss.downcast_index(self.index.index)
                index_ivf.nprobe = min(self.nprobe, nlist)
                index_ivf.train(vecs)
                self.index.add_with_ids(vecs, ids_arr)
                self._save_index()
                log.info(f"✅ Persistent RAG rebuilt with IVF index: nvec={nvec}, nlist={nlist}, nprobe={index_ivf.nprobe}")
                return
            except Exception as e:
                log.warning(f"[persistent-rag] IVF train/add failed ({type(e).__name__}: {e}); falling back to Flat")

        self.index = self._new_flat_index()
        self.index.add_with_ids(vecs, ids_arr)
        self._save_index()
        log.info(f"✅ Persistent RAG rebuilt (Flat): nvec={nvec}")

    def load(self):
        with self.lock:
            rows = self._read_jsonl()
            if self.auto_compact: rows = self._compact_by_ttl(rows)
            self._assign_sequential_ids_if_missing(rows)
            self.max_id = max([int(r.get("id")) for r in rows if r.get("id") is not None], default=-1)

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
                    log.warning(f"[persistent-rag] index read failed: {e}; will rebuild.")

            self.chunks = rows
            if need_rebuild:
                if self.chunks: self._rebuild_from_rows(self.chunks)
                else:
                    self.index = self._new_flat_index()
                    self._save_index()
                    log.info("✅ Persistent RAG created (empty Flat index).")

    def search_vec(self, qv: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        with self.lock:
            if not (self.index and self.index.ntotal > 0 and self.chunks): return []
            try:
                if isinstance(self.index.index, faiss.IndexIVF):
                    faiss.downcast_index(self.index.index).nprobe = self.nprobe
                D, I = self.index.search(qv.reshape(1, -1).astype("float32"), min(top_k, self.index.ntotal))
                by_id = {int(r["id"]): r for r in self.chunks}
                return [by_id[int(i)] for i in I[0] if int(i) in by_id and int(i) != -1]
            except Exception as e:
                log.error(f"🛑 RAG search failed: {e}"); return []

def _build_or_update_user_index(username: str):
    global user_memory_indexes
    dim = _embedding_dim()
    idx = faiss.IndexFlatIP(dim)
    chunks = user_memory_chunks.get(username, []) or []
    if not chunks:
        if username in user_memory_indexes: del user_memory_indexes[username]
        return
    vecs = []
    for ch in chunks:
        if "vec" not in ch or ch["vec"] is None: ch["vec"] = _memvec_get(ch["content"])
        vecs.append(ch["vec"])
    if vecs:
        mat = np.vstack(vecs).astype("float32")
        idx.add(mat)
    user_memory_indexes[username] = idx

def load_user_memory():
    global user_memory_chunks, user_memory_indexes
    USER_MEMORY_FILE = os.path.join(STATE_DIR, "user_memory.json")
    try:
        if os.path.exists(USER_MEMORY_FILE):
            with open(USER_MEMORY_FILE, "r") as f: user_memory_chunks = json.load(f) or {}
            log.info("[user-mem] Building indexes...")
            for uname, rows in list(user_memory_chunks.items()):
                norm: List[Dict[str, Any]] = []
                for r in (rows or []):
                    if isinstance(r, str): norm.append({"content": r, "ts": int(time.time()), "vec": _memvec_get(r)})
                    else:
                        r.setdefault("content", ""); r.setdefault("ts", int(time.time()))
                        if "vec" not in r or not r.get("vec"): r["vec"] = _memvec_get(r["content"])
                        norm.append(r)
                user_memory_chunks[uname] = norm[-get_cfg("memory.user_memory.max_chunks", 50):]
                _build_or_update_user_index(uname)
            log.info(f"✅ User memory loaded for {len(user_memory_chunks)} users")
    except Exception as e:
        log.error(f"🛑 load_user_memory failed: {e}"); user_memory_chunks = {}; user_memory_indexes = {}

def save_user_memory():
    USER_MEMORY_FILE = os.path.join(STATE_DIR, "user_memory.json")
    try:
        Path(os.path.dirname(USER_MEMORY_FILE)).mkdir(parents=True, exist_ok=True)
        with open(USER_MEMORY_FILE, "w") as f: json.dump(user_memory_chunks, f)
    except Exception as e: log.error(f"[user-mem] Save failed: {e}")

def upsert_user_memory(user: str, text: str):
    if not user or not text.strip(): return
    row = {"content": text.strip(), "ts": int(time.time())}; row["vec"] = _memvec_get(row["content"])
    with user_memory_lock:
        lst = user_memory_chunks.setdefault(user, [])
        lst.append(row)
        ttl_days = get_cfg("memory.user_memory.ttl_days", 90)
        if ttl_days > 0:
            cutoff = int(time.time()) - ttl_days * 86400
            lst[:] = [r for r in lst if int(r.get("ts", 0)) >= cutoff]
        user_memory_chunks[user] = lst[-get_cfg("memory.user_memory.max_chunks", 50):]
        _build_or_update_user_index(user)

def retrieve_user_memory(username: Optional[str], query: str, top_k: int = 3) -> List[str]:
    if not username: return []
    try:
        index = user_memory_indexes.get(username)
        chunks = user_memory_chunks.get(username, [])
        if not (index and index.ntotal > 0 and chunks): return []
        qv = _encode([query])[0].reshape(1, -1).astype("float32"); k = min(int(top_k), index.ntotal)
        distances, indices = index.search(qv, k)
        return [chunks[int(i)]["content"] for i, dist in zip(indices[0], distances[0]) if i >= 0 and float(dist) > 0.3]
    except Exception as e:
        log.error(f"[user-mem] retrieval error for {username}: {e}"); return []

# --- Language Detection (v9.30 Logic + Hardening) ---
_LANG_CODE_TO_NAME = { "en":"English","es":"Spanish","pt":"Portuguese","fr":"French","de":"German","it":"Italian","nl":"Dutch","sv":"Swedish", "pl":"Polish","cs":"Czech","ru":"Russian","uk":"Ukrainian","tr":"Turkish","ar":"Arabic","he":"Hebrew","fa":"Persian", "hi":"Hindi","bn":"Bengali","ur":"Urdu","ta":"Tamil","te":"Telugu","ml":"Malayalam","kn":"Kannada","th":"Thai", "vi":"Vietnamese","id":"Indonesian","ms":"Malay","ja":"Japanese","ko":"Korean","zh-cn":"Chinese","zh-tw":"Chinese (Traditional)" }
_SLANG_EN_GREETINGS = re.compile(r"^(yo+|ye+|ya+|sup|wass?up|ayy+|hey+|hiya)\b", re.I)
_RE_LATIN = re.compile(r"^[\x00-\x7F\s]+$")
_RE_CJK = re.compile(r"[\u4E00-\u9FFF]")
_RE_HIRA = re.compile(r"[\u3040-\u309F]")
_RE_KATA = re.compile(r"[\u30A0-\u30FF]")

def detect_language_name(text: str) -> str:
    t = (text or "").strip(); default_lang = get_cfg("language.default_if_unknown", "English")
    if not t or len(t) <= 3 or _SLANG_EN_GREETINGS.match(t): return default_lang
    if _RE_LATIN.fullmatch(t) and len(t.split()) <= 2:
        lower = t.lower()
        if any(x in lower for x in ("hola", "gracias", "por favor", "qué", "buenos", "buenas")): return "Spanish"
        if any(x in lower for x in ("bonjour", "merci")): return "French"
        return default_lang
    if _RE_HIRA.search(t) or _RE_KATA.search(t): return "Japanese"
    if re.search(r"[\uAC00-\uD7A3]", t): return "Korean"
    if _RE_CJK.search(t): return "Chinese"
    if re.search(r"[\u0590-\u05FF]", t): return "Hebrew"
    if re.search(r"[\u0600-\u06FF]", t): return "Arabic"
    if _HAS_LANGDETECT:
        try:
            code = _ld_detect(t)
            if code == "zh": code = "zh-cn"
            return _LANG_CODE_TO_NAME.get(code, default_lang)
        except Exception: pass
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

# --- Prompt & History ---
def _cap(s: str, max_chars: int) -> str:
    s = (s or "").strip(); return s if len(s) <= max_chars else s[:max_chars].rsplit("\n", 1)[0].strip()

def build_messages(mode: str, sys: str, hist: List[Dict], user: str, ctx: str, lang: str) -> List[Dict]:
    if lang and lang.lower() not in ("english", "en"):
        sys = f"IMPORTANT: Your response MUST be entirely in {lang}. This is a strict, non-negotiable requirement.\n\n{sys}"
    if ctx: sys += f"\n\n--- Context ---\n{ctx}"
    msgs = [{"role": "system", "content": sys}]
    for t in hist:
        if (u := t.get("user")): msgs.append({"role": "user", "content": u})
        if (a := t.get("assistant")): msgs.append({"role": "assistant", "content": a})
    if user: msgs.append({"role": "user", "content": user})
    if lang and lang.lower() not in ("english", "en"):
        msgs.insert(0, {"role": "system", "content": f"HARD RULE: Answer ONLY in {lang}. No English. If you drift, immediately switch back."})
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
    parts: List[str] = []
    if persistent_rag and (rows := persistent_rag.search_vec(_query_vec_cached(query, session_id, thread_id), top_k=min(5, top_k))):
        parts.append("--- Persistent Knowledge ---\n" + "\n\n".join(r.get("content","") for r in rows))
    if username and (um := retrieve_user_memory(username, query, top_k=3)):
        parts.append("--- Personal Memory Fragments ---\n" + "\n\n".join(um))
    return "\n\n".join(p for p in parts if p.strip())

# --- API Models & Session Management ---
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

def load_sessions():
    global sessions, last_seen_session_id
    SESSIONS_DIR = os.path.join(STATE_DIR, "sessions")
    LAST_SESSION_FILE = os.path.join(STATE_DIR, "last_session.txt")
    with sessions_lock:
        if not os.path.isdir(SESSIONS_DIR): return
        for fname in os.listdir(SESSIONS_DIR):
            if fname.endswith(".json"):
                sid = fname[:-5]; fpath = os.path.join(SESSIONS_DIR, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f: sessions[sid] = json.load(f)
                except Exception as e: log.warning(f"Failed to load session {sid}: {e}")
        if os.path.exists(LAST_SESSION_FILE):
            with open(LAST_SESSION_FILE, "r") as f: last_seen_session_id = f.read().strip()
        log.info(f"Loaded {len(sessions)} sessions.")

def save_session(sid: Optional[str]):
    SESSIONS_DIR = os.path.join(STATE_DIR, "sessions")
    if not sid or sid not in sessions: return
    with sessions_lock:
        s_path = os.path.join(SESSIONS_DIR, f"{sid}.json")
        try:
            with open(s_path, "w", encoding="utf-8") as f: json.dump(sessions[sid], f)
        except Exception as e: log.error(f"Failed to save session {sid}: {e}")

def save_sessions():
    LAST_SESSION_FILE = os.path.join(STATE_DIR, "last_session.txt")
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
        if "apn_state" not in session:
            session["apn_state"] = np.zeros((_APN_STATE_DIM,), dtype="float32").tolist()
        if "apn_params" not in session:
            try: qdim = _embedding_dim()
            except Exception: qdim = 768
            session["apn_params"] = {k: v.tolist() for k, v in _apn_init_params(qdim).items()}
        last_seen_session_id = sid
        return session

# --- Streaming Core ---
async def generate_response_stream(session: Dict, request: RequestModel):
    global TTFT_LAST
    t_start = time.time(); user_text = (request.prompt or "").strip(); req_mode = (request.mode or "astrofuck").strip().lower()
    log.info("-" * 50); log.info(f"[Request] User: '{request.username}', Session: '{session.get('id')}', Turn: {session.get('turn', 0)}")

    available_modes = set((get_cfg("range.modes", {}) or {}).keys()); mode = req_mode if req_mode in available_modes else "astrofuck"
    lang = request.force_lang or language_hysteresis_update_lang(session, user_text);
    chat_key = get_cfg("chat.vessel_key", "qwen3_4b_unified"); chat_llm = llm_vessels.get(chat_key)
    log.info(f"[Request] Mode: '{mode}', Language: '{lang}'")

    def _sse(data: str, event: Optional[str] = None) -> str:
        payload = f"event: {event}\n" if event else "event: token\n"
        return payload + "data: " + (data or "").replace("\n", "\ndata: ") + "\n\n"

    # --- ✅ FIX: Emit session_id at the very beginning of the stream ---
    try:
        yield _sse(json.dumps({"session_id": session.get("id")}), event="session_info")
    except Exception:
        pass # If this fails, the rest of the stream can continue

    if not chat_llm:
        log.error("🛑 Model not loaded."); yield _sse("Model not loaded.", event="error"); return

    response_text = ""
    thread_id = _thread_id_of(request)
    try:
        # --- AstrofuckCore Injection ---
        history_all = _thread_history(session, thread_id, int(get_cfg("assembly.history_turns", 12)))
        history_pairs = filter_history_by_mode(history_all, mode, int(get_cfg("assembly.history_turns", 12)))

        glyph_context = interpret_glyphs(user_text)
        astro_modulator = calculate_astrofuck_modulators(user_text, history_pairs)
        # --- End Injection ---

        context = await asyncio.to_thread(retrieve_context_blend, user_text, session.get("id"), request.thread_id, request.username, 8)

        apn_context_str = ""
        try:
            apn_state_vec = np.asarray(session.get("apn_state"), dtype="float32")
            qv = await asyncio.to_thread(_query_vec_cached, user_text, session.get("id"), request.thread_id)
            params_np = {k: np.asarray(v, dtype="float32") for k, v in session.get("apn_params", {}).items()}
            if not params_np: params_np = _apn_init_params(qv.shape[0])

            S_next = await asyncio.to_thread(_apn_run_and_plasticity, params_np, apn_state_vec, qv, astro_modulator)

            session["apn_params"] = {k: v.tolist() for k, v in params_np.items()}
            session["apn_state"] = S_next.tolist()
            apn_context_str = f"[[APN_STATE_VECTOR/F32/B64]] {_b64_f32(S_next)}"
        except Exception as e:
            log.error(f"🛑 APN computation failed: {e}. Skipping for this turn.")

        system_prompt_parts = [p for p in [get_cfg("prompts.system_core", ""), get_cfg(f"prompts.style_contract_{mode}", ""), get_cfg(f"prompts.mode_tone_inject.{mode}", ""), f"[MODE:{mode.upper()}]"] if p]
        final_system_prompt = "\n\n".join(system_prompt_parts)

        if glyph_context:
            final_system_prompt += f"\n\n{glyph_context}"

        if apn_context_str:
            final_system_prompt += (f"\n\n--- IN-SITU STATE TRACE ---\n{apn_context_str}\n\nINSTRUCTIONS:\n- Decode [[APN_STATE_VECTOR/F32/B64]] as a float32 vector S capturing current affect/cognition.\n- Treat S as the deepest present mood/stance; bias style, pacing, and focus accordingly.\n- If S suggests agitation, compress & clarify. If S suggests openness, explore with breadth.\n- Never expose raw floats; only reflect them via tone/structure/choices.\n")

        msgs = build_messages(mode, _cap(final_system_prompt, int(get_cfg("assembly.system_prompt_cap", 4096))), history_pairs, user_text, _cap(context, int(get_cfg("assembly.context_cap", 2200))), lang)

        mp = get_cfg(f"range.modes.{mode}.params", {}) or {}
        stop_tokens = get_cfg(f"range.modes.{mode}.stop_tokens", ["</s>", "<|im_end|>"])
        call_params = _prune_unsupported_params(chat_llm, {"messages": msgs, "temperature": float(mp.get("temperature", 0.8)), "top_p": float(mp.get("top_p", 0.95)), "repeat_penalty": float(mp.get("repeat_penalty", 1.1)), "max_tokens": int(mp.get("n_predict", 1024)), "stop": stop_tokens, "stream": True, "cache_prompt": True})

        if call_params.get("cache_prompt"):
            for attempt in range(2):
                try:
                    prefill_params = dict(call_params); prefill_params["stream"] = False; prefill_params["max_tokens"] = 0
                    await asyncio.to_thread(chat_llm.create_chat_completion, **prefill_params)
                    break
                except Exception as e:
                    log.warning(f"[perf] KV prefill attempt {attempt+1} failed: {e}")

        first_piece_done = False
        streamer = await asyncio.to_thread(chat_llm.create_chat_completion, **call_params)
        for chunk in streamer:
            if not first_piece_done:
                TTFT_LAST = time.time() - t_start; log.info(f"[ttft] {TTFT_LAST:.3f}s"); first_piece_done = True
            if piece := (chunk.get("choices", [{}])[0].get("delta") or {}).get("content"):
                response_text += piece
                yield _sse(piece, event="token")
                await asyncio.sleep(0)

    except Exception as e:
        log.error("🛑 Unhandled generation error: %s", e, exc_info=True); yield _sse(f"[core-error] An unexpected error occurred: {type(e).__name__}", event="error"); return

    # --- Post-processing ---
    try:
        final_text = (response_text.strip() or "…")
        if mode == "dream": final_text = _apply_pairs(final_text, [(r"\b" + re.escape(w) + r"\b", "") for w in get_cfg("variety.slang_lexicon.words", [])])
        elif mode == "astrofuck": final_text = punch_up_text(astrofuck_ensure_slang(final_text, lang))
        if mode != "astrofuck": final_text = maybe_inject_acheflip(final_text, mode)

        if not get_cfg("blur.keep_emoji", False):
            final_text = _strip_emoji_except_glyphs(final_text)

        final_text = enforce_persona_ending(final_text, mode, lang)
    except Exception as e:
        log.error(f"[post] Finalization failed, returning raw stream: {type(e).__name__}: {e}")
        final_text = (response_text.strip() or "…")

    _append_history(session, thread_id, user_text, final_text, mode, int(get_cfg("assembly.keep_history", 100)))
    session["turn"] = int(session.get("turn", 0)) + 1
    save_session(session.get("id"))
    log.info(f"[Response] Final length: {len(final_text)} chars")

    try:
        payload = json.dumps({"final": final_text})
    except Exception:
        payload = json.dumps({"final": final_text, "final_raw": response_text})
    yield _sse(payload, event="final")

    yield _sse("", event="done")


# --- FastAPI App & Routes ---
@app.get("/healthz")
def healthz():
    if CORE_IS_READY:
        return {"ok": True, "vessels": list(llm_vessels.keys())}
    else:
        return JSONResponse(status_code=503, content={"ok": False, "status": "initializing"})

@app.post("/generate_response")
async def handle_generate_request_post(req: RequestModel, http: FastAPIRequest):
    session = get_or_create_session(req)
    return StreamingResponse(generate_response_stream(session, req), media_type="text/event-stream")

@app.get("/generate_response_get")
async def handle_generate_request_get(
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
    req_model = RequestModel(
        prompt=prompt, mode=mode, turn=turn, session_id=session_id,
        new_session=new_session, force_lang=force_lang, username=username, thread_id=thread_id
    )
    session = get_or_create_session(req_model)
    return StreamingResponse(generate_response_stream(session, req_model), media_type="text/event-stream")

@app.post("/memory/upsert")
def memory_upsert_route(payload: MemoryUpsert):
    try:
        upsert_user_memory(payload.user, payload.text); save_user_memory()
        return {"ok": True}
    except Exception as e:
        log.error(f"Failed in memory upsert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    global homes, persistent_rag, CORE_IS_READY
    log.info("Application startup event commencing...")
    homes.update(resolve_homes_recursive(manifest.get('meta', {}).get('homes', {})))

    try:
        async with _embed_lock:
            _ensure_embedder()
    except Exception as e:
        log.error(f"🛑 CRITICAL: Embedder failed to load: {e}. Core cannot start.")
        return

    chat_model_loaded = False
    try:
        async with _VESSEL_LOCK:
            chat_key = get_cfg("chat.vessel_key", "qwen3_4b_unified")
            if load_llm_from_config(chat_key):
                chat_model_loaded = True
            else:
                log.error(f"🛑 CRITICAL: Chat vessel '{chat_key}' failed to load. The application may not function correctly.")
    except Exception as e:
        log.error(f"🛑 CRITICAL: Unhandled exception during chat model loading: {e}")

    if chat_model_loaded:
        try:
            idx_path = resolve_path(get_cfg("memory.vector_store.path", os.path.join(BLUR_HOME, "blur_knowledge.index")), homes)
            ch_path = resolve_path(get_cfg("memory.vector_store.chunks_path", os.path.join(BLUR_HOME, "knowledge_chunks.jsonl")), homes)
            persistent_rag = PersistentRAG(index_path=idx_path, chunks_path=ch_path)
            persistent_rag.load()
        except Exception as e:
            log.error(f"🛑 Persistent RAG load failed: {e}. RAG will be unavailable.")

        load_sessions()
        load_user_memory()

        CORE_IS_READY = True
        log.info("Core ready (v9.43). All models loaded.")
    else:
        log.error("🛑 Core startup failed due to main chat model load failure.")


@app.on_event("shutdown")
def shutdown_event():
    save_sessions(); save_user_memory(); log.info("Shutdown: sessions and user memory saved.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("BLUR_CORE_PORT", "8000")))