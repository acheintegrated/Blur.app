// electron/main.mjs — REFORGED v10.3 (sleep-safe + defensive saves)
import { app, BrowserWindow, shell, ipcMain, globalShortcut, powerMonitor } from "electron";
import { spawn } from "child_process";
import { fileURLToPath } from "url";
import path, { dirname, join, resolve, basename } from "path";
import os from "os";
import fs, {
  readFileSync, existsSync, mkdirSync, writeFileSync, appendFileSync,
  copyFileSync, readdirSync, statSync
} from "fs";
import http from "http";
import https from "https";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

/* ============================================================================
   FLAGS
============================================================================ */
app.disableHardwareAcceleration();
app.commandLine.appendSwitch("allow-running-insecure-content");
app.commandLine.appendSwitch("disable-features", "OutOfBlinkCors");

/* ============================================================================
   GLOBALS
============================================================================ */
let aiProc = null;
let mainWindow = null;
let aiReady = false;
let logFile = null;
let model_status = { status: "loading", message: "Initializing…" };

// 🔒 sleep/quit coordination flags
let isQuitting = false;
let lastRendererFinalAt = 0;

const CORE_PORT = String(process.env.BLUR_CORE_PORT || "8000");
const CORE_HOST = "127.0.0.1";
const CORE_BASE = `http://${CORE_HOST}:${CORE_PORT}`;
const CORE_HEALTH = `${CORE_BASE}/healthz`;

const DEV_PORT = process.env.VITE_DEV_SERVER_PORT || "6969";
const DEV_URL  = process.env.VITE_DEV_SERVER_URL  || `http://localhost:${DEV_PORT}`;
const APP_NAME = "Blur";

/* ============================================================================
   APP NAME & USERDATA
============================================================================ */
app.setName(APP_NAME);
const appDataRoot = app.getPath("appData");
const canonicalUserData = join(appDataRoot, APP_NAME);
app.setPath("userData", canonicalUserData);

// best-effort migrate legacy userData
try {
  if (!existsSync(canonicalUserData)) mkdirSync(canonicalUserData, { recursive: true });
  const oldDirs = [join(appDataRoot, "blurface"), join(appDataRoot, "Blurface")];
  for (const src of oldDirs) {
    if (!existsSync(src)) continue;
    const candidates = ["userprefs.json","threads.db","threads.sqlite","threads.sqlite3","data.db","data.sqlite","data.sqlite3"];
    for (const name of candidates) {
      const from = join(src, name), to = join(canonicalUserData, name);
      if (existsSync(from) && !existsSync(to)) { try { copyFileSync(from, to); } catch {} }
    }
    const oldLogs = join(src, "logs"), newLogs = join(canonicalUserData, "logs");
    if (existsSync(oldLogs) && !existsSync(newLogs)) {
      mkdirSync(newLogs, { recursive: true });
      for (const f of readdirSync(oldLogs)) { try { copyFileSync(join(oldLogs, f), join(newLogs, f)); } catch {} }
    }
    break;
  }
} catch {}

/* ============================================================================
   LOGGER
============================================================================ */
const initLogger = () => {
  try {
    const logDir = join(app.getPath("userData"), "logs");
    if (!existsSync(logDir)) mkdirSync(logDir, { recursive: true });
    logFile = join(logDir, "main.log");
  } catch {}
};
const log = (...args) => {
  const line = args.map(a => (typeof a === "string" ? a : JSON.stringify(a))).join(" ");
  console.log(line);
  if (!logFile) return;
  try { appendFileSync(logFile, `[${new Date().toISOString()}] ${line}\n`); } catch {}
};

/* ============================================================================
   BLUR_HOME + RESOURCES
============================================================================ */
function homeBlurPath() { return resolve(os.homedir(), "blur"); }
function packagedBlurPath() { return join(app.getPath("appData"), APP_NAME, "blur"); }
function getBundledBlurRoot() {
  const res = process.resourcesPath || join(__dirname, "..");
  const cand = join(res, "blur");
  return existsSync(cand) ? cand : null;
}
function statSafe(p) { try { return statSync(p); } catch { return null; } }
function cpRecursive(src, dst) {
  const st = statSafe(src);
  if (!st) return;
  if (st.isDirectory()) {
    mkdirSync(dst, { recursive: true });
    for (const name of readdirSync(src)) cpRecursive(join(src, name), join(dst, name));
  } else {
    mkdirSync(path.dirname(dst), { recursive: true });
    copyFileSync(src, dst);
  }
}
function rmrf(p) {
  try {
    const st = statSafe(p);
    if (!st) return;
    if (st.isDirectory()) {
      for (const f of readdirSync(p)) rmrf(join(p, f));
      fs.rmdirSync(p);
    } else fs.unlinkSync(p);
  } catch {}
}
function readYamlVersion(yamlText = "") {
  const m = yamlText.match(/^\s*version:\s*['"]?([^'"\n]+)['"]?/mi);
  return (m && m[1] && m[1].trim()) || "";
}

async function ensureBlurHomeAndUnpack() {
  const envSet = !!process.env.BLUR_HOME;
  const homePath = homeBlurPath();
  const homeExists = existsSync(homePath);
  const bundledRoot = getBundledBlurRoot();

  if (!envSet) {
    process.env.BLUR_HOME = homeExists ? homePath : (app.isPackaged ? packagedBlurPath() : homePath);
  }
  const BLUR_HOME = process.env.BLUR_HOME;
  mkdirSync(BLUR_HOME, { recursive: true });

  const usingHome = BLUR_HOME === homePath;
  if (usingHome) {
    log(`[resources] Using existing home tree: ${BLUR_HOME}`);
  } else if (app.isPackaged && bundledRoot) {
    const srcCfg = join(bundledRoot, "config.yaml");
    const dstCfg = join(BLUR_HOME, "config.yaml");

    let needsCopy = false;
    try {
      const srcText = existsSync(srcCfg) ? readFileSync(srcCfg, "utf8") : "";
      const dstText = existsSync(dstCfg) ? readFileSync(dstCfg, "utf8") : "";
      const vSrc = readYamlVersion(srcText);
      const vDst = readYamlVersion(dstText);
      needsCopy = !existsSync(dstCfg) || (!!vSrc && vSrc !== vDst);
      log(`[resources] bundled→appData version check: src=${vSrc || "?"} dst=${vDst || "?"} needsCopy=${needsCopy}`);
    } catch (e) {
      needsCopy = !existsSync(dstCfg);
      log("[resources] version compare failed; defaulting to copy:", String(e?.message || e));
    }

    if (needsCopy) {
      const toCopy = ["config.yaml", "acheflip.yaml", join("core","ouinet","blurchive","ecosystem"), "models", join("core","bin")];
      for (const rel of toCopy) {
        const src = join(bundledRoot, rel);
        const dst = join(BLUR_HOME, rel);
        try {
          if (!existsSync(src)) continue;
          if (statSafe(src)?.isDirectory()) {
            rmrf(dst);
            cpRecursive(src, dst);
          } else {
            mkdirSync(path.dirname(dst), { recursive: true });
            copyFileSync(src, dst);
          }
          log(`[resources] copied ${rel}`);
        } catch (e) {
          log(`[resources] copy error for ${rel}:`, String(e?.message || e));
        }
      }
    }
  } else {
    log(`[resources] No bundled root or not packaged; BLUR_HOME=${BLUR_HOME}`);
  }

  const BLUR_BIN = join(BLUR_HOME, "core", "bin");
  const prevPath = process.env.PATH || "";
  if (!prevPath.split(path.delimiter).includes(BLUR_BIN) && existsSync(BLUR_BIN)) {
    process.env.PATH = [BLUR_BIN, prevPath].filter(Boolean).join(path.delimiter);
  }

  if (!process.env.BLUR_CONFIG_PATH) {
    const userCfg = join(BLUR_HOME, "config.yaml");
    if (existsSync(userCfg)) process.env.BLUR_CONFIG_PATH = userCfg;
  }
  if (!process.env.BLUR_WHISPER_ROOT) {
    const maybeWhisper = join(BLUR_HOME, "models", "whisper");
    if (existsSync(maybeWhisper)) process.env.BLUR_WHISPER_ROOT = maybeWhisper;
  }

  process.env.BLUR_RESOURCES_DIR = process.resourcesPath || "";
  process.env.BLUR_PACKAGED = app.isPackaged ? "1" : "0";
}

/* ============================================================================
   CONFIG BOOTSTRAP
============================================================================ */
function minimalConfigYAML() {
  const home = process.env.BLUR_HOME || homeBlurPath();
  return [
    `# Blur minimal bootstrap config`,
    `meta:`,
    `  homes:`,
    `    blur_home: ${home}`,
    `    models: ${home}/models`,
    `    pipes: ${home}/run/pipes`,
    `    data: ${home}/core`,
    ``,
    `chat:`,
    `  vessel_key: qwen3_4b_unified`,
    ``,
    `engines:`,
    `  llama_cpp:`,
    `    python_n_ctx: 8192`,
    `    n_gpu_layers: 3`,
    ``,
    `models:`,
    `  qwen3_4b_unified:`,
    `    engine: llama_cpp`,
    `    path: ${home}/models/Qwen3-4B-Instruct-2507-UD-Q8_K_XL.gguf`,
    `  snowflake_arctic_embed:`,
    `    engine: llama_cpp`,
    `    path: ${home}/models/snowflake-arctic-embed-m-Q4_K_M.gguf`,
    `  bge_reranker_tiny:`,
    `    engine: llama_cpp`,
    `    path: ${home}/models/bge-reranker-v2-m3-Q8_0.gguf`,
    ``,
    `memory:`,
    `  vector_store:`,
    `    engine: faiss`,
    `    path: ${home}/core/ouinet/blurchive/ecosystem/blur_knowledge.index`,
    `    chunks_path: ${home}/core/ouinet/blurchive/ecosystem/knowledge_chunks.jsonl`,
    `    embed_model: snowflake_arctic_embed`,
    `    ttl_days_persistent: 120`,
    `    auto_compact_on_start: true`,
    ``,
  ].join("\n");
}
function bootstrapConfigIfMissing() {
  const envPath = process.env.BLUR_CONFIG_PATH;
  if (envPath && existsSync(envPath)) { log(`[config] Using BLUR_CONFIG_PATH=${envPath}`); return envPath; }

  const home = process.env.BLUR_HOME || homeBlurPath();
  const homeCfg = join(home, "config.yaml");
  if (existsSync(homeCfg)) {
    process.env.BLUR_CONFIG_PATH = homeCfg;
    log(`[config] Using home config @ ${homeCfg}`);
    return homeCfg;
  }

  const bundled = getBundledBlurRoot();
  const bundledCfg = bundled && join(bundled, "config.yaml");
  if (app.isPackaged && bundledCfg && existsSync(bundledCfg)) {
    process.env.BLUR_CONFIG_PATH = bundledCfg;
    log(`[config] Using bundled config @ ${bundledCfg}`);
    return bundledCfg;
  }

  try {
    mkdirSync(path.dirname(homeCfg), { recursive: true });
    writeFileSync(homeCfg, minimalConfigYAML(), { flag: "wx" });
    process.env.BLUR_CONFIG_PATH = homeCfg;
    log(`[config] Bootstrapped home config @ ${homeCfg}`);
    return homeCfg;
  } catch {
    const userCfg = join(app.getPath("userData"), "config.yaml");
    try {
      mkdirSync(path.dirname(userCfg), { recursive: true });
      writeFileSync(userCfg, minimalConfigYAML(), { flag: "w" });
      process.env.BLUR_CONFIG_PATH = userCfg;
      log(`[config] Bootstrapped userData config @ ${userCfg}`);
      return userCfg;
    } catch {
      const fallback = join(app.getPath("userData"), `config.${Date.now()}.yaml`);
      writeFileSync(fallback, minimalConfigYAML());
      process.env.BLUR_CONFIG_PATH = fallback;
      log(`[config] Bootstrapped fallback config @ ${fallback}`);
      return fallback;
    }
  }
}

/* ============================================================================
   PREFS (IPC)
============================================================================ */
class JsonPrefs {
  constructor(file) { this.file = file; this.store = {}; this._loaded = false; }
  _load() { if (this._loaded) return; try { if (existsSync(this.file)) this.store = JSON.parse(readFileSync(this.file, "utf-8") || "{}"); } catch { this.store = {}; } this._loaded = true; }
  _save() { try { writeFileSync(this.file, JSON.stringify(this.store, null, 2)); } catch {} }
  get(key, fallback = null) { this._load(); return key ? (this.store[key] ?? fallback) : this.store; }
  set(key, value) { this._load(); this.store[key] = value; this._save(); return true; }
  has(key) { this._load(); return key in this.store; }
  delete(key) { this._load(); delete this.store[key]; this._save(); return true; }
}
let prefs = null;
function initPrefsIPC() {
  const prefsPath = join(app.getPath("userData"), "userprefs.json");
  prefs = new JsonPrefs(prefsPath);
  ipcMain.handle("prefs:get",   (_e, key, fallback = null) => prefs.get(key, fallback));
  ipcMain.handle("prefs:set",   (_e, key, value) => prefs.set(key, value));
  ipcMain.handle("prefs:has",   (_e, key) => prefs.has(key));
  ipcMain.handle("prefs:delete",(_e, key) => prefs.delete(key));
  ipcMain.handle("env:get",     (_e, k) => process.env[k] ?? null);
  log(`[prefs] ready @ ${prefsPath}`);
}

/* ============================================================================
   THREADS PERSISTENCE (IPC) — sleep/quit safe
============================================================================ */
const THREADS_FILE = join(app.getPath("userData"), "threads.v1.json");
const THREADS_TMP  = join(app.getPath("userData"), "threads.v1.json.tmp");
const THREADS_BAK  = join(app.getPath("userData"), "threads.v1.json.bak");

const MAX_THREADS = 200;
const MAX_MSGS_PER_THREAD = 200;
const MAX_TEXT_LEN = 100_000;

function clampStr(s) { s = String(s ?? ""); return s.length > MAX_TEXT_LEN ? s.slice(0, MAX_TEXT_LEN) : s; }
function isSender(x){ return x === "Blur" || x === "You" || x === "System"; }
function normMessage(m) {
  if (!m || typeof m !== "object") return null;
  const sender = isSender(m.sender) ? m.sender : "System";
  const text = clampStr(m.text ?? "");
  const systemType = (m.systemType === "announcement" || m.systemType === "normal") ? m.systemType : undefined;
  if (!text) return null;
  return { sender, text, ...(systemType ? { systemType } : {}) };
}
function normThread(t) {
  if (!t || typeof t !== "object") return null;
  const id = typeof t.id === "string" && t.id.trim() ? t.id : `thr-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const title = clampStr(t.title ?? "");
  const autoGenerated = !!t.autoGenerated;
  const sessionId = typeof t.sessionId === "string" ? t.sessionId : null;
  const msgs = Array.isArray(t.messages) ? t.messages.map(normMessage).filter(Boolean).slice(-MAX_MSGS_PER_THREAD) : [];
  return { id, title, autoGenerated, messages: msgs, sessionId };
}
function normThreads(arr) {
  if (!Array.isArray(arr)) return [];
  const uniq = new Map();
  for (const t of arr.slice(-MAX_THREADS)) {
    const nt = normThread(t);
    if (nt) uniq.set(nt.id, nt);
  }
  return Array.from(uniq.values());
}

function readThreadsFile() {
  try {
    if (!existsSync(THREADS_FILE)) return [];
    const raw = readFileSync(THREADS_FILE, "utf8");
    return normThreads(JSON.parse(raw));
  } catch (e) {
    log("[threads] read failed, trying .bak:", e?.message || e);
    try { if (existsSync(THREADS_BAK)) return normThreads(JSON.parse(readFileSync(THREADS_BAK, "utf8"))); } catch {}
    return [];
  }
}

// helper: guard saves
function isNonEmptyThreads(arr) {
  return Array.isArray(arr) && arr.some(t => t && t.messages && t.messages.length);
}

// write atomically, but never clobber with []
function writeThreadsFileAtomic(arr) {
  if (!isNonEmptyThreads(arr)) {
    log("[threads] skip save: payload empty or invalid");
    return false;
  }
  const normalized = normThreads(arr);
  if (!isNonEmptyThreads(normalized)) {
    log("[threads] skip save: normalized payload empty");
    return false;
  }
  const data = JSON.stringify(normalized, null, 2);
  try {
    if (existsSync(THREADS_FILE)) { try { copyFileSync(THREADS_FILE, THREADS_BAK); } catch {} }
    writeFileSync(THREADS_TMP, data, "utf8");
    fs.renameSync(THREADS_TMP, THREADS_FILE);
    log("[threads] saved.");
    return true;
  } catch (e) {
    log("[threads] write failure:", e?.message || e);
    try { if (existsSync(THREADS_TMP)) fs.unlinkSync(THREADS_TMP); } catch {}
    return false;
  }
}

function migrateFromPrefsIfEmpty() {
  try {
    if (readThreadsFile().length > 0) return;
    const fromPrefs = prefs?.get?.("threads.v1", null);
    if (Array.isArray(fromPrefs) && fromPrefs.length > 0) {
      log("[threads] migrating from prefs:threads.v1 → threads.v1.json");
      writeThreadsFileAtomic(fromPrefs);
    }
  } catch (e) { log("[threads] migration failed:", e?.message || e); }
}

// IPC surface used by preload → renderer
ipcMain.handle("threads:load", () => {
  const data = readThreadsFile();
  log(`[threads] load -> ${data.length} threads`);
  return data;
});

ipcMain.handle("threads:save", (_e, payload) => {
  return writeThreadsFileAtomic(payload);
});

let finalThreadsPayload = null;
ipcMain.handle("threads:send-final-state-for-quit", (_e, payload) => {
  if (isNonEmptyThreads(payload)) {
    finalThreadsPayload = payload;
    lastRendererFinalAt = Date.now();
    log("[quit] final threads payload received:", payload.length, "threads");
  } else {
    log("[quit] ignored final payload (empty)");
  }
});

app.whenReady().then(() => migrateFromPrefsIfEmpty());

/* ============================================================================
   BACKEND DISCOVERY & PYTHON RESOLUTION
============================================================================ */
function looksLikeBackendDir(p) {
  try {
    if (!p || !existsSync(p)) return false;
    const st = statSafe(p); if (!st.isDirectory()) return false;
    const files = readdirSync(p);
    return ["convo_chat_core.py", "core_server.py", "core.py", "server.py", "app.py"].some(f => files.includes(f));
  } catch { return false; }
}
function findBackendDir() {
  const candidates = [
    join(process.cwd(),"electron","backend"),
    join(__dirname,"backend"),
    join(__dirname,"..","electron","backend"),
    join(app.getAppPath(),"electron","backend"),
    join(process.resourcesPath||"","app.asar.unpacked","electron","backend"),
    join(process.resourcesPath||"","electron","backend"),
    resolve(process.env.BLUR_HOME||homeBlurPath(),"electron","backend"),
    resolve("/opt/blurface/electron/backend"),
    resolve("/opt/blur/electron/backend"),
  ];
  for (const c of candidates) { if (looksLikeBackendDir(c)) { log(`[backend] Found backend directory at: ${c}`); return c; } }
  const fallbackDir = join(process.cwd(), "electron", "backend");
  log(`[backend] Warning: Could not find backend directory, falling back to: ${fallbackDir}`);
  return fallbackDir;
}
function resolveCoreModule(backendDir) {
  const order = ["convo_chat_core.py", "core_server.py", "core.py", "server.py", "app.py"];
  for (const name of order) { if (existsSync(join(backendDir, name))) return `${basename(name, ".py")}:app`; }
  return "convo_chat_core:app";
}
function readPackagedCfg() {
  try {
    const cfgPath = join(process.resourcesPath || join(__dirname, "..", "build"), "blur_backend.json");
    return JSON.parse(readFileSync(cfgPath, "utf-8"));
  } catch { return null; }
}
function resolvePython() {
  log("[python] Resolving Python executable...");
  const cfg = readPackagedCfg();
  if (cfg?.pythonPath) {
    const resolvedPath = (app.isPackaged && cfg.pythonPath.startsWith("@@RESOURCES@@"))
      ? cfg.pythonPath.replace("@@RESOURCES@@", process.resourcesPath)
      : cfg.pythonPath;
    if (existsSync(resolvedPath)) { log(`[python] packaged config → ${resolvedPath}`); return resolvedPath; }
  }
  const priority = [
    join(process.resourcesPath || "", "blur_env", "bin", "python3"),
    join(process.resourcesPath || "", "blur_env-darwin-arm64", "bin", "python3")
  ];
  for (const p of priority) { if (existsSync(p)) { log(`[python] found ${p}`); return p; } }
  const fallbacks = ["/opt/blur_env-darwin-arm64/bin/python3", "/opt/blur_env-darwin-x64/bin/python3", "/opt/blur_env/bin/python3"];
  for (const p of fallbacks) { if (existsSync(p)) { log(`[python] fallback ${p}`); return p; } }
  log("[python] using system python3");
  return "python3";
}

/* ============================================================================
   AI CORE SPAWN + READINESS
============================================================================ */
function buildUvicornArgs(backendDir, useFastLoop = true) {
  const moduleSpec = resolveCoreModule(backendDir);
  const base = ["-m", "uvicorn", moduleSpec, "--host", CORE_HOST, "--port", CORE_PORT, "--log-level", process.env.BLUR_CORE_LOG || "info"];
  if (useFastLoop) base.push("--loop", "uvloop", "--http", "httptools");
  return base;
}
function startAIServer(useFastLoop = true) {
  if (aiProc) { log("[main] AI Core start ignored: already running."); return; }
  const backendDir = findBackendDir();
  const pythonCommand = resolvePython();
  const uvicornArgs = buildUvicornArgs(backendDir, useFastLoop);
  log("[main] Starting AI Core…");
  log("--- [AI Core Launch Details] ---");
  log(`[exec]  ${pythonCommand}`);
  log(`[args]  ${uvicornArgs.join(" ")}`);
  log(`[cwd]   ${backendDir}`);
  const cfgPathForLog = process.env.BLUR_CONFIG_PATH || "";
  log(`[env]   BLUR_HOME=${process.env.BLUR_HOME}`);
  log(`[env]   BLUR_CONFIG_PATH=${cfgPathForLog} (exists=${cfgPathForLog ? existsSync(cfgPathForLog) : false})`);
  log("--------------------------------");
  aiProc = spawn(pythonCommand, uvicornArgs, {
    cwd: backendDir, stdio: ["ignore", "pipe", "pipe"],
    env: {
      ...process.env,
      PYTHONUNBUFFERED: "1",
      PYTHONPATH: [backendDir, process.env.PYTHONPATH || ""].filter(Boolean).join(path.delimiter),
      BLUR_CORE_PORT: CORE_PORT,
      KMP_DUPLICATE_LIB_OK: "TRUE",
      OMP_NUM_THREADS: process.env.OMP_NUM_THREADS || "1",
      VECLIB_MAXIMUM_THREADS: "1"
    },
    detached: process.platform !== "win32",
  });
  aiProc.on("error", (err) => {
    log("--- [CRITICAL AI CORE SPAWN ERROR] ---",
        `Failed to spawn: ${pythonCommand}`,
        String(err?.stack||err?.message||err),
        "--- [END SPAWN ERROR] ---");
    aiReady=false;
    model_status={status:"error",message:"Fatal: Failed to start AI Core. Check main.log."};
    broadcastStatus();
  });
  aiProc.stderr.on("data", (d) => {
    const line=String(d).trim();
    log(`[AI Core STDERR]: ${line}`);
    if(/bad interpreter/i.test(line)){
      log("[main] Python venv broken."); stopAIServer(); aiReady=false;
      model_status={status:"error",message:"Fatal: Python environment is broken."}; broadcastStatus(); return;
    }
    if(/(No module named 'uvloop'|No module named 'httptools')/i.test(line)&&useFastLoop){
      log("[main] Missing uvloop/httptools — restarting without fast loop…");
      stopAIServer(); startAIServer(false); return;
    }
    if(/Error loading ASGI app|Could not import module/i.test(line)){
      aiReady=false; model_status={status:"error",message:"Core import failed. Check backend dir & module."}; broadcastStatus(); return;
    }
    if(/\b(ERROR|FATAL|Traceback|failed)\b/i.test(line)){
      aiReady=false; model_status={status:"error",message:"AI Core startup error (see main.log)"}; broadcastStatus();
    }
  });
  aiProc.stdout.on("data", (d) => log(`[AI Core]: ${String(d).trim()}`));
  aiProc.on("exit", (code, signal) => {
    log(`--- [AI Core EXIT] code=${code} signal=${signal} ---`);
    aiProc = null; aiReady = false;
    if (model_status.status !== "stopping") {
      model_status = code && code !== 0 && model_status.status !== "error"
        ? { status: "error", message: `AI Core exited abnormally (code ${code}).` }
        : { status: "stopped", message: "AI Core stopped" };
      broadcastStatus();
    }
  });
}
function stopAIServer() {
  if (!aiProc) { log("[main] AI Core stop ignored: not running."); return; }
  log("[main] Stopping AI Core…");
  try {
    if (process.platform === "win32") spawn("taskkill", ["/PID", String(aiProc.pid), "/T", "/F"]);
    else { try { process.kill(-aiProc.pid, "SIGTERM"); } catch { process.kill(aiProc.pid, "SIGTERM"); } }
  } catch {
    try { aiProc.kill("SIGTERM"); } catch {}
    setTimeout(() => { try { aiProc.kill("SIGKILL"); } catch {} }, 1500);
  }
  aiProc = null; aiReady = false; model_status = { status: "stopping", message: "AI Core stopping…" }; broadcastStatus();
}

/* ============================================================================
   NET PING & UI
============================================================================ */
function ping(urlStr) { return new Promise((resolve, reject) => {
  try {
    const lib = urlStr.startsWith("https:") ? https : http;
    const req = lib.get(urlStr, { timeout: 1500 }, (res) => {
      const ok = (res.statusCode || 0) >= 200 && (res.statusCode || 0) < 400;
      res.resume(); ok ? resolve(true) : reject(new Error(`status ${res.statusCode}`));
    });
    req.on("error", reject);
    req.on("timeout", () => req.destroy(new Error("timeout")));
  } catch (e) { reject(e); }
});}
function escapeForHtml(s = "") { return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/`/g,"\\`");}
function splashHTML(msg = "Initializing…") {
  const safe = escapeForHtml(msg);
  return `<!doctype html><html><head><meta charset="utf-8"/><title>Blur</title><style>
  html,body{height:100%;margin:0;background:#000;color:#fff}
  .mono{font-family:Menlo,Monaco,Consolas,'Courier New',monospace}
  .wrap{display:flex;align-items:center;justify-content:center;height:100%}
  .dim{color:#aaa}.ok{color:#9efc9e}.warn{color:#ffd27a}.err{color:#ff8c8c}
  </style></head><body><div class="wrap"><div style="text-align:center">
  <div class="mono" style="font-size:22px;margin-bottom:10px">Blur</div>
  <div id="status" class="mono dim" style="font-size:13px">${safe}</div>
  <div id="hint" class="mono dim" style="font-size:11px;margin-top:6px">If this persists, check main.log in userData/logs.</div>
  </div></div></body></html>`;
}
async function loadSplash(win, msg) {
  const dataUrl = "data:text/html;charset=utf-8," + encodeURIComponent(splashHTML(msg));
  try { await win.loadURL(dataUrl); return true; } catch {} return false;
}
function ensureOverlay(win) {
  const js = `(function(){try{
    let o=document.getElementById('__blur_overlay');
    if(!o){o=document.createElement('div');o.id='__blur_overlay';
      o.style.cssText='position:fixed;left:8px;top:8px;z-index:2147483647;background:rgba(0,0,0,.65);color:#fff;padding:6px 8px;border-radius:6px;font:12px Menlo,monospace;pointer-events:none;display:none;';
      document.body.appendChild(o);}
    window.__blurSetOverlay=function(txt){try{o.style.display='block';o.textContent=txt;}catch{}}
  }catch(e){}})();`;
  win.webContents.executeJavaScript(js).catch(()=>{});
}
function setOverlayText(win, txt) {
  const safe = escapeForHtml(String(txt || ""));
  win.webContents.executeJavaScript(`window.__blurSetOverlay && window.__blurSetOverlay("${safe}")`).catch(()=>{});
}
function broadcastStatus() {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("ai-status-update", model_status);
    const cls = model_status.status === "ready" ? "ok" : model_status.status === "degraded" ? "warn" : model_status.status === "error" ? "err" : "dim";
    const safeMsg = escapeForHtml(model_status.message || "");
    mainWindow.webContents.executeJavaScript(`(function(){
      const el=document.getElementById('status'); if(el){el.className='mono ${cls}'; el.textContent='${safeMsg}';}
    })();`).catch(()=>{});
  }
}

/* ============================================================================
   WINDOW CREATE
============================================================================ */
async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1560, height: 1040, title: "Blur", backgroundColor: "#000000", show: false,
    webPreferences: {
      preload: join(__dirname, "preload.mjs"),
      contextIsolation: true, nodeIntegration: false, sandbox: false,
      webSecurity: false, allowRunningInsecureContent: true, backgroundThrottling: false,
    },
  });
  mainWindow.on("ready-to-show", () => { mainWindow.show(); });
  try { app.dock && app.dock.show(); } catch {}

  mainWindow.webContents.on("console-message", (_e, level, message, line, sourceId) => {
    log(`[renderer:console][L${level}] ${message} (${sourceId}:${line})`);
  });
  ipcMain.on("renderer:error", (_e, payload) => { log("[renderer:error]", payload); setOverlayText(mainWindow, `error: ${payload?.message||""}`); });
  ipcMain.on("renderer:unhandledRejection", (_e, payload) => { log("[renderer:unhandledRejection]", payload); setOverlayText(mainWindow, `rej: ${payload?.reason||""}`); });
  mainWindow.webContents.on("did-start-navigation", (_e, url) => log("[nav] start:", url));
  mainWindow.webContents.on("did-finish-load", () => { log("[nav] finished"); ensureOverlay(mainWindow); });
  mainWindow.webContents.on("did-fail-load", (_e, ec, desc, url) => { log("[nav] fail:", ec, desc, url); setOverlayText(mainWindow, `nav fail: ${desc}`); });
  mainWindow.webContents.on("render-process-gone", (_e, d) => { log("[renderer gone]", d && d.reason); setOverlayText(mainWindow, `renderer gone: ${d?.reason}`); });
  mainWindow.webContents.on("unresponsive", () => { log("[renderer] unresponsive"); setOverlayText(mainWindow, "renderer unresponsive"); });

  await loadSplash(mainWindow, model_status.message);

  if (!app.isPackaged) {
    let devReady = false; let attempts = 0; let timer = null;
    const stopTimer = () => { if (timer) { clearInterval(timer); timer = null; } };
    mainWindow.webContents.on("console-message", (_e, _lvl, message) => {
      if (typeof message === "string" && message.includes("[vite] connected")) { devReady = true; stopTimer(); }
    });
    const tryLoad = async () => {
      try { await mainWindow.loadURL(DEV_URL); ensureOverlay(mainWindow); setOverlayText(mainWindow, "dev loaded"); }
      catch (e) { log("[dev] loadURL error:", e?.message || e); }
    };
    await tryLoad();
    timer = setInterval(async () => {
      if (mainWindow?.isDestroyed()) return stopTimer();
      if (devReady) return stopTimer();
      attempts++;
      try { const ok = await ping(DEV_URL); if (ok && !devReady) await tryLoad(); } catch {}
      if (attempts % 5 === 0) setOverlayText(mainWindow, `waiting dev… (${attempts})`);
    }, 1500);
    mainWindow.webContents.openDevTools({ mode: "detach" });
  } else {
    try { await mainWindow.loadFile(join(__dirname, "../dist/index.html")); ensureOverlay(mainWindow); }
    catch (e) { log("[main] loadFile error:", e); }
  }

  mainWindow.webContents.setWindowOpenHandler(({ url }) => { shell.openExternal(url); return { action: "deny" }; });
  mainWindow.on("closed", () => { mainWindow = null; });

  app.whenReady().then(() => {
    const reg = (accel, fn) => { try { globalShortcut.register(accel, fn); } catch {} };
    reg(process.platform === "darwin" ? "CommandOrControl+Shift+R" : "Ctrl+Shift+R", () => {
      if (!mainWindow || mainWindow.isDestroyed()) return;
      setOverlayText(mainWindow, "manual reload…");
      if (!app.isPackaged) mainWindow.loadURL(DEV_URL).catch(()=>{});
      else mainWindow.loadFile(join(__dirname, "../dist/index.html")).catch(()=>{});
    });
  });

  const statusInterval = setInterval(() => {
    if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send("ai-status-update", model_status);
  }, 2000);
  mainWindow.on("closed", () => clearInterval(statusInterval));
}

/* ============================================================================
   IPC: core info
============================================================================ */
ipcMain.handle("core:getInfo", () => ({
  port: Number(CORE_PORT),
  baseUrl: CORE_BASE,
  ready: !!aiReady,
  status: model_status,
}));

/* ============================================================================
   POWER/SLEEP HOOKS — flush before suspend; don’t clobber on resume
============================================================================ */
const flushRendererNow = () => {
  try {
    if (mainWindow && !mainWindow.isDestroyed()) {
      log("[power] requesting flush from renderer (suspend)");
      mainWindow.webContents.send("main-process-quitting"); // reuse same channel to get final state
    }
  } catch {}
};

powerMonitor.on("suspend", () => {
  log("[power] suspend detected");
  flushRendererNow();
});
powerMonitor.on("lock-screen", () => {
  log("[power] lock-screen detected");
  flushRendererNow();
});
powerMonitor.on("resume", () => {
  log("[power] resume detected");
  // no-op: renderer will rebuild UI; we rely on defensive writer to avoid [] clobber
});
powerMonitor.on("unlock-screen", () => {
  log("[power] unlock-screen detected");
});

/* ============================================================================
   SINGLE INSTANCE & LIFECYCLE
============================================================================ */
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (mainWindow) { if (mainWindow.isMinimized()) mainWindow.restore(); mainWindow.focus(); }
  });

  app.whenReady().then(async () => {
    initLogger();
    log(`[startup] userData = ${app.getPath("userData")}`);
    initPrefsIPC();

    await ensureBlurHomeAndUnpack();
    bootstrapConfigIfMissing();
    await createWindow();
    startAIServer(true);
    model_status = { status: "starting", message: "Initializing AI Core… (first run may copy models)" };
    broadcastStatus();

    (async () => {
      const started = Date.now();
      let ok = false;
      while (Date.now() - started < 60000) {
        try {
          await new Promise(r => setTimeout(r, 1500));
          if (await ping(CORE_HEALTH)) { ok = true; break; }
        } catch {}
      }
      if (ok) { aiReady = true; model_status = { status: "ready", message: "AI Core ready" }; }
      else if (model_status.status === "loading" || model_status.status === "starting") {
        model_status = { status: "degraded", message: "AI Core startup timeout. Using degraded mode." };
      }
      broadcastStatus();
    })();
  });
}

/* ============================================================================
   ROBUST SHUTDOWN
============================================================================ */
const cleanQuit = () => {
  if (isQuitting) return;
  isQuitting = true;

  if (mainWindow && !mainWindow.isDestroyed()) {
    log("[quit] Requesting final state from renderer…");
    mainWindow.webContents.send("main-process-quitting");

    // give the renderer a breath; then save only if non-empty
    setTimeout(() => {
      const okArray = isNonEmptyThreads(finalThreadsPayload);
      if (okArray) {
        log("[quit] Saving final threads state…");
        writeThreadsFileAtomic(finalThreadsPayload);
      } else {
        // preserve existing file; never clobber with []
        try {
          const existing = readThreadsFile();
          if (isNonEmptyThreads(existing)) log("[quit] No final payload; preserving existing threads file.");
          else log("[quit] No final payload and no existing threads; skipping write.");
        } catch {}
      }
      stopAIServer();
      app.quit();
    }, 400); // slightly longer grace on sleepy laptops
    return;
  }

  // no window → just stop core and quit
  stopAIServer();
  app.quit();
};

app.on("before-quit", (e) => {
  log("[quit] 'before-quit' triggered.");
  // only intercept the first time; afterwards allow default quit
  if (!isQuitting) {
    e.preventDefault();
    cleanQuit();
  }
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") cleanQuit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

process.on("uncaughtException", (err) => log("[uncaughtException]", err?.stack || String(err)));
process.on("unhandledRejection", (reason) => { const r = reason && (reason.stack || String(reason)); log("[unhandledRejection]", r); });
