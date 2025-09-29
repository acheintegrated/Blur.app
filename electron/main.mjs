// electron/main.mjs — REFORGED v10.1 (BLUR_HOME-first, home-preferred config, versioned optional unpack, robust python launch)
import { app, BrowserWindow, shell, ipcMain, globalShortcut } from "electron";
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
   Goal: Prefer ~/blur if it exists. If not, unpack bundled resources to
         ~/Library/Application Support/Blur/blur (mac) and use that.
============================================================================ */
function homeBlurPath() { return resolve(os.homedir(), "blur"); }
function packagedBlurPath() { return join(app.getPath("appData"), APP_NAME, "blur"); } // ~/Library/Application Support/Blur/blur
function getBundledBlurRoot() {
  const res = process.resourcesPath || join(__dirname, "..");
  const cand = join(res, "blur"); // electron-builder extraResources → Contents/Resources/blur
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

/**
 * Ensure BLUR_HOME is set:
 *  - If env BLUR_HOME provided → use it.
 *  - Else if ~/blur exists → use it (no copy).
 *  - Else use packaged appData path and (if packaged & bundled) copy initial resources there.
 * Also sets BLUR_CONFIG_PATH to ${BLUR_HOME}/config.yaml if present.
 */
async function ensureBlurHomeAndUnpack() {
  const envSet = !!process.env.BLUR_HOME;
  const homePath = homeBlurPath();
  const homeExists = existsSync(homePath);
  const bundledRoot = getBundledBlurRoot();

  // Decide BLUR_HOME
  if (!envSet) {
    if (homeExists) {
      process.env.BLUR_HOME = homePath; // home-first if present
    } else {
      process.env.BLUR_HOME = app.isPackaged ? packagedBlurPath() : homePath; // dev defaults to ~/blur
    }
  }
  const BLUR_HOME = process.env.BLUR_HOME;
  mkdirSync(BLUR_HOME, { recursive: true });

  // If we’re using home tree, DO NOT overwrite with bundled files.
  const usingHome = BLUR_HOME === homePath;
  if (usingHome) {
    log(`[resources] Using existing home tree: ${BLUR_HOME}`);
  } else {
    // Using appData path; if packaged, copy/upgrade from bundled resources (version-aware)
    if (app.isPackaged && bundledRoot) {
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
        const toCopy = [
          "config.yaml",
          "acheflip.yaml",
          join("core","ouinet","blurchive","ecosystem"),
          "models",
          join("core","bin"),
        ];
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
  }

  // PATH for helper bins (won’t hurt in dev)
  const BLUR_BIN = join(BLUR_HOME, "core", "bin");
  const prevPath = process.env.PATH || "";
  if (!prevPath.split(path.delimiter).includes(BLUR_BIN) && existsSync(BLUR_BIN)) {
    process.env.PATH = [BLUR_BIN, prevPath].filter(Boolean).join(path.delimiter);
  }

  // Pin config if present and not already specified
  if (!process.env.BLUR_CONFIG_PATH) {
    const userCfg = join(BLUR_HOME, "config.yaml");
    if (existsSync(userCfg)) process.env.BLUR_CONFIG_PATH = userCfg;
  }

  // Optional whisper root
  if (!process.env.BLUR_WHISPER_ROOT) {
    const maybeWhisper = join(BLUR_HOME, "models", "whisper");
    if (existsSync(maybeWhisper)) process.env.BLUR_WHISPER_ROOT = maybeWhisper;
  }

  process.env.BLUR_RESOURCES_DIR = process.resourcesPath || "";
  process.env.BLUR_PACKAGED = app.isPackaged ? "1" : "0";
}

/* ============================================================================
   CONFIG BOOTSTRAP (only used if nothing provided)
   Order now: BLUR_CONFIG_PATH (if set) → ${BLUR_HOME}/config.yaml → bundled config → write minimal to ${BLUR_HOME}/config.yaml
============================================================================ */
function minimalConfigYAML() {
  const home = process.env.BLUR_HOME || homeBlurPath();
  return [
    "# Blur minimal bootstrap config",
    "meta:",
    "  homes:",
    `    blur_home: ${home}`,
    `    models: ${home}/models`,
    `    pipes: ${home}/run/pipes`,
    `    data: ${home}/core`,
    "",
    "chat:",
    "  vessel_key: qwen3_4b_unified",
    "",
    "engines:",
    "  llama_cpp:",
    "    python_n_ctx: 8192",
    "    n_gpu_layers: 3",
    "",
    "models:",
    "  qwen3_4b_unified:",
    "    engine: llama_cpp",
    `    path: ${home}/models/Qwen3-4B-Instruct-2507-UD-Q8_K_XL.gguf`,
    "  snowflake_arctic_embed:",
    "    engine: llama_cpp",
    `    path: ${home}/models/snowflake-arctic-embed-m-Q4_K_M.gguf`,
    "  bge_reranker_tiny:",
    "    engine: llama_cpp",
    `    path: ${home}/models/bge-reranker-v2-m3-Q8_0.gguf`,
    "",
    "memory:",
    "  vector_store:",
    "    engine: faiss",
    `    path: ${home}/core/ouinet/blurchive/ecosystem/blur_knowledge.index`,
    `    chunks_path: ${home}/core/ouinet/blurchive/ecosystem/knowledge_chunks.jsonl`,
    "    embed_model: snowflake_arctic_embed",
    "    ttl_days_persistent: 120",
    "    auto_compact_on_start: true",
    "",
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

  // last: create in HOME (not userData) so it’s stable
  try {
    mkdirSync(path.dirname(homeCfg), { recursive: true });
    writeFileSync(homeCfg, minimalConfigYAML(), { flag: "wx" });
    process.env.BLUR_CONFIG_PATH = homeCfg;
    log(`[config] Bootstrapped home config @ ${homeCfg}`);
    return homeCfg;
  } catch {
    // ultra-fallback
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
  get(key, fallback = null) { this._load(); return key ? (key in this.store ? this.store[key] : fallback) : this.store; }
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
   BACKEND DISCOVERY
============================================================================ */
function looksLikeBackendDir(p) {
  try {
    if (!p || !existsSync(p)) return false;
    const st = statSync(p); if (!st.isDirectory()) return false;
    const files = readdirSync(p);
    return ["convo_chat_core.py", "core_server.py", "core.py", "server.py", "app.py"].some(f => files.includes(f));
  } catch { return false; }
}
function findBackendDir() {
  const candidates = [
    join(process.cwd(), "electron", "backend"),
    join(__dirname, "backend"),
    join(__dirname, "..", "electron", "backend"),
    join(app.getAppPath(), "electron", "backend"),
    join(process.resourcesPath || "", "app.asar.unpacked", "electron", "backend"),
    join(process.resourcesPath || "", "electron", "backend"),
    resolve(process.env.BLUR_HOME || homeBlurPath(), "electron", "backend"),
    resolve("/opt/blurface/electron/backend"),
    resolve("/opt/blur/electron/backend"),
  ];
  for (const c of candidates) {
    if (looksLikeBackendDir(c)) {
      log(`[backend] Found backend directory at: ${c}`);
      return c;
    }
  }
  const fallbackDir = join(process.cwd(), "electron", "backend");
  log(`[backend] Warning: Could not find backend directory, falling back to: ${fallbackDir}`);
  return fallbackDir;
}
function resolveCoreModule(backendDir) {
  const order = ["convo_chat_core.py", "core_server.py", "core.py", "server.py", "app.py"];
  for (const name of order) {
    const p = join(backendDir, name);
    if (existsSync(p)) return `${basename(name, ".py")}:app`;
  }
  return "convo_chat_core:app";
}

/* ============================================================================
   PYTHON RESOLUTION
============================================================================ */
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
      ? cfg.pythonPath.replace("@@RESOURCES@@", process.resourcesPath) : cfg.pythonPath;
    if (existsSync(resolvedPath)) {
      log(`[python] packaged config → ${resolvedPath}`);
      return resolvedPath;
    }
  }
  const priority = [
    join(process.resourcesPath || "", "blur_env", "bin", "python3"),             // shipped venv (preferred)
    join(process.resourcesPath || "", "blur_env-darwin-arm64", "bin", "python3") // legacy name
  ];
  for (const p of priority) { if (existsSync(p)) { log(`[python] found ${p}`); return p; } }
  const fallbacks = [
    "/opt/blur_env-darwin-arm64/bin/python3",
    "/opt/blur_env-darwin-x64/bin/python3",
    "/opt/blur_env/bin/python3",
  ];
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
    cwd: backendDir,
    stdio: ["ignore", "pipe", "pipe"],
    env: {
      ...process.env,
      PYTHONUNBUFFERED: "1",
      PYTHONPATH: [backendDir, process.env.PYTHONPATH || ""].filter(Boolean).join(path.delimiter),
      BLUR_CORE_PORT: CORE_PORT,
      KMP_DUPLICATE_LIB_OK: "TRUE",
      OMP_NUM_THREADS: process.env.OMP_NUM_THREADS || "1",

      // 🔧 OpenMP duplicate guard (prevents SIGABRT on macOS)
      KMP_DUPLICATE_LIB_OK: "TRUE",
      // Optional: keep OpenMP thread counts sane
      OMP_NUM_THREADS: String(Math.max(1, os.cpus()?.length || 8)),
      VECLIB_MAXIMUM_THREADS: "1"
    },
    detached: process.platform !== "win32",
  });

  aiProc.on("error", (err) => {
    log("--- [CRITICAL AI CORE SPAWN ERROR] ---");
    log(`Failed to spawn: ${pythonCommand}`);
    log(String(err?.stack || err?.message || err));
    log("--- [END SPAWN ERROR] ---");
    aiReady = false;
    model_status = { status: "error", message: "Fatal: Failed to start AI Core. Check main.log." };
    broadcastStatus();
  });

  aiProc.stderr.on("data", (d) => {
    const line = String(d).trim();
    log(`[AI Core STDERR]: ${line}`);
    if (/bad interpreter/i.test(line)) {
      log("[main] Python venv broken.");
      stopAIServer();
      aiReady = false;
      model_status = { status: "error", message: "Fatal: Python environment is broken." };
      broadcastStatus();
      return;
    }
    if (/(No module named 'uvloop'|No module named 'httptools')/i.test(line) && useFastLoop) {
      log("[main] Missing uvloop/httptools — restarting without fast loop…");
      stopAIServer(); startAIServer(false); return;
    }
    if (/Error loading ASGI app|Could not import module/i.test(line)) {
      aiReady = false; model_status = { status: "error", message: "Core import failed. Check backend dir & module." }; broadcastStatus(); return;
    }
    if (/\b(ERROR|FATAL|Traceback|failed)\b/i.test(line)) {
      aiReady = false; model_status = { status: "error", message: "AI Core startup error (see main.log)" }; broadcastStatus();
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
   NET PING
============================================================================ */
function ping(urlStr) {
  return new Promise((resolve, reject) => {
    try {
      const lib = urlStr.startsWith("https:") ? https : http;
      const req = lib.get(urlStr, { timeout: 1500 }, (res) => {
        const ok = (res.statusCode || 0) >= 200 && (res.statusCode || 0) < 400;
        res.resume(); ok ? resolve(true) : reject(new Error(`status ${res.statusCode}`));
      });
      req.on("error", reject); req.on("timeout", () => req.destroy(new Error("timeout")));
    } catch (e) { reject(e); }
  });
}

/* ============================================================================
   SPLASH + OVERLAY
============================================================================ */
function escapeForHtml(s = "") {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/`/g,"\\`");
}
function splashHTML(msg = "Initializing…") {
  const safe = escapeForHtml(msg);
  return `<!doctype html><html><head><meta charset="utf-8"/><title>Blur</title>
  <style>html,body{height:100%;margin:0;background:#000;color:#fff}
  .mono{font-family:Menlo,Monaco,Consolas,'Courier New',monospace}
  .wrap{display:flex;align-items:center;justify-content:center;height:100%}
  .dim{color:#aaa}.ok{color:#9efc9e}.warn{color:#ffd27a}.err{color:#ff8c8c}</style></head>
  <body><div class="wrap"><div style="text-align:center">
  <div class="mono" style="font-size:22px;margin-bottom:10px">Blur</div>
  <div id="status" class="mono dim" style="font-size:13px">${safe}</div>
  <div id="hint" class="mono dim" style="font-size:11px;margin-top:6px">If this persists, check main.log in userData/logs.</div>
  </div></div></body></html>`;
}
async function loadSplash(win, msg) {
  const dataUrl = "data:text/html;charset=utf-8," + encodeURIComponent(splashHTML(msg));
  try { await win.loadURL(dataUrl); return true; } catch {}
  return false;
}
function ensureOverlay(win) {
  const js = `
    (function(){
      try{
        let o = document.getElementById('__blur_overlay');
        if(!o){
          o = document.createElement('div');
          o.id='__blur_overlay';
          o.style.cssText='position:fixed;left:8px;top:8px;z-index:2147483647;background:rgba(0,0,0,.65);color:#fff;padding:6px 8px;border-radius:6px;font:12px Menlo,monospace;pointer-events:none;display:none;';
          document.body.appendChild(o);
        }
        window.__blurSetOverlay = function(txt){ try{ o.style.display='block'; o.textContent = txt; }catch{} };
      }catch(e){}
    })();
  `;
  win.webContents.executeJavaScript(js).catch(() => {});
}
function setOverlayText(win, txt) {
  const safe = escapeForHtml(String(txt || ""));
  win.webContents.executeJavaScript(`window.__blurSetOverlay && window.__blurSetOverlay("${safe}")`).catch(()=>{});
}
function broadcastStatus() {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("ai-status-update", model_status);
    const cls = model_status.status === "ready" ? "ok" :
                model_status.status === "degraded" ? "warn" :
                model_status.status === "error" ? "err" : "dim";
    const safeMsg = escapeForHtml(model_status.message || "");
    mainWindow.webContents.executeJavaScript(`
      (function(){ const el=document.getElementById('status'); if(el){ el.className='mono ${cls}'; el.textContent='${safeMsg}'; } })();
    `).catch(() => {});
  }
}

/* ============================================================================
   WINDOW CREATE
============================================================================ */
async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1560, height: 1040, title: "Blur",
    backgroundColor: "#000000", show: false,
    webPreferences: {
      preload: join(__dirname, "preload.mjs"),
      contextIsolation: true, nodeIntegration: false, sandbox: false,
      webSecurity: false, allowRunningInsecureContent: true,
      backgroundThrottling: false,
    },
  });

  mainWindow.on("ready-to-show", () => { mainWindow.show(); });

  try { app.dock && app.dock.show(); } catch {}

  // diagnostics
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
    let devReady = false;
    let attempts = 0;
    let timer = null;

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
      try {
        const ok = await ping(DEV_URL);
        if (ok && !devReady) await tryLoad();
      } catch {}
      if (attempts % 5 === 0) setOverlayText(mainWindow, `waiting dev… (${attempts})`);
    }, 1500);

    mainWindow.webContents.openDevTools({ mode: "detach" });
  } else {
    try { await mainWindow.loadFile(join(__dirname, "../dist/index.html")); ensureOverlay(mainWindow); }
    catch (e) { log("[main] loadFile error:", e); /* keep splash */ }
  }

  // external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => { shell.openExternal(url); return { action: "deny" }; });
  mainWindow.on("closed", () => { mainWindow = null; });

  // hotkeys for reload
  app.whenReady().then(() => {
    const reg = (accel, fn) => { try { globalShortcut.register(accel, fn); } catch {} };
    reg(process.platform === "darwin" ? "CommandOrControl+Shift+R" : "Ctrl+Shift+R", () => {
      if (!mainWindow || mainWindow.isDestroyed()) return;
      setOverlayText(mainWindow, "manual reload…");
      if (!app.isPackaged) mainWindow.loadURL(DEV_URL).catch(()=>{});
      else mainWindow.loadFile(join(__dirname, "../dist/index.html")).catch(()=>{});
    });
  });

  // status tick
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

    // 🔑 1) Ensure BLUR_HOME (prefer ~/blur) and optionally unpack bundled resources if not using home
    await ensureBlurHomeAndUnpack();

    // 🔑 2) Ensure a config path exists (if none was provided)
    bootstrapConfigIfMissing();

    // 3) Create window (splash first), then start backend
    await createWindow();
    startAIServer(true);
    model_status = { status: "starting", message: "Initializing AI Core… (first run may copy models)" };
    broadcastStatus();

    // 4) wait for health
    (async () => {
      const started = Date.now();
      let ok = false;
      while (Date.now() - started < 60000) {
        try {
          await new Promise(r => setTimeout(r, 1500));
          const res = await ping(CORE_HEALTH);
          if (res) { ok = true; break; }
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

const cleanQuit = () => { stopAIServer(); app.quit(); };
app.on("before-quit", stopAIServer);
app.on("will-quit", stopAIServer);
app.on("window-all-closed", () => { if (process.platform !== "darwin") cleanQuit(); });
app.on("activate", () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });

process.on("uncaughtException", (err) => log("[uncaughtException]", err?.stack || String(err)));
process.on("unhandledRejection", (reason) => { const r = reason && (reason.stack || String(reason)); log("[unhandledRejection]", r); });
