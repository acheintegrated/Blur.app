// electron/main.mjs — REFORGED v11.8 (Standalone & Self-Carrying)
// - STANDALONE: Uses embedded blur_env + bundled Blur payload first; no system Python required in prod.
// - SPLASH: Instant, glowing bar; no black gap.
// - FAST-BOOT: Seeds userData only on version change; AI Core spawns in parallel.
// - SAFETY: In packaged apps, refuses to start if embedded Python is missing.
// - TOGGLES: BLUR_DISABLE_GPU=1, BLUR_NO_FASTLOOP=1, BLUR_SKIP_VERSION_COPY=1

import { app, BrowserWindow, shell, ipcMain, powerMonitor } from "electron";
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
   FLAGS & PERF
============================================================================ */
if (process.env.BLUR_DISABLE_GPU === "1") app.disableHardwareAcceleration();
app.commandLine.appendSwitch("allow-running-insecure-content");
app.commandLine.appendSwitch("disable-features", "OutOfBlinkCors");
app.commandLine.appendSwitch("enable-features", "CanvasOopRasterization,PartialSwap");
const SHOW_OVERLAY = !!process.env.BLUR_DEBUG_OVERLAY;
const isMac = process.platform === "darwin";

/* ============================================================================
   GLOBALS
============================================================================ */
let aiProc = null;
let mainWindow = null;
let splashWindow = null;
let aiReady = false;
let logFile = null;
let model_status = { status: "loading", message: "initiating…" };
let isQuitting = false;
let finalThreadsPayload = null;

const CORE_PORT = String(process.env.BLUR_CORE_PORT || "8000");
const CORE_HOST = process.env.BLUR_CORE_HOST || "127.0.0.1";
const CORE_BASE = `http://${CORE_HOST}:${CORE_PORT}`;
const CORE_HEALTH = `${CORE_BASE}/healthz`;

const DEV_URL = process.env.VITE_DEV_SERVER_URL || `http://localhost:6969`;
const APP_NAME = "Blur";
app.setName(APP_NAME);
const appDataRoot = app.getPath("appData");
const canonicalUserData = join(appDataRoot, APP_NAME);
app.setPath("userData", canonicalUserData);
try { if (!existsSync(canonicalUserData)) mkdirSync(canonicalUserData, { recursive: true }); } catch {}

/* ============================================================================
   LOGGING
============================================================================ */
const t0 = Date.now();
const bootMark = (label) => { const ms = ((Date.now() - t0)/1000).toFixed(2); log(`[boot] +${ms}s ${label}`); };
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
   PATHS / RESOURCES
============================================================================ */
function homeBlurPath() { return resolve(os.homedir(), "blur"); }
function packagedBlurPath() { return join(app.getPath("appData"), APP_NAME, "blur"); }
function getBundledBlurRoot() {
  const res = process.resourcesPath || join(__dirname, "..");
  const cands = [ join(res, "Blur"), join(res, "blur") ];
  for (const c of cands) { if (existsSync(c)) return c; }
  return null;
}
function statSafe(p) { try { return statSync(p); } catch { return null; } }
function cpRecursive(src, dst) {
  const st = statSafe(src); if (!st) return;
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
    const st = statSafe(p); if (!st) return;
    if (st.isDirectory()) { for (const f of readdirSync(p)) rmrf(join(p, f)); fs.rmdirSync(p); }
    else fs.unlinkSync(p);
  } catch {}
}
function readYamlVersion(text = "") {
  const m = text.match(/\bversion:\s*['"]?([^'"\n]+)['"]?/i);
  return (m && m[1] && m[1].trim()) || "";
}

async function ensureBlurHomeAndUnpack() {
  // DEV: prefer repo ./resources
  if (!app.isPackaged) {
    const devRoot = resolve(app.getAppPath(), "resources");
    if (existsSync(devRoot)) { process.env.BLUR_HOME = devRoot; log(`[dev] BLUR_HOME = ${devRoot}`); }
    else { log("[dev] ./resources missing; fallback to ~/blur"); }
  }

  const envSet = !!(process.env.BLUR_HOME && process.env.BLUR_HOME.trim());
  const homePath = homeBlurPath();
  const homeExists = existsSync(homePath);

  if (!envSet) {
    const bundledRoot = getBundledBlurRoot();
    if (app.isPackaged && bundledRoot) {
      // BUNDLE-FIRST in packaged apps
      process.env.BLUR_HOME = bundledRoot;
    } else {
      process.env.BLUR_HOME = homeExists ? homePath : (app.isPackaged ? packagedBlurPath() : homePath);
    }
  }

  const BLUR_HOME = process.env.BLUR_HOME;
  mkdirSync(BLUR_HOME, { recursive: true });
  log(`[resources] Using BLUR_HOME: ${BLUR_HOME}`);

  // keep Spotlight quiet
  try {
    const modelsPath = join(BLUR_HOME, "models");
    mkdirSync(modelsPath, { recursive: true });
    writeFileSync(join(modelsPath, ".metadata_never_index"), "");
  } catch {}

  // Seed only when not pointing directly at the bundle
  const bundledRoot = getBundledBlurRoot();
  const skipCopy = process.env.BLUR_SKIP_VERSION_COPY === "1";
  if (!skipCopy && bundledRoot && BLUR_HOME !== bundledRoot) {
    const filesCopyIfMissing = ["config.yaml", "acheflip.yaml"];
    const dirsReplaceOnBump = ["core"]; // immutable/critical payloads

    let needsCopy = false;
    try {
      const srcCfg = join(bundledRoot, "config.yaml");
      const dstCfg = join(BLUR_HOME, "config.yaml");
      const vSrc = existsSync(srcCfg) ? readYamlVersion(readFileSync(srcCfg, "utf8")) : "";
      const vDst = existsSync(dstCfg) ? readYamlVersion(readFileSync(dstCfg, "utf8")) : "";
      needsCopy = !existsSync(dstCfg) || (!!vSrc && vSrc !== vDst);
      log(`[resources] version check bundle→home: src=${vSrc||"?"} dst=${vDst||"?"} needsCopy=${needsCopy}`);
    } catch (e) { log("[resources] version compare failed:", String(e?.message || e)); }

    if (needsCopy) {
      for (const rel of filesCopyIfMissing) {
        const src = join(bundledRoot, rel);
        const dst = join(BLUR_HOME, rel);
        try {
          if (!existsSync(src)) continue;
          if (!existsSync(dst)) { mkdirSync(path.dirname(dst), { recursive: true }); copyFileSync(src, dst); log(`[resources] copied (new) ${rel}`); }
          else { log(`[resources] kept user ${rel}`); }
        } catch (e) { log(`[resources] copy error for ${rel}: ${String(e?.message || e)}`); }
      }
      for (const rel of dirsReplaceOnBump) {
        const src = join(bundledRoot, rel);
        const dst = join(BLUR_HOME, rel);
        try { if (existsSync(src)) { rmrf(dst); cpRecursive(src, dst); log(`[resources] replaced dir ${rel}`); } }
        catch (e) { log(`[resources] dir copy error for ${rel}: ${String(e?.message || e)}`); }
      }
    }

    // one-time models seed
    if (process.env.BLUR_SKIP_SEED_MODELS !== "1") {
      try {
        const modelsDst = join(BLUR_HOME, "models");
        const modelsSentinel = join(modelsDst, ".copied_ok");
        if (!existsSync(modelsSentinel)) {
          const src = join(bundledRoot, "models");
          if (existsSync(src)) {
            rmrf(modelsDst); cpRecursive(src, modelsDst); writeFileSync(modelsSentinel, String(Date.now()));
            log("[resources] Copied models (first-time seed).");
          }
        } else { log("[resources] Models directory present, skipping heavy copy."); }
      } catch (e) { log("[resources] Models copy error:", String(e?.message || e)); }
    }
  }

  // export useful env
  process.env.BLUR_RESOURCES_DIR = process.resourcesPath || "";
  process.env.BLUR_PACKAGED = app.isPackaged ? "1" : "0";
  if (!process.env.BLUR_CONFIG_PATH) {
    const cfg = join(BLUR_HOME, "config.yaml");
    if (existsSync(cfg)) process.env.BLUR_CONFIG_PATH = cfg;
  }
}

/* ============================================================================
   CONFIG
============================================================================ */
function requireConfigOrFail() {
  const envPath = process.env.BLUR_CONFIG_PATH;
  const home = process.env.BLUR_HOME;
  const homeCfg = home && join(home, "config.yaml");
  const candidates = [ envPath, homeCfg ].filter(p => p && existsSync(p));
  const cfg = candidates[0] || null;
  if (!cfg) {
    const msg = `Missing config.yaml. Set BLUR_CONFIG_PATH or place it in BLUR_HOME (${home || 'not set'}).`;
    log(`[config] FATAL: ${msg}`);
    model_status = { status: "error", message: msg };
  } else {
    process.env.BLUR_CONFIG_PATH = cfg;
    log(`[config] Using config @ ${cfg}`);
  }
  return cfg;
}

/* ============================================================================
   PREFS / THREADS
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
const THREADS_FILE = join(app.getPath("userData"), "threads.v1.json");
const THREADS_TMP  = join(app.getPath("userData"), "threads.v1.json.tmp");
const THREADS_BAK  = join(app.getPath("userData"), "threads.v1.json.bak");
function isNonEmptyThreads(arr) { return Array.isArray(arr) && arr.some(t => t && t.messages && t.messages.length); }
function writeThreadsFileAtomic(arr) {
  const empty = !Array.isArray(arr) || arr.length === 0;
  if (empty) {
    try { if (existsSync(THREADS_FILE)) fs.unlinkSync(THREADS_FILE); if (existsSync(THREADS_BAK)) fs.unlinkSync(THREADS_BAK); log("[threads] deleted all threads — files removed."); } catch (e) { log("[threads] delete error:", e?.message || e); }
    return true;
  }
  const data = JSON.stringify(arr, null, 2);
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
ipcMain.handle("threads:load", () => {
  try { if (!existsSync(THREADS_FILE)) return []; return JSON.parse(readFileSync(THREADS_FILE, "utf8")); }
  catch (e) { log("[threads] read failed, trying .bak:", e?.message || e); try { if (existsSync(THREADS_BAK)) return JSON.parse(readFileSync(THREADS_BAK, "utf8")); } catch {} return []; }
});
ipcMain.handle("threads:save", (_e, payload) => writeThreadsFileAtomic(payload));
ipcMain.handle("threads:send-final-state-for-quit", (_e, payload) => {
  if (isNonEmptyThreads(payload)) { finalThreadsPayload = payload; }
});

/* ============================================================================
   BACKEND / PYTHON
============================================================================ */
function looksLikeBackendDir(p) {
  try {
    if (!p || !existsSync(p)) return false;
    const st = statSafe(p); if (!st.isDirectory()) return false;
    const files = readdirSync(p);
    return ["convo_chat_core.py","core_server.py","core.py","server.py","app.py"].some(f => files.includes(f));
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
    resolve(process.env.BLUR_HOME,"electron","backend"),
    resolve(process.env.BLUR_HOME,"backend"),
  ];
  for (const c of candidates) { if (looksLikeBackendDir(c)) { log(`[backend] Found backend directory at: ${c}`); return c; } }
  const fallbackDir = join(process.cwd(), "electron", "backend");
  log(`[backend] Warning: Could not find backend directory, falling back to: ${fallbackDir}`);
  return fallbackDir;
}
function resolveCoreModule(backendDir) {
  const order = ["convo_chat_core.py","core_server.py","core.py","server.py","app.py"];
  for (const name of order) { if (existsSync(join(backendDir, name))) return `${basename(name, ".py")}:app`; }
  return "convo_chat_core:app";
}
function embeddedPythonCandidates() {
  const r = process.resourcesPath || "";
  return [
    join(r, "blur_env", "bin", "python3"),
    join(r, "blur_env-darwin-arm64", "bin", "python3"),
    join(process.env.BLUR_HOME || "", "blur_env", "bin", "python3"),
    join(process.env.BLUR_HOME || "", "blur_env-darwin-arm64", "bin", "python3"),
  ].filter(Boolean);
}
function resolvePython() {
  log("[python] Resolving Python executable…");
  // In packaged apps: require embedded python
  if (app.isPackaged) {
    for (const p of embeddedPythonCandidates()) {
      try { if (p && existsSync(p)) { log(`[python] embedded found ${p}`); return p; } } catch {}
    }
    // hard stop: not standalone if missing
    const msg = "Embedded Python not found in app bundle (Resources/blur_env).";
    log("[python] FATAL:", msg);
    model_status = { status: "error", message: msg };
    return null;
  }
  // dev: allow system python
  for (const p of embeddedPythonCandidates()) { try { if (p && existsSync(p)) { log(`[python] embedded found ${p}`); return p; } } catch {} }
  log("[python] dev mode using system python3");
  return "python3";
}

/* ============================================================================
   AI CORE
============================================================================ */
function buildUvicornArgs(backendDir, useFastLoop = true) {
  const moduleSpec = resolveCoreModule(backendDir);
  const base = ["-m","uvicorn", moduleSpec, "--host", CORE_HOST, "--port", CORE_PORT, "--log-level", process.env.BLUR_CORE_LOG || "info"];
  if (useFastLoop) base.push("--loop","uvloop","--http","httptools");
  return base;
}
function startAIServer(useFastLoop = (process.env.BLUR_NO_FASTLOOP === "1" ? false : true)) {
  if (aiProc) { log("[main] AI Core start ignored: already running."); return; }
  const cfgOK = requireConfigOrFail();
  if (!cfgOK) { broadcastStatus(); return; }
  const backendDir = findBackendDir();
  const pythonCommand = resolvePython();
  if (!pythonCommand) { broadcastStatus(); return; } // fail loudly in packaged

  const uvicornArgs = buildUvicornArgs(backendDir, useFastLoop);
  const threads = String(process.env.BLAS_THREADS || process.env.OPENBLAS_NUM_THREADS || 4);

  log("[main] Starting AI Core…");
  log("--- [AI Core Launch Details] ---");
  log(`[exec]  ${pythonCommand}`);
  log(`[args]  ${uvicornArgs.join(" ")}`);
  log(`[cwd]   ${backendDir}`);
  log(`[env]   BLUR_HOME=${process.env.BLUR_HOME}`);
  log(`[env]   BLUR_CONFIG_PATH=${process.env.BLUR_CONFIG_PATH}`);
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
      OMP_NUM_THREADS: threads,
      VECLIB_MAXIMUM_THREADS: threads,
      OPENBLAS_NUM_THREADS: threads,
      GOTO_NUM_THREADS: threads,
      MKL_NUM_THREADS: threads,
    },
    detached: process.platform !== "win32",
  });

  aiProc.on("error", (err) => {
    log("--- [CRITICAL AI CORE SPAWN ERROR] ---", String(err?.stack||err?.message||err));
    aiReady=false; model_status={status:"error",message:"Fatal: Failed to start AI Core. Check main.log."}; broadcastStatus();
  });
  aiProc.stderr.on("data", (d) => {
    const line=String(d).trim();
    log(`[AI Core STDERR]: ${line}`);
    if(/bad interpreter/i.test(line)){
      log("[main] Python venv broken."); stopAIServer(); aiReady=false; model_status={status:"error",message:"Fatal: Python environment is broken."}; broadcastStatus();
    }
    if(/(No module named 'uvloop'|No module named 'httptools')/i.test(line)&&useFastLoop){
      log("[main] Missing uvloop/httptools — restarting without fast loop…");
      stopAIServer(); startAIServer(false);
    }
  });
  aiProc.stdout.on("data", (d) => log(`[AI Core]: ${String(d).trim()}`));
  aiProc.on("exit", (code, signal) => {
    log(`--- [AI Core EXIT] code=${code} signal=${signal} ---`);
    aiProc = null; aiReady = false;
    if (model_status.status !== "stopping") {
      model_status = { status: "error", message: `AI Core exited abnormally (code ${code}).` };
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
   UI / SPLASH
============================================================================ */
function escapeForHtml(s = "") { return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/`/g,"\\`"); }
function splashHTML(msg = "initiating…") {
  const safe = escapeForHtml(msg);
  return `<!doctype html><html><head><meta charset="utf-8"/><title>Blur</title><style>
  html,body{height:100%;margin:0;background:#000;color:#fff}
  .mono{font-family:'Courier New',Menlo,Monaco,Consolas,monospace}
  .wrap{display:flex;align-items:center;justify-content:center;height:100%;flex-direction:column}
  .content{text-align:center}
  .dim{color:#aaa}.ok{color:#9efc9e}.warn{color:#ffd27a}.err{color:#ff8c8c}
  .logo{font-size:22px;margin-bottom:10px}
  .loader-bar{position:relative;width:220px;height:6px;background:rgba(218,112,214,.15);border-radius:6px;margin:18px auto 0;overflow:hidden}
  .loader-fill{position:absolute;inset:0;background:linear-gradient(90deg,#da70d633,#da70d688,#da70d633);animation:glow 1.2s ease-in-out infinite}
  @keyframes glow{0%{transform:translateX(-100%)}50%{transform:translateX(0%)}100%{transform:translateX(100%)}}
  </style></head><body><div class="wrap">
    <div class="content">
      <div class="mono logo"><b>Blur.</b></div>
      <div id="status" class="mono dim" style="font-size:13px">${safe}</div>
    </div>
    <div class="loader-bar"><div class="loader-fill"></div></div>
  </div></body></html>`;
}
function showSplash(initialMsg = model_status.message) {
  if (splashWindow && !splashWindow.isDestroyed()) return;
  splashWindow = new BrowserWindow({
    width: 420, height: 220, resizable: false, fullscreenable: false, frame: false,
    show: true, backgroundColor: "#000000", title: "Blur — booting",
    webPreferences: { backgroundThrottling: false }
  });
  const dataUrl = "data:text/html;charset=utf-8," + encodeURIComponent(splashHTML(initialMsg));
  splashWindow.loadURL(dataUrl).catch(()=>{});
  splashWindow.on("closed", () => { splashWindow = null; });
}
function updateSplash(msg, cls="dim") {
  try {
    if (splashWindow && !splashWindow.isDestroyed()) {
      const safe = escapeForHtml(String(msg||""));
      splashWindow.webContents.executeJavaScript(
        `(function(){var e=document.getElementById('status'); if(e){e.className='mono ${cls}'; e.textContent='${safe}';}})();`
      ).catch(()=>{});
    }
  } catch {}
}
function closeSplash() { try { if (splashWindow && !splashWindow.isDestroyed()) splashWindow.close(); } catch {} }

function ensureOverlay(win) {
  if (!SHOW_OVERLAY) return;
  const js = `(()=>{try{var o=document.getElementById('__blur_overlay');if(!o){o=document.createElement('div');o.id='__blur_overlay';o.style.cssText='position:fixed;left:8px;top:8px;z-index:2147483647;background:rgba(0,0,0,.65);color:#fff;padding:6px 8px;border-radius:6px;font:12px Menlo,monospace;pointer-events:none;display:none;';document.body.appendChild(o);}window.__blurSetOverlay=function(t){try{o.style.display='block';o.textContent=t;}catch(e){}};}catch(e){}})();`;
  win.webContents.executeJavaScript(js).catch(()=>{});
}
function broadcastStatus() {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("ai-status-update", model_status);
  }
  const cls = model_status.status === "ready" ? "ok" : model_status.status === "degraded" ? "warn" : "dim";
  updateSplash(model_status.message || "", cls);
}

/* ============================================================================
   WINDOW
============================================================================ */
async function createWindow() {
  const backgroundThrottling = false;
  mainWindow = new BrowserWindow({
    width: 1560, height: 1040, title: "Blur", backgroundColor: "#000000", show: false,
    webPreferences: {
      preload: join(__dirname, "preload.mjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
      backgroundThrottling,
      spellcheck: false
    },
  });
  try { app.dock?.show(); } catch {}

  mainWindow.webContents.on("console-message", (_e, level, message) => log(`[renderer:L${level}] ${message}`));
  mainWindow.webContents.on("did-finish-load", () => { ensureOverlay(mainWindow); bootMark("renderer finished load"); });
  mainWindow.webContents.once("dom-ready", () => {
    closeSplash();
    if (mainWindow && !mainWindow.isDestroyed() && !mainWindow.isVisible()) {
      mainWindow.showInactive();
    }
    bootMark("dom-ready");
  });

  if (!app.isPackaged) {
    await mainWindow.loadURL(DEV_URL).catch(e => log("[dev] loadURL error:", e));
    mainWindow.webContents.openDevTools({ mode: "detach" });
  } else {
    // prefer app.asar/dist, fallback to Resources/dist
    try { await mainWindow.loadFile(join(process.resourcesPath, "app.asar", "dist", "index.html")); }
    catch { await mainWindow.loadFile(join(process.resourcesPath, "dist", "index.html")); }
  }

  mainWindow.webContents.setWindowOpenHandler(({ url }) => { shell.openExternal(url); return { action: "deny" }; });
  mainWindow.on("closed", () => { mainWindow = null; });
}

/* ============================================================================
   NET / HEALTH
============================================================================ */
function ping(urlStr) { return new Promise((resolve, reject) => {
  try {
    const lib = urlStr.startsWith("https:") ? https : http;
    const req = lib.get(urlStr, { timeout: 1200 }, (res) => {
      const ok = (res.statusCode || 0) >= 200 && (res.statusCode || 0) < 400;
      res.resume(); ok ? resolve(true) : reject(new Error(`status ${res.statusCode}`));
    });
    req.on("error", reject);
    req.on("timeout", () => req.destroy(new Error("timeout")));
  } catch (e) { reject(e); }
});}
async function isCoreHealthy() { try { return await ping(CORE_HEALTH); } catch { return false; } }

/* ============================================================================
   IPC
============================================================================ */
ipcMain.handle("core:getInfo", () => ({ ready: !!aiReady, status: model_status }));
ipcMain.handle("core:healthz", async () => ({ ok: await isCoreHealthy(), status: model_status }));

/* ============================================================================
   POWER
============================================================================ */
function flushRendererNow() {
  if (mainWindow && !mainWindow.isDestroyed()) {
    log("[power] requesting flush from renderer");
    mainWindow.webContents.send("main-process-quitting");
  }
}
powerMonitor.on("suspend", () => { flushRendererNow(); });
powerMonitor.on("lock-screen", () => { flushRendererNow(); });

/* ============================================================================
   BOOT
============================================================================ */
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) { app.quit(); }
else {
  app.on("second-instance", () => { if (mainWindow) { if (mainWindow.isMinimized()) mainWindow.restore(); mainWindow.focus(); } });

  app.whenReady().then(async () => {
    initLogger(); bootMark("logger ready");
    log(`[startup] userData = ${app.getPath("userData")}`);
    showSplash(model_status.message);

    // resource prep in parallel with window boot
    const ensureP = (async () => { await ensureBlurHomeAndUnpack(); bootMark("resources ensured"); })();
    initPrefsIPC(); bootMark("prefs ready");
    await createWindow(); bootMark("window created & loading");
    await ensureP;

    const cfgOK = requireConfigOrFail();
    if (cfgOK) {
      startAIServer();
      model_status = { status: "starting", message: "initializing AI Core…" }; broadcastStatus();

      (async () => {
        const started = Date.now();
        while (Date.now() - started < 60000) {
          if (await isCoreHealthy()) {
            aiReady = true; model_status = { status: "ready", message: "AI Core ready" }; broadcastStatus(); return;
          }
          await new Promise(r => setTimeout(r, 1200));
        }
        if (model_status.status === "starting") {
          model_status = { status: "degraded", message: "AI Core startup timeout." }; broadcastStatus();
        }
      })();
    } else {
      broadcastStatus();
    }
  });
}

/* ============================================================================
   QUIT
============================================================================ */
function cleanQuit() {
  if (isQuitting) return; isQuitting = true;
  if (mainWindow && !mainWindow.isDestroyed()) {
    log("[quit] Requesting final state from renderer…");
    mainWindow.webContents.send("main-process-quitting");
    setTimeout(() => {
      if (isNonEmptyThreads(finalThreadsPayload)) { log("[quit] Saving final threads state…"); writeThreadsFileAtomic(finalThreadsPayload); }
      stopAIServer(); app.quit();
    }, 400);
    return;
  }
  stopAIServer(); app.quit();
}
app.on("before-quit", (e) => { if (!isQuitting) { e.preventDefault(); cleanQuit(); } });
app.on("window-all-closed", () => { if (!isMac) cleanQuit(); });
app.on("activate", () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
process.on("uncaughtException", (err) => log("[uncaughtException]", err?.stack || String(err)));
process.on("unhandledRejection", (reason) => log("[unhandledRejection]", reason?.stack || String(reason)));
