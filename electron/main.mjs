// electron/main.mjs — REFORGED v11.1 (Performance Optimizations)
// - FIX: Prevents heavy 'models' directory from being re-copied on simple config version bumps.
// - OPTIMIZE: Disables Spotlight indexing on the models directory to reduce system lag.
// - OPTIMIZE: Clamps additional Python math backends (OpenBLAS, GOTO, MKL) to a single thread.
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

const SHOW_OVERLAY = !!process.env.BLUR_DEBUG_OVERLAY;

/* ============================================================================
   GLOBALS
============================================================================ */
let aiProc = null;
let mainWindow = null;
let aiReady = false;
let logFile = null;
let model_status = { status: "loading", message: "Initializing…" };

let isQuitting = false;
let lastRendererFinalAt = 0;

const CORE_PORT = String(process.env.BLUR_CORE_PORT || "8000");
const CORE_HOST = "127.0.0.1";
const CORE_BASE = `http://${CORE_HOST}:${CORE_PORT}`;
const CORE_HEALTH = `${CORE_BASE}/healthz`;

const DEV_URL  = process.env.VITE_DEV_SERVER_URL  || `http://localhost:6969`;
const APP_NAME = "Blur";

/* ============================================================================
   APP NAME & USERDATA
============================================================================ */
app.setName(APP_NAME);
const appDataRoot = app.getPath("appData");
const canonicalUserData = join(appDataRoot, APP_NAME);
app.setPath("userData", canonicalUserData);

try {
  if (!existsSync(canonicalUserData)) mkdirSync(canonicalUserData, { recursive: true });
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
   BLUR_HOME + RESOURCES (HYBRID LOGIC)
============================================================================ */
function homeBlurPath() { return resolve(os.homedir(), "blur"); }
function packagedBlurPath() { return join(app.getPath("appData"), APP_NAME, "blur"); }
function getBundledBlurRoot() {
  const res = process.resourcesPath || join(__dirname, "..");
  const cand = join(res, "blur");
  if (existsSync(cand)) return cand;
  const cand2 = join(res, "resources");
  return existsSync(cand2) ? cand2 : null;
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
  if (!app.isPackaged) {
    const devResourcesPath = resolve(homeBlurPath(), "resources");
    if (existsSync(devResourcesPath)) {
        process.env.BLUR_HOME = devResourcesPath;
        log(`[dev] BLUR_HOME set to project resources: ${devResourcesPath}`);
    }
  }

  const envSet = !!process.env.BLUR_HOME;
  const homePath = homeBlurPath();
  const homeExists = existsSync(homePath);
  const bundledRoot = getBundledBlurRoot();

  if (!envSet) {
    process.env.BLUR_HOME = homeExists ? homePath : (app.isPackaged ? packagedBlurPath() : homePath);
  }
  const BLUR_HOME = process.env.BLUR_HOME;
  mkdirSync(BLUR_HOME, { recursive: true });

  log(`[resources] Using BLUR_HOME: ${BLUR_HOME}`);
  
  // ✅ OPTIMIZE: Kill Spotlight indexing on the models directory.
  const modelsPath = join(process.env.BLUR_HOME, "models");
  try {
    mkdirSync(modelsPath, { recursive: true }); // Ensure it exists
    writeFileSync(join(modelsPath, ".metadata_never_index"), "");
    log("[resources] Disabled Spotlight on models directory.");
  } catch {}

  if (app.isPackaged && bundledRoot && BLUR_HOME !== homePath) {
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
      log("[resources] version compare failed:", String(e?.message || e));
    }
    
    // ✅ FIX: Copy lightweight files if needed, but not heavy models.
    if (needsCopy) {
      const toCopyAlways = ["config.yaml", "acheflip.yaml", join("core","ouinet","blurchive","ecosystem"), join("core","bin")];
      for (const rel of toCopyAlways) {
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

    // ✅ FIX: Copy heavy models only once using a sentinel file.
    const modelsDst = join(BLUR_HOME, "models");
    const modelsSentinel = join(modelsDst, ".copied_ok");
    try {
      if (!existsSync(modelsSentinel)) {
        const src = join(bundledRoot, "models");
        if (existsSync(src)) {
          rmrf(modelsDst); // Clean slate for a fresh copy
          cpRecursive(src, modelsDst);
          writeFileSync(modelsSentinel, String(Date.now()));
          log("[resources] Copied models (first-time seed).");
        }
      } else {
        log("[resources] Models directory present, skipping heavy copy.");
      }
    } catch (e) {
      log("[resources] Models copy error:", String(e?.message || e));
    }
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

  process.env.BLUR_RESOURCES_DIR = process.resourcesPath || "";
  process.env.BLUR_PACKAGED = app.isPackaged ? "1" : "0";
}

/* ============================================================================
   CONFIG DISCOVERY
============================================================================ */
function requireConfigOrFail() {
  const envPath = process.env.BLUR_CONFIG_PATH;
  const home = process.env.BLUR_HOME;
  const homeCfg = home && join(home, "config.yaml");

  const candidates = [
    envPath && existsSync(envPath) ? envPath : null,
    homeCfg && existsSync(homeCfg) ? homeCfg : null,
  ].filter(Boolean);

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
   THREADS PERSISTENCE (IPC)
============================================================================ */
const THREADS_FILE = join(app.getPath("userData"), "threads.v1.json");
const THREADS_TMP  = join(app.getPath("userData"), "threads.v1.json.tmp");
const THREADS_BAK  = join(app.getPath("userData"), "threads.v1.json.bak");

function isNonEmptyThreads(arr) {
  return Array.isArray(arr) && arr.some(t => t && t.messages && t.messages.length);
}

function writeThreadsFileAtomic(arr) {
  const empty = !Array.isArray(arr) || arr.length === 0;
  if (empty) {
    try {
      if (existsSync(THREADS_FILE)) fs.unlinkSync(THREADS_FILE);
      if (existsSync(THREADS_BAK)) fs.unlinkSync(THREADS_BAK);
      log("[threads] deleted all threads — files removed.");
    } catch (e) {
      log("[threads] delete error:", e?.message || e);
    }
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
  try {
    if (!existsSync(THREADS_FILE)) return [];
    return JSON.parse(readFileSync(THREADS_FILE, "utf8"));
  } catch (e) {
    log("[threads] read failed, trying .bak:", e?.message || e);
    try { if (existsSync(THREADS_BAK)) return JSON.parse(readFileSync(THREADS_BAK, "utf8")); } catch {}
    return [];
  }
});
ipcMain.handle("threads:save", (_e, payload) => writeThreadsFileAtomic(payload));

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
    resolve(process.env.BLUR_HOME,"electron","backend"),
    resolve(process.env.BLUR_HOME,"backend"),
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
function resolvePython() {
  log("[python] Resolving Python executable...");
  const priority = [
    join(process.env.BLUR_HOME, "blur_env-darwin-arm64", "bin", "python3"),
    join(process.resourcesPath || "", "blur_env", "bin", "python3"),
    join(process.resourcesPath || "", "blur_env-darwin-arm64", "bin", "python3"),
  ];
  for (const p of priority) { if (existsSync(p)) { log(`[python] found ${p}`); return p; } }
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
  const cfgOK = requireConfigOrFail();
  if (!cfgOK) { broadcastStatus(); return; }
  const backendDir = findBackendDir();
  const pythonCommand = resolvePython();
  const uvicornArgs = buildUvicornArgs(backendDir, useFastLoop);
  log("[main] Starting AI Core…");
  log("--- [AI Core Launch Details] ---");
  log(`[exec]  ${pythonCommand}`);
  log(`[args]  ${uvicornArgs.join(" ")}`);
  log(`[cwd]   ${backendDir}`);
  log(`[env]   BLUR_HOME=${process.env.BLUR_HOME}`);
  log(`[env]   BLUR_CONFIG_PATH=${process.env.BLUR_CONFIG_PATH}`);
  log("--------------------------------");
  aiProc = spawn(pythonCommand, uvicornArgs, {
    cwd: backendDir, stdio: ["ignore", "pipe", "pipe"],
    env: {
      ...process.env,
      PYTHONUNBUFFERED: "1",
      PYTHONPATH: [backendDir, process.env.PYTHONPATH || ""].filter(Boolean).join(path.delimiter),
      BLUR_CORE_PORT: CORE_PORT,
      // ✅ OPTIMIZE: Clamp Python math backends to a single thread.
      KMP_DUPLICATE_LIB_OK: "TRUE",
      OMP_NUM_THREADS: process.env.OMP_NUM_THREADS || "1",
      VECLIB_MAXIMUM_THREADS: "1",
      OPENBLAS_NUM_THREADS: "1",
      GOTO_NUM_THREADS: "1",
      MKL_NUM_THREADS: "1",
    },
    detached: process.platform !== "win32",
  });
  aiProc.on("error", (err) => {
    log("--- [CRITICAL AI CORE SPAWN ERROR] ---", String(err?.stack||err?.message||err));
    aiReady=false;
    model_status={status:"error",message:"Fatal: Failed to start AI Core. Check main.log."};
    broadcastStatus();
  });
  aiProc.stderr.on("data", (d) => {
    const line=String(d).trim();
    log(`[AI Core STDERR]: ${line}`);
    if(/bad interpreter/i.test(line)){
      log("[main] Python venv broken."); stopAIServer(); aiReady=false;
      model_status={status:"error",message:"Fatal: Python environment is broken."}; broadcastStatus();
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

async function isCoreHealthy() {
  try { return await ping(CORE_HEALTH); } catch { return false; }
}

function escapeForHtml(s = "") { return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/`/g,"\\`");}

function splashHTML(msg = "Initializing…") {
  const safe = escapeForHtml(msg);
  return `<!doctype html><html><head><meta charset="utf-8"/><title>Blur</title><style>
  html,body{height:100%;margin:0;background:#000;color:#fff}
  .mono{font-family:'Courier New',Menlo,Monaco,Consolas,monospace}
  .wrap{display:flex;align-items:center;justify-content:center;height:100%;flex-direction:column}
  .content{text-align:center}
  .dim{color:#aaa}.ok{color:#9efc9e}.warn{color:#ffd27a}.err{color:#ff8c8c}
  
  @keyframes load { 0% { width: 0%; } 100% { width: 100%; } }
  @keyframes pulse {
    0% { box-shadow: 0 0 5px #00e5ff, 0 0 10px #00e5ff; }
    50% { box-shadow: 0 0 15px #00e5ff, 0 0 25px #00e5ff; }
    100% { box-shadow: 0 0 5px #00e5ff, 0 0 10px #00e5ff; }
  }
  .loader-bar {
    width: 200px;
    height: 4px;
    background-color: rgba(0, 229, 255, 0.15);
    border-radius: 4px;
    margin: 20px auto 0 auto;
    overflow: hidden;
  }
  .loader-bar-inner {
    height: 100%;
    width: 100%;
    background-color: #00e5ff;
    border-radius: 4px;
    animation: load 60s linear infinite, pulse 1.5s ease-in-out infinite;
  }
  </style></head><body><div class="wrap">
    <div class="content">
      <div class="mono" style="font-size:22px;margin-bottom:10px">Blur</div>
      <div id="status" class="mono dim" style="font-size:13px">${safe}</div>
    </div>
    <div class="loader-bar"><div class="loader-bar-inner"></div></div>
  </div></body></html>`;
}

async function loadSplash(win, msg) {
  const dataUrl = "data:text/html;charset=utf-8," + encodeURIComponent(splashHTML(msg));
  try { await win.loadURL(dataUrl); } catch {}
}
function ensureOverlay(win) {
  if (!SHOW_OVERLAY) return;
  const js = `(function(){try{var o=document.getElementById('__blur_overlay');if(!o){o=document.createElement('div');o.id='__blur_overlay';o.style.cssText='position:fixed;left:8px;top:8px;z-index:2147483647;background:rgba(0,0,0,.65);color:#fff;padding:6px 8px;border-radius:6px;font:12px Menlo,monospace;pointer-events:none;display:none;';document.body.appendChild(o);}
window.__blurSetOverlay=function(t){try{o.style.display='block';o.textContent=t;}catch(e){}};
}catch(e){}})();`;
  win.webContents.executeJavaScript(js).catch(()=>{});
}
function setOverlayText(win, txt) {
  if (!SHOW_OVERLAY || !win || win.isDestroyed()) return;
  const safe = JSON.stringify(String(txt || ""));
  win.webContents.executeJavaScript(`window.__blurSetOverlay && window.__blurSetOverlay(${safe})`).catch(()=>{});
}
function broadcastStatus() {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("ai-status-update", model_status);
    const cls = model_status.status === "ready" ? "ok" : model_status.status === "degraded" ? "warn" : "err";
    const safeMsg = escapeForHtml(model_status.message || "");
    mainWindow.webContents.executeJavaScript(`(function(){var e=document.getElementById('status');if(e){e.className='mono ${cls}';e.textContent='${safeMsg}';}})();`).catch(()=>{});
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
    },
  });
  mainWindow.on("ready-to-show", () => { mainWindow.show(); });
  try { app.dock?.show(); } catch {}

  mainWindow.webContents.on("console-message", (_e, level, message) => log(`[renderer:L${level}] ${message}`));
  mainWindow.webContents.on("did-finish-load", () => { log("[nav] finished"); ensureOverlay(mainWindow); });
  
  await loadSplash(mainWindow, model_status.message);

  if (!app.isPackaged) {
    await mainWindow.loadURL(DEV_URL).catch(e => log("[dev] loadURL error:", e));
    mainWindow.webContents.openDevTools({ mode: "detach" });
  } else {
    await mainWindow.loadFile(join(__dirname, "../dist/index.html")).catch(e => log("[main] loadFile error:", e));
  }

  mainWindow.webContents.setWindowOpenHandler(({ url }) => { shell.openExternal(url); return { action: "deny" }; });
  mainWindow.on("closed", () => { mainWindow = null; });
}

/* ============================================================================
   IPC: core info
============================================================================ */
ipcMain.handle("core:getInfo", () => ({ ready: !!aiReady, status: model_status }));
ipcMain.handle("core:healthz", async () => ({ ok: await isCoreHealthy(), status: model_status }));

/* ============================================================================
   POWER/SLEEP HOOKS
============================================================================ */
const flushRendererNow = () => {
  if (mainWindow && !mainWindow.isDestroyed()) {
    log("[power] requesting flush from renderer (suspend/lock)");
    mainWindow.webContents.send("main-process-quitting");
  }
};
powerMonitor.on("suspend", () => { log("[power] suspend detected"); flushRendererNow(); });
powerMonitor.on("lock-screen", () => { log("[power] lock-screen detected"); flushRendererNow(); });

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
    await createWindow();
    
    const cfgOK = requireConfigOrFail();

    if (cfgOK) {
      startAIServer(true);
      model_status = { status: "starting", message: "Initializing AI Core…" };
      broadcastStatus();

      (async () => {
        const started = Date.now();
        while (Date.now() - started < 60000) {
          if (await isCoreHealthy()) {
            aiReady = true;
            model_status = { status: "ready", message: "AI Core ready" };
            broadcastStatus();
            return;
          }
          await new Promise(r => setTimeout(r, 2000));
        }
        if (model_status.status === "starting") {
          model_status = { status: "degraded", message: "AI Core startup timeout." };
          broadcastStatus();
        }
      })();
    } else {
      broadcastStatus(); // Show the config error
    }
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

    setTimeout(() => {
      if (isNonEmptyThreads(finalThreadsPayload)) {
        log("[quit] Saving final threads state…");
        writeThreadsFileAtomic(finalThreadsPayload);
      } else {
        log("[quit] No final payload; preserving existing threads file if any.");
      }
      stopAIServer();
      app.quit();
    }, 400);
    return;
  }

  stopAIServer();
  app.quit();
};

app.on("before-quit", (e) => {
  log("[quit] 'before-quit' triggered.");
  if (!isQuitting) { e.preventDefault(); cleanQuit(); }
});
app.on("window-all-closed", () => { if (process.platform !== "darwin") cleanQuit(); });
app.on("activate", () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });

process.on("uncaughtException", (err) => log("[uncaughtException]", err?.stack || String(err)));
process.on("unhandledRejection", (reason) => log("[unhandledRejection]", reason?.stack || String(reason)));