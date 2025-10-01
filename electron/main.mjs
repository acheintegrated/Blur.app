// electron/main.mjs — REFORGED v11.1 (Full Preservation + Robust Packaging)
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
if (!existsSync(canonicalUserData)) mkdirSync(canonicalUserData, { recursive: true });

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
  const line = `[${new Date().toISOString()}] ` + args.map(a => (typeof a === "string" ? a : JSON.stringify(a))).join(" ");
  console.log(line);
  if (logFile) try { appendFileSync(logFile, line + "\n"); } catch {}
};

/* ============================================================================
   BLUR_HOME + RESOURCE UNPACKING (NEW LOGIC)
============================================================================ */
function cpRecursive(src, dst) {
  if (!existsSync(src)) return;
  const stats = statSync(src);
  if (stats.isDirectory()) {
    mkdirSync(dst, { recursive: true });
    for (const file of readdirSync(src)) {
      cpRecursive(join(src, file), join(dst, file));
    }
  } else {
    mkdirSync(dirname(dst), { recursive: true });
    copyFileSync(src, dst);
  }
}

function initializeBlurHome() {
  const homeProdPath = join(app.getPath("userData"), "blur_home");

  if (!app.isPackaged) {
    const devResources = resolve(__dirname, "..", "resources");
    if (!existsSync(devResources)) {
      const msg = "FATAL: `resources` directory not found. Create it and move models, core, etc. there.";
      log(msg);
      model_status = { status: "error", message: msg };
      process.env.BLUR_HOME = devResources;
      return;
    }
    process.env.BLUR_HOME = devResources;
    log(`[dev] BLUR_HOME set to project resources: ${devResources}`);
    return;
  }

  process.env.BLUR_HOME = homeProdPath;
  log(`[prod] BLUR_HOME is: ${homeProdPath}`);

  const versionFile = join(homeProdPath, ".version");
  const currentAppVersion = app.getVersion();
  let existingVersion = null;

  if (existsSync(versionFile)) {
    try { existingVersion = readFileSync(versionFile, "utf8").trim(); } catch {}
  }

  if (currentAppVersion !== existingVersion) {
    log(`Version mismatch (app: ${currentAppVersion}, home: ${existingVersion}). Unpacking resources...`);
    const bundledResourcesPath = join(process.resourcesPath, "resources");
    
    if (existsSync(bundledResourcesPath)) {
      try {
        if (existsSync(homeProdPath)) {
            log(`Removing old BLUR_HOME at ${homeProdPath} for fresh unpack.`);
            fs.rmSync(homeProdPath, { recursive: true, force: true });
        }
        mkdirSync(homeProdPath, { recursive: true });
        cpRecursive(bundledResourcesPath, homeProdPath);
        writeFileSync(versionFile, currentAppVersion);
        log("Successfully unpacked resources to BLUR_HOME.");
      } catch (e) {
        log("FATAL: Failed to unpack resources:", e);
        model_status = { status: "error", message: "Failed to initialize app resources." };
      }
    } else {
      log("FATAL: Bundled 'resources' directory not found inside the app.");
      model_status = { status: "error", message: "App is corrupted; resources are missing." };
    }
  } else {
    log("BLUR_HOME is up to date.");
  }
}

/* ============================================================================
   CONFIG DISCOVERY & PYTHON RESOLUTION (NEW LOGIC)
============================================================================ */
function configureEnvironment() {
    const BLUR_HOME = process.env.BLUR_HOME;
    if (!BLUR_HOME || !existsSync(BLUR_HOME)) {
        log("FATAL: BLUR_HOME is not set or does not exist!");
        model_status = { status: "error", message: "BLUR_HOME environment is not configured." };
        return false;
    }

    const configPath = join(BLUR_HOME, "config.yaml");
    if (existsSync(configPath)) {
        process.env.BLUR_CONFIG_PATH = configPath;
        log(`Using config file: ${configPath}`);
    } else {
        const msg = `FATAL: config.yaml not found in BLUR_HOME (${BLUR_HOME})`;
        log(msg);
        model_status = { status: "error", message: msg };
        return false;
    }

    const pythonVenvPath = join(BLUR_HOME, 'blur_env-darwin-arm64');
    const pythonExecutable = join(pythonVenvPath, 'bin', 'python3');

    if (existsSync(pythonExecutable)) {
        process.env.BLUR_PYTHON_PATH = pythonExecutable;
        log(`Found Python executable: ${pythonExecutable}`);
    } else {
        log(`Python executable not found at expected path: ${pythonExecutable}. Falling back to system 'python3'.`);
        process.env.BLUR_PYTHON_PATH = "python3";
    }
    
    process.env.BLUR_PACKAGED = app.isPackaged ? "1" : "0";
    return true;
}


/* ============================================================================
   PREFS (IPC) - Preserved
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
   THREADS PERSISTENCE (IPC) — Preserved
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

function isNonEmptyThreads(arr) {
  return Array.isArray(arr) && arr.some(t => t && t.messages && t.messages.length);
}

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

let finalThreadsPayload = null;
ipcMain.handle("threads:load", () => readThreadsFile());
ipcMain.handle("threads:save", (_e, payload) => writeThreadsFileAtomic(payload));
ipcMain.handle("threads:send-final-state-for-quit", (_e, payload) => {
  if (isNonEmptyThreads(payload)) {
    finalThreadsPayload = payload;
    log("[quit] final threads payload received.");
  } else {
    log("[quit] ignored final payload (empty)");
  }
});


/* ============================================================================
   AI CORE SPAWN + READINESS - Preserved
============================================================================ */
function startAIServer() {
  if (aiProc) { log("[main] AI Core start ignored: already running."); return; }
  
  if (!configureEnvironment()) {
    broadcastStatus();
    return;
  }

  const pythonCommand = process.env.BLUR_PYTHON_PATH;
  const backendDir = join(__dirname, "backend");
  const uvicornArgs = ["-m", "uvicorn", "convo_chat_core:app", "--host", CORE_HOST, "--port", CORE_PORT, "--log-level", "info"];
  
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
      KMP_DUPLICATE_LIB_OK: "TRUE" // This is the critical fix
    },
    detached: process.platform !== "win32",
  });

  aiProc.on("error", (err) => {
    log("--- [CRITICAL AI CORE SPAWN ERROR] ---", err);
    aiReady=false;
    model_status={status:"error",message:"Fatal: Failed to start AI Core. Check main.log."};
    broadcastStatus();
  });
  aiProc.stderr.on("data", (d) => log(`[AI Core STDERR]: ${String(d).trim()}`));
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
  if (!aiProc) return;
  log("[main] Stopping AI Core…");
  try {
    if (process.platform === "win32") spawn("taskkill", ["/PID", String(aiProc.pid), "/T", "/F"]);
    else process.kill(-aiProc.pid, "SIGTERM");
  } catch {
    try { aiProc.kill("SIGKILL"); } catch {}
  }
  aiProc = null; aiReady = false; model_status = { status: "stopping", message: "AI Core stopping…" }; broadcastStatus();
}

/* ============================================================================
   NET PING & UI - Preserved
============================================================================ */
function ping(urlStr) { return new Promise((resolve) => {
  const lib = urlStr.startsWith("https:") ? https : http;
  const req = lib.get(urlStr, { timeout: 1500 }, (res) => {
    res.resume();
    resolve((res.statusCode || 0) >= 200 && (res.statusCode || 0) < 400);
  });
  req.on("error", () => resolve(false));
  req.on("timeout", () => req.destroy());
});}

async function isCoreHealthy() {
  try { return await ping(CORE_HEALTH); } catch { return false; }
}

function escapeForHtml(s = "") { return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/`/g,"\\`");}

// electron/main.mjs

function splashHTML(msg = "Initializing…") {
  const safe = escapeForHtml(msg);
  return `<!doctype html><html><head><meta charset="utf-8"/><title>Blur</title><style>
  html,body{height:100%;margin:0;background:#000;color:#fff}
  .mono{font-family:'Courier New',Menlo,Monaco,Consolas,monospace}
  .wrap{display:flex;align-items:center;justify-content:center;height:100%;flex-direction:column}
  .content{text-align:center}
  .dim{color:#aaa}.ok{color:#9efc9e}.warn{color:#ffd27a}.err{color:#ff8c8c}
  
  /* --- Animated Loading Bar --- */
  @keyframes load {
    0% { width: 0%; }
    100% { width: 100%; }
  }
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
   WINDOW CREATE - Preserved
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
  mainWindow.webContents.on("render-process-gone", (_e, d) => { log("[renderer gone]", d?.reason); setOverlayText(mainWindow, `renderer gone: ${d?.reason}`); });
  mainWindow.webContents.on("unresponsive", () => { log("[renderer] unresponsive"); setOverlayText(mainWindow, "renderer unresponsive"); });
  
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
   IPC: core info - Preserved
============================================================================ */
ipcMain.handle("core:getInfo", () => ({ ready: !!aiReady, status: model_status }));
ipcMain.handle("core:healthz", async () => ({ ok: await isCoreHealthy(), status: model_status }));

/* ============================================================================
   POWER/SLEEP HOOKS — Preserved
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
   SINGLE INSTANCE & LIFECYCLE - Preserved & Integrated
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

    initializeBlurHome(); // New integrated step
    
    await createWindow();

    if (model_status.status !== "error") {
      startAIServer();
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
      broadcastStatus(); // Show the error from initialization
    }
  });
}

/* ============================================================================
   ROBUST SHUTDOWN - Preserved
============================================================================ */
const cleanQuit = () => {
  if (isQuitting) return;
  isQuitting = true;
  log("[quit] Initiating clean shutdown...");

  if (finalThreadsPayload) {
    log("[quit] Saving final threads state…");
    writeThreadsFileAtomic(finalThreadsPayload);
  } else {
    log("[quit] No final payload received from renderer, skipping save.");
  }
  
  stopAIServer();
  setTimeout(() => {
    log("[quit] Exiting application.");
    app.quit();
  }, 500);
};

app.on("before-quit", (e) => {
  log("[quit] 'before-quit' triggered.");
  if (!isQuitting) {
    e.preventDefault();
    if (mainWindow && !mainWindow.isDestroyed()) {
      log("[quit] Requesting final state from renderer…");
      mainWindow.webContents.send("main-process-quitting");
      setTimeout(cleanQuit, 400); // Wait a moment for IPC to complete
    } else {
      cleanQuit();
    }
  }
});
app.on("window-all-closed", () => { if (process.platform !== "darwin") app.quit(); });
app.on("activate", () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });

process.on("uncaughtException", (err) => log("[uncaughtException]", err?.stack || String(err)));
process.on("unhandledRejection", (reason) => log("[unhandledRejection]", reason?.stack || String(reason)));