// /electron/preload.mjs — v9.9 (idempotent • IPC-first • fires DOM readiness events)
import { contextBridge, ipcRenderer } from "electron";

// ---- idempotency guard (prevents double wiring on HMR/resume) ----
if (!globalThis.__blur_preload_loaded) {
  globalThis.__blur_preload_loaded = true;

  /* ========================= utils ========================= */
  const safeInvoke = async (ch, ...args) => {
    try { return await ipcRenderer.invoke(ch, ...args); }
    catch (err) {
      console.warn(`[preload] invoke(${ch}) failed:`, err?.message || String(err));
      return null;
    }
  };
  const deepClone = (v) => {
    try { return (v && typeof v === "object") ? structuredClone(v) : v; }
    catch { try { return JSON.parse(JSON.stringify(v)); } catch { return v; } }
  };
  const dispatch = (name, detail) => {
    try { window.dispatchEvent(new CustomEvent(name, { detail })); } catch {}
  };

  /* ========================= prefs/env/active ========================= */
  const THREADS_KEY = "threads.v1";

  const prefsBridge = Object.freeze({
    get: (key, fallback = null) => safeInvoke("prefs:get", key ?? null, fallback),
    set: (key, value) => safeInvoke("prefs:set", key, value),
    has: (key) => safeInvoke("prefs:has", key),
    del: (key) => safeInvoke("prefs:delete", key),
    all: () => safeInvoke("prefs:get", null, null),
  });

  const envBridge = Object.freeze({
    get: (key) => safeInvoke("env:get", key),
  });

  const activeBridge = Object.freeze({
    load: () => safeInvoke("prefs:get", "activeThreadId", null),
    save: (id) => safeInvoke("prefs:set", "activeThreadId", id ?? null),
  });

  /* ========================= validators ========================= */
  const isMsg = (m) => m && typeof m === "object"
    && (m.sender === "Blur" || m.sender === "You" || m.sender === "System")
    && typeof m.text === "string";

  const isThread = (t) => t && typeof t === "object"
    && typeof t.id === "string"
    && typeof t.title === "string"
    && Array.isArray(t.messages) && t.messages.every(isMsg);

  /* ========================= threads (IPC-first, prefs fallback) ========================= */
  const threadsFacade = Object.freeze({
    load: async () => {
      let arr = await safeInvoke("threads:load");
      if (!Array.isArray(arr)) arr = await prefsBridge.get(THREADS_KEY, []);
      return Array.isArray(arr) ? arr.filter(isThread).map(deepClone) : [];
    },
    save: async (arr) => {
      if (!Array.isArray(arr)) return false;
      const normalized = arr.filter(isThread);
      const ok = await safeInvoke("threads:save", deepClone(normalized));
      if (ok !== null) return !!ok;                // IPC path present
      await prefsBridge.set(THREADS_KEY, deepClone(normalized)); // legacy fallback
      return true;
    },
  });

  /* ========================= AI status / core ========================= */
  const aiStatusBridge = Object.freeze({
    onStatusUpdate: (cb) => {
      if (typeof cb !== "function") return () => {};
      const handler = (_e, payload) => { try { cb(payload); } catch {} };
      ipcRenderer.on("ai-status-update", handler);
      return () => ipcRenderer.removeListener("ai-status-update", handler);
    },
    removeAll: () => ipcRenderer.removeAllListeners("ai-status-update"),
  });

  const coreBridge = Object.freeze({
    getInfo: () => safeInvoke("core:getInfo"),
    healthz: () => safeInvoke("core:healthz"), // main handles health checks; no CORS/no console spam
  });

  // --- NEW: Internal wiring to fire DOM events for UI smoothing ---------
  // We forward ai-status -> DOM so renderer components can react without importing ipcRenderer.
  // Events:
  //   blur:core-status            { status, message? }
  //   blur:mode-switch-start      { status }
  //   blur:mode-ready             { status }
  //   blur:mode-switch-error      { status, message? }
  const aiStatusToDom = (_e, payload) => {
    const status = payload?.status;
    if (!status) return;
    dispatch('blur:core-status', payload);
    if (status === 'loading_model' || status === 'connecting' || status === 'initializing') {
      dispatch('blur:mode-switch-start', payload);
    }
    if (status === 'ready') {
      dispatch('blur:mode-ready', payload);
    }
    if (status === 'error') {
      dispatch('blur:mode-switch-error', payload);
    }
  };
  ipcRenderer.on('ai-status-update', aiStatusToDom);

  /* ========================= error taps → main ========================= */
  window.addEventListener("error", (e) => {
    try {
      ipcRenderer.send("renderer:error", {
        message: String(e?.error?.stack || e?.message || e),
        source: e?.filename, lineno: e?.lineno, colno: e?.colno
      });
    } catch {}
  });
  window.addEventListener("unhandledrejection", (e) => {
    try { ipcRenderer.send("renderer:unhandledRejection", { reason: String(e?.reason?.stack || e?.reason || "") }); } catch {}
  });

  /* ========================= quit + lifecycle flush ========================= */
  const grabLatestThreads = async () => {
    // prefer in-memory from renderer (fastest), fallback to disk snapshot
    try {
      const mem = globalThis.__BLUR_GET_LATEST_THREADS?.();
      if (Array.isArray(mem) && mem.length) return deepClone(mem);
    } catch {}
    try {
      const disk = await threadsFacade.load();
      if (Array.isArray(disk) && disk.length) return deepClone(disk);
    } catch {}
    return [];
  };

  const flushThreads = async () => {
    try {
      const payload = await grabLatestThreads();
      if (Array.isArray(payload)) await threadsFacade.save(payload);
    } catch {}
  };

  // main requests final state (on quit or suspend); reply immediately
  ipcRenderer.on("main-process-quitting", async () => {
    try {
      const latest = await grabLatestThreads();
      await ipcRenderer.invoke("threads:send-final-state-for-quit", latest);
    } catch {}
  });

  // also flush on lifecycle edges (sleep/tab discard)
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") void flushThreads();
  });
  window.addEventListener("pagehide", () => { void flushThreads(); });
  // Chromium "freeze" (optional)
  // @ts-ignore
  window.addEventListener?.("freeze", () => { void flushThreads(); });

  /* ========================= expose ========================= */
  const api = Object.freeze({
    prefs: prefsBridge,
    env: envBridge,
    threads: threadsFacade,
    aiStatus: aiStatusBridge,
    core: coreBridge,
    active: activeBridge,
  });

  const expose = (name, obj) => { try { contextBridge.exposeInMainWorld(name, obj); } catch {} };
  expose("prefs", prefsBridge);
  expose("env", envBridge);
  expose("core", coreBridge);
  expose("electron", api);
  expose("api", api);   // legacy alias
  expose("g", api);     // legacy alias
  expose("active", activeBridge);
}
