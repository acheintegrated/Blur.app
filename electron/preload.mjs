// /electron/preload.mjs — REFORGED v9.3
import { contextBridge, ipcRenderer } from "electron";

/* ---------------- utils ---------------- */
const safeInvoke = async (ch, ...args) => {
  try { return await ipcRenderer.invoke(ch, ...args); }
  catch (err) {
    console.warn(`[preload] invoke(${ch}) failed:`, err?.message || String(err));
    return null;
  }
};
const deepClone = (v) => (v && typeof v === "object" ? structuredClone(v) : v);

/* ---------------- prefs ---------------- */
const THREADS_KEY = "threads.v1";
const prefsBridge = {
  get: (key, fallback = null) => safeInvoke("prefs:get", key ?? null, fallback),
  set: (key, value) => safeInvoke("prefs:set", key, value),
  has: (key) => safeInvoke("prefs:has", key),
  del: (key) => safeInvoke("prefs:delete", key),
  all: () => safeInvoke("prefs:get", null, null),
};
const envBridge = { get: (key) => safeInvoke("env:get", key) };
const threadsFacade = {
  load: async () => {
    const arr = await prefsBridge.get(THREADS_KEY, []);
    return Array.isArray(arr) ? deepClone(arr) : [];
  },
  save: async (arr) => {
    if (!Array.isArray(arr)) return false;
    await prefsBridge.set(THREADS_KEY, deepClone(arr));
    return true;
  },
};

/* --------------- AI status --------------- */
const aiStatusBridge = {
  onStatusUpdate: (cb) => {
    if (typeof cb !== "function") return () => {};
    const handler = (_e, payload) => { try { cb(payload); } catch {} };
    ipcRenderer.on("ai-status-update", handler);
    return () => ipcRenderer.removeListener("ai-status-update", handler);
  },
  onceStatusUpdate: (cb) => {
    if (typeof cb !== "function") return () => {};
    const handler = (_e, payload) => cb(payload);
    ipcRenderer.once("ai-status-update", handler);
    return () => ipcRenderer.removeListener("ai-status-update", handler);
  },
  removeAll: () => ipcRenderer.removeAllListeners("ai-status-update"),
};

/* --------------- Core bridge --------------- */
const coreBridge = {
  getInfo: () => safeInvoke("core:getInfo"),
  onPortBump: (cb) => {
    if (typeof cb !== "function") return () => {};
    const handler = (_e, payload) => { try { cb(payload); } catch {} };
    ipcRenderer.on("core:port-bump", handler);
    return () => ipcRenderer.removeListener("core:port-bump", handler);
  },
};

/* --------------- Error tap -> main --------------- */
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

/* --------------- Expose --------------- */
const api = { prefs: prefsBridge, env: envBridge, threads: threadsFacade, aiStatus: aiStatusBridge, core: coreBridge };

contextBridge.exposeInMainWorld("prefs", prefsBridge);
contextBridge.exposeInMainWorld("env", envBridge);
contextBridge.exposeInMainWorld("core", coreBridge);
contextBridge.exposeInMainWorld("electron", api);

// legacy aliases (some UIs expect these)
try { contextBridge.exposeInMainWorld("api", api); } catch {}
try { contextBridge.exposeInMainWorld("g", api); } catch {}
