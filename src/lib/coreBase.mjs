// coreBase.mjs — Single source of truth for AI Core base URL

// provided by preload
const coreAPI = (typeof window !== 'undefined' && window && window.core) ? window.core : null;

// Use a variable to store the resolved base URL
let resolvedCoreBase = "http://127.0.0.1:25421";

/**
 * Resolves the AI Core base URL, falling back to port 25421.
 * The Electron main process's coreAPI is the preferred source.
 * @returns {Promise<string>} The base URL (e.g., "http://127.0.0.1:25421")
 */
export async function initCoreBase() {
  try {
    const info = await coreAPI?.getInfo?.();
    if (info?.baseUrl) {
      resolvedCoreBase = String(info.baseUrl);
      return resolvedCoreBase;
    }
  } catch {}
  return resolvedCoreBase;
}

/**
 * Returns the currently resolved AI Core base URL.
 * NOTE: Ensure initCoreBase() has been called before using this.
 * @returns {string} The core base URL.
 */
export function getCoreBase() {
  return resolvedCoreBase;
}