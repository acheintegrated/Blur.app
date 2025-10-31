// /opt/blurface/src/lib/prefsClient.ts
export type PrefsPayload = {
  username?: string | null;
  wants?: Record<string, unknown>;
};

// Resolve the AI core base URL safely at call time
async function coreBase(): Promise<string> {
  try {
    const port =
      (await (window as any)?.env?.get?.("BLUR_CORE_PORT")) ||
        process.env?.BLUR_CORE_PORT ||
        "25421";
    return `http://127.0.0.1:${String(port)}`;
  } catch {
    return "http://127.0.0.1:25421";
  }
}

export async function fetchPrefs(): Promise<PrefsPayload> {
  const base = await coreBase();
  try {
    const r = await fetch(`${base}/prefs`, { credentials: "include" });
    if (!r.ok) return {};
    const data = (await r.json()) as PrefsPayload;
    return data ?? {};
  } catch {
    return {};
  }
}

export async function savePrefs(body: PrefsPayload): Promise<void> {
  const base = await coreBase();
  try {
    await fetch(`${base}/prefs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify(body ?? {}),
    });
  } catch {
    // swallow errors; backend prefs are optional
  }
}
