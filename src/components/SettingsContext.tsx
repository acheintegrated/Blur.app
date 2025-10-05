// REFORGED v6.0 — Removed userName from Core Settings
import React, {
  useEffect,
  useState,
  createContext,
  useContext,
  useCallback,
  useRef,
  useMemo,
  type ReactNode,
} from "react";

// ---------- Types ----------
// ✅ REMOVED: `userName` from the interface
export interface Settings {
  theme: "dark";
  interfaceFont: string;
  bodyFont: string;
  glowEffects: boolean;
  animations: boolean;
  tone: "dream" | "astrofuck";
  defaultMode: "dream" | "astrofuck";
  ragAutoIngest: boolean;
  audience: "public" | "internal";
  lastRoute?: string;
}

const DEFAULTS: Settings = {
  theme: "dark",
  interfaceFont: "Courier New",
  bodyFont: "Courier New",
  glowEffects: true,
  animations: true,
  // ✅ REMOVED: userName
  tone: "dream",
  defaultMode: "dream",
  ragAutoIngest: true,
  audience: "public",
  lastRoute: undefined,
};

const SETTINGS_KEY = "settings.v1";
const LOCALSTORAGE_KEY = "acheintegrated-settings";

type Ctx = {
  settings: Settings;
  updateSettings: (patch: Partial<Settings>) => Promise<void>;
  // ✅ REMOVED: setUsername
  setAudience: (a: "public" | "internal") => Promise<void>;
  setLastRoute: (route?: string) => Promise<void>;
  setWants: (
    w: Partial<Pick<Settings, "tone" | "defaultMode" | "ragAutoIngest">>
  ) => Promise<void>;
  applyTheme: () => void;
  applyFonts: () => void;
};

const SettingsContext = createContext<Ctx | undefined>(undefined);

// ---------- Prefs bridge ----------
function getPrefsHandle(): any | null {
  const w: any = window as any;
  return w?.prefs || w?.electron?.prefs || null;
}

async function bridgeHasIO(timeoutMs = 3000): Promise<boolean> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const p = getPrefsHandle();
    if (!p) break;
    if ((p.get && p.set) || (p.load && p.save)) return true;
    await new Promise((r) => setTimeout(r, 50));
  }
  return false;
}

async function loadFromPrefsSafe(): Promise<Partial<Settings> | null> {
  try {
    const p = getPrefsHandle();
    if (p) {
      if (p.get) {
        const raw = await p.get(SETTINGS_KEY, null);
        if (raw && typeof raw === "object") return filterShape(raw);
      } else if (p.load) {
        const raw = await p.load(SETTINGS_KEY);
        if (raw && typeof raw === "object") return filterShape(raw);
      }
    }
    const ls = localStorage.getItem(LOCALSTORAGE_KEY);
    if (!ls) return null;
    const parsed = JSON.parse(ls);
    if (parsed && typeof parsed === "object") return filterShape(parsed);
    return null;
  } catch (e) {
    console.warn("[settings] load error:", e);
    return null;
  }
}

async function persistToPrefsSafe(s: Settings): Promise<void> {
  try {
    const p = getPrefsHandle();
    if (p) {
      if (p.set) return void (await p.set(SETTINGS_KEY, s));
      if (p.save) return void (await p.save(SETTINGS_KEY, s));
    }
    localStorage.setItem(LOCALSTORAGE_KEY, JSON.stringify(s));
  } catch (e) {
    console.warn("[settings] persist error:", e);
  }
}

// ---------- Shape + invariants ----------
function filterShape(raw: any): Partial<Settings> {
  // ✅ REMOVED: userName from the filtered object
  return {
    theme: raw.theme,
    interfaceFont: raw.interfaceFont,
    bodyFont: raw.bodyFont,
    glowEffects: raw.glowEffects,
    animations: raw.animations,
    tone: raw.tone,
    defaultMode: raw.defaultMode,
    ragAutoIngest: raw.ragAutoIngest,
    audience: raw.audience,
    lastRoute: raw.lastRoute,
  };
}

function enforceInvariants(s: Settings): Settings {
  // ✅ REMOVED: userName cleaning
  return {
    ...s,
    theme: "dark",
    glowEffects: true,
    animations: true,
  };
}

function deepEqual(a: any, b: any): boolean {
  if (a === b) return true;
  if (!a || !b || typeof a !== "object" || typeof b !== "object") return false;
  const ka = Object.keys(a);
  const kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  for (const k of ka) {
    if (!kb.includes(k)) return false;
    if (!deepEqual(a[k], b[k])) return false;
  }
  return true;
}

// ---------- DOM helpers ----------
function applyThemeStatic() {
  document.documentElement.classList.add("dark-theme");
  document.documentElement.classList.remove("light-theme");
  document.body.className = "dark-theme";
}
function applyFontsStatic(s: Settings) {
  document.documentElement.style.setProperty("--interface-font", s.interfaceFont);
  document.documentElement.style.setProperty("--body-font", s.bodyFont);
  document.body.style.fontFamily = s.bodyFont || "inherit";
}

// ---------- Provider ----------
export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, _setSettings] = useState<Settings>(DEFAULTS);
  const [loaded, setLoaded] = useState(false);

  const settingsRef = useRef<Settings>(DEFAULTS);
  const lastSavedJSONRef = useRef<string>(JSON.stringify(DEFAULTS));
  const saveTimerRef = useRef<number | null>(null);

  useEffect(() => {
    settingsRef.current = settings;
  }, [settings]);

  useEffect(() => {
    (async () => {
      await bridgeHasIO().catch(() => {});
      const loadedSettings = await loadFromPrefsSafe();
      const merged = enforceInvariants({ ...DEFAULTS, ...(loadedSettings || {}) });
      _setSettings(merged);
      lastSavedJSONRef.current = JSON.stringify(merged);
      setLoaded(true);
      applyFontsStatic(merged);
      applyThemeStatic();
    })();
  }, []);

  useEffect(() => {
    if (!loaded) return;
    const json = JSON.stringify(settings);
    if (json === lastSavedJSONRef.current) return;

    if (saveTimerRef.current) {
      window.clearTimeout(saveTimerRef.current);
      saveTimerRef.current = null;
    }
    saveTimerRef.current = window.setTimeout(async () => {
      try {
        await persistToPrefsSafe(settingsRef.current);
        lastSavedJSONRef.current = JSON.stringify(settingsRef.current);
      } finally {
        if (saveTimerRef.current) {
          window.clearTimeout(saveTimerRef.current);
          saveTimerRef.current = null;
        }
      }
    }, 200);

    return () => {
      if (saveTimerRef.current) {
        window.clearTimeout(saveTimerRef.current);
        saveTimerRef.current = null;
      }
    };
  }, [settings, loaded]);

  useEffect(() => {
    const handler = () => {
      try {
        const json = JSON.stringify(settingsRef.current);
        if (json !== lastSavedJSONRef.current) {
          localStorage.setItem(LOCALSTORAGE_KEY, json);
          lastSavedJSONRef.current = json;
        }
      } catch {}
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, []);

  const commit = useCallback((next: Settings) => {
    next = enforceInvariants(next);
    _setSettings((prev) => (deepEqual(prev, next) ? prev : next));
  }, []);

  const updateSettings = useCallback(async (patch: Partial<Settings>) => {
    commit({ ...settingsRef.current, ...patch });
  }, [commit]);

  // ✅ REMOVED: setUsername function
  const setAudience = useCallback(
    async (aud: "public" | "internal") => updateSettings({ audience: aud }),
    [updateSettings]
  );

  const setLastRoute = useCallback(
    async (route?: string) => updateSettings({ lastRoute: route }),
    [updateSettings]
  );

  const setWants = useCallback(
    async (
      w: Partial<Pick<Settings, "tone" | "defaultMode" | "ragAutoIngest">>
    ) => updateSettings(w),
    [updateSettings]
  );

  const applyTheme = useCallback(() => applyThemeStatic(), []);
  const applyFonts = useCallback(() => applyFontsStatic(settingsRef.current), []);

  // ✅ REMOVED: setUsername from context value
  const value = useMemo<Ctx>(
    () => ({
      settings,
      updateSettings,
      setAudience,
      setLastRoute,
      setWants,
      applyTheme,
      applyFonts,
    }),
    [settings, updateSettings, setAudience, setLastRoute, setWants, applyTheme, applyFonts]
  );

  if (!loaded) return null; // Simplified loading state

  return (
    <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>
  );
}

// ---------- Hook ----------
export function useSettings() {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error("useSettings must be used within a SettingsProvider");
  return ctx;
}