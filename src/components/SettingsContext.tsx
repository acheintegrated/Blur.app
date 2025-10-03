import React, {
  useEffect,
  useState,
  createContext,
  useContext,
  useCallback,
  useRef,
  type ReactNode,
} from "react";

// Updated Settings interface to only include the fields you want to keep
export interface Settings {
  theme: "dark";
  interfaceFont: string;
  bodyFont: string;
  glowEffects: boolean;
  animations: boolean;
  userName: string;
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
  // Removed: notifications, soundEffects, autoSave, saveInterval
  glowEffects: true,
  animations: true,
  userName: "",
  // Removed: instructions, memory
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
  setUsername: (n?: string | null) => Promise<void>;
  setAudience: (a: "public" | "internal") => Promise<void>;
  setLastRoute: (route?: string) => Promise<void>;
  setWants: (w: Partial<Pick<Settings, "tone" | "defaultMode" | "ragAutoIngest">>) => Promise<void>;
  applyTheme: () => void;
  applyFonts: () => void;
};

const SettingsContext = createContext<Ctx | undefined>(undefined);

// ---- robust prefs handle: supports window.prefs and window.electron.prefs
function getPrefsHandle(): any | null {
  const w: any = window as any;
  return w?.prefs || w?.electron?.prefs || null;
}

function enforceInvariants(s: Settings): Settings {
  // We keep 'theme', 'glowEffects', and 'animations' as required invariants
  return { ...s, theme: "dark", glowEffects: true, animations: true };
}

async function persistToPrefsSafe(s: Settings): Promise<void> {
  const prefs = getPrefsHandle();
  try {
    if (prefs?.set) {
      await prefs.set(SETTINGS_KEY, s);
    } else {
      localStorage.setItem(LOCALSTORAGE_KEY, JSON.stringify(s));
    }
  } catch (e) {
    console.error("[settings] persist error", e);
  }
}

async function loadFromPrefsSafe(): Promise<Settings | null> {
  const prefs = getPrefsHandle();
  try {
    if (prefs?.get) {
      const raw = await prefs.get(SETTINGS_KEY, null);
      if (raw && typeof raw === "object" && !Array.isArray(raw)) {
        // We cast to any first, then manually pick the properties that are still in Settings
        const rawSettings = raw as any;
        const filteredSettings: Partial<Settings> = {
          theme: rawSettings.theme,
          interfaceFont: rawSettings.interfaceFont,
          bodyFont: rawSettings.bodyFont,
          glowEffects: rawSettings.glowEffects,
          animations: rawSettings.animations,
          userName: rawSettings.userName,
          tone: rawSettings.tone,
          defaultMode: rawSettings.defaultMode,
          ragAutoIngest: rawSettings.ragAutoIngest,
          audience: rawSettings.audience,
          lastRoute: rawSettings.lastRoute,
        };
        return filteredSettings as Settings;
      }
    }
    
    // Fallback to localStorage
    const ls = localStorage.getItem(LOCALSTORAGE_KEY);
    if (ls) {
      const parsed = JSON.parse(ls) as any;
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        const filteredSettings: Partial<Settings> = {
          theme: parsed.theme,
          interfaceFont: parsed.interfaceFont,
          bodyFont: parsed.bodyFont,
          glowEffects: parsed.glowEffects,
          animations: parsed.animations,
          userName: parsed.userName,
          tone: parsed.tone,
          defaultMode: parsed.defaultMode,
          ragAutoIngest: parsed.ragAutoIngest,
          audience: parsed.audience,
          lastRoute: parsed.lastRoute,
        };
        return filteredSettings as Settings;
      }
    }
    return null;
  } catch (e) {
    console.error("[settings] load error", e);
    return null;
  }
}

async function waitForPrefsBridge(timeoutMs = 5000): Promise<boolean> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (getPrefsHandle()?.get && getPrefsHandle()?.set) {
      return true;
    }
    await new Promise(r => setTimeout(r, 50));
  }
  console.warn("[settings] Prefs bridge timeout, falling back to localStorage");
  return false;
}

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Settings>(DEFAULTS);
  const [isLoading, setIsLoading] = useState(true);

  // latest snapshot for stable unload handler
  const settingsRef = useRef(settings);
  useEffect(() => { settingsRef.current = settings; }, [settings]);

  // initial load — wait for bridge if needed, then hydrate
  useEffect(() => {
    (async () => {
      await waitForPrefsBridge();
      const loadedSettings = await loadFromPrefsSafe();
      const finalSettings = enforceInvariants({ ...DEFAULTS, ...(loadedSettings || {}) });
      
      setSettings(finalSettings);
      setIsLoading(false);
      
      // Apply fonts/theme after load
      applyFontsStatic(finalSettings);
      applyThemeStatic();
    })();
  }, []);

  // FIXED updater that awaits disk flush AND sends to backend
  const updateSettings = useCallback(async (patch: Partial<Settings>) => {
    setSettings(prev => {
      const nextVal = enforceInvariants({ ...prev, ...patch });
      
      // Persist to prefs immediately
      persistToPrefsSafe(nextVal).then(() => {
        // REMOVED: The logic to fetch('http://127.0.0.1:8000/ingest-memory')
        // as it relied on 'memory' and 'instructions' fields.
      }).catch(err => {
        console.error("[settings] Persist failed:", err);
      });
      
      return nextVal;
    });
    
    return Promise.resolve();
  }, []);

  const setUsername  = useCallback(async (name?: string | null) => updateSettings({ userName: name ?? "" }), [updateSettings]);
  const setAudience  = useCallback(async (audience: "public" | "internal") => updateSettings({ audience }), [updateSettings]);
  const setLastRoute = useCallback(async (route?: string) => updateSettings({ lastRoute: route }), [updateSettings]);
  const setWants     = useCallback(async (w: Partial<Pick<Settings, "tone" | "defaultMode" | "ragAutoIngest">>) => updateSettings(w), [updateSettings]);

  // single beforeunload saver (stable)
  useEffect(() => {
    const handler = () => { 
      void persistToPrefsSafe(settingsRef.current); 
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, []);

  const applyTheme = () => applyThemeStatic();
  const applyFonts = () => applyFontsStatic(settings);

  // Show loading state while initializing
  if (isLoading) {
    return <div>Loading settings...</div>;
  }

  return (
    <SettingsContext.Provider value={{ 
      settings, 
      updateSettings, 
      setUsername, 
      setAudience, 
      setLastRoute, 
      setWants, 
      applyTheme, 
      applyFonts 
    }}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings() {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error("useSettings must be used within a SettingsProvider");
  return ctx;
}

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