import { create } from "zustand";
import { persist } from "zustand/middleware";

export type Wants = {
  tone?: "dream" | "astrofuck";
  defaultMode?: "dream" | "astrofuck";
  ragAutoIngest?: boolean;
};

export type PrefsState = {
  username?: string | null;
  audience?: "public" | "internal";
  lastRoute?: string;
  wants: Wants;

  setUsername: (n?: string | null) => void;
  setAudience: (a: "public" | "internal") => void;
  setLastRoute: (route?: string) => void;
  setWants: (w: Partial<Wants>) => void;
};

// Single, canonical export:
export const usePrefs = create<PrefsState>()(
  persist(
    (set) => ({
      username: undefined,
      audience: "public",
      lastRoute: undefined,
      wants: { tone: "dream", defaultMode: "dream", ragAutoIngest: true },

      setUsername: (username) => set({ username }),
      setAudience: (audience) => set({ audience }),
      setLastRoute: (lastRoute) => set({ lastRoute }),
      setWants: (w) => set((s) => ({ wants: { ...s.wants, ...w } })),
    }),
    { name: "blur:prefs" }
  )
);
