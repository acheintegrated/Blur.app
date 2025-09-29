// /opt/blurface/src/components/hooks/useLastRoute.ts

import { useEffect } from "react";
import { useLocation } from "react-router-dom";
// --- FIXED: Import from the unified SettingsContext instead of the old Zustand store ---
import { useSettings } from "../SettingsContext";

export function useLastRoute() {
  const loc = useLocation();
  // --- FIXED: Get the setter function from useSettings ---
  const { setLastRoute } = useSettings();
  
  useEffect(() => {
    setLastRoute?.(loc.pathname + loc.search + loc.hash);
  }, [loc.pathname, loc.search, loc.hash, setLastRoute]);
}