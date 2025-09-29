// /opt/blurface/src/AppRouter.tsx
import React from "react";
import { BrowserRouter, HashRouter, Routes, Route } from "react-router-dom";
import { App } from "./App"; // <-- named import

const isProd = import.meta.env.PROD;

export function AppRouter() {
  const Router: React.ComponentType<React.PropsWithChildren> =
    isProd ? HashRouter : BrowserRouter;

  return (
    <Router>
      <Routes>
        <Route path="/*" element={<App />} />
      </Routes>
    </Router>
  );
}

export default AppRouter;
