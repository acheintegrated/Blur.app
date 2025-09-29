// /opt/blurface/src/components/ContextMenu.tsx
import React, { useEffect, useRef } from "react";

interface Props {
  x: number;
  y: number;
  onClose: () => void;
  onDelete: () => void;
  onRename: () => void;
}

export const ContextMenu: React.FC<Props> = ({ x, y, onClose, onDelete, onRename }) => {
  const ref = useRef<HTMLDivElement>(null);

  // close on outside click / right-click elsewhere / escape
  useEffect(() => {
    const onDocMouseDown = (e: MouseEvent) => {
      if (!ref.current) return;
      if (!ref.current.contains(e.target as Node)) onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("mousedown", onDocMouseDown);
    document.addEventListener("contextmenu", onDocMouseDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocMouseDown);
      document.removeEventListener("contextmenu", onDocMouseDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [onClose]);

  return (
    <div
      id="context-menu"
      ref={ref}
      className="fixed z-50 bg-black/95 border border-zinc-700 rounded shadow-2xl py-1 min-w-[160px] backdrop-blur-sm select-none"
      style={{ left: x, top: y }}
      onMouseDown={(e) => e.stopPropagation()}
      onClick={(e) => e.stopPropagation()}
    >
      <button
        className="w-full text-left px-3 py-2 text-sm text-gray-300 hover:bg-zinc-800 hover:text-white transition-colors"
        onClick={(e) => {
          e.preventDefault();
          onRename();
        }}
      >
        rename
      </button>

      <button
        className="w-full text-left px-3 py-2 text-sm text-red-400 hover:bg-red-900/20 hover:text-red-300 transition-colors"
        onClick={(e) => {
          e.preventDefault();
          onDelete();
        }}
      >
        delete
      </button>

      <hr className="border-zinc-700 my-1" />

      <button
        className="w-full text-left px-3 py-2 text-sm text-gray-400 hover:bg-zinc-800 hover:text-white transition-colors"
        onClick={(e) => {
          e.preventDefault();
          onClose();
        }}
      >
        Cancel
      </button>
    </div>
  );
};
