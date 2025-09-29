// src/components/ThreadsSection.tsx
import React from "react";

interface Thread {
  id: string;
  title: string;
}

interface RenameState {
  isRenaming: boolean;
  itemId: string;
  newName: string;
  isThread?: boolean;
}

interface ThreadsSectionProps {
  threads: Thread[];
  activeItem: string;
  expandedSections?: { threads: boolean };
  searchQuery?: string;
  renameState?: RenameState;
  renameInputRef?: React.RefObject<HTMLInputElement>;
  onSectionClick: () => void;
  onToggleSection: (e: React.MouseEvent<HTMLDivElement>) => void;
  onItemClick: (id: string) => void;
  onContextMenu?: (e: React.MouseEvent, id: string, isThread: boolean) => void;
  onRenameChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onRenameKeyDown?: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  onRenameBlur?: () => void;
}

const DEFAULT_RENAME: RenameState = {
  isRenaming: false,
  itemId: "",
  newName: "",
  isThread: true,
};

export const ThreadsSection: React.FC<ThreadsSectionProps> = (props) => {
  const {
    threads,
    activeItem,
    expandedSections = { threads: true },
    searchQuery = "",
    renameState,
    renameInputRef,
    onSectionClick,
    onToggleSection,
    onItemClick,
    onContextMenu,
    onRenameChange,
    onRenameKeyDown,
    onRenameBlur,
  } = props;

  const safeRename = renameState ?? DEFAULT_RENAME;

  const filteredThreads = (threads || []).filter((thread) => {
    const title = thread.title || "";
    return title.toLowerCase().includes(searchQuery.toLowerCase());
  });

  const handleSectionClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) onSectionClick();
  };

  const handleItemContextMenu = (e: React.MouseEvent, id: string) => {
    e.preventDefault();
    e.stopPropagation();
    onContextMenu?.(e, id, true);
  };

  const handleItemClick = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    onItemClick(id);
  };

  return (
    <div className="p-4" onClick={handleSectionClick} role="region" aria-label="Threads">
      <div
        className="flex items-center mb-2 px-2 py-1.5 rounded"
        onClick={onToggleSection}
        role="button"
        aria-expanded={expandedSections.threads}
        aria-label={expandedSections.threads ? "Collapse threads" : "Expand threads"}
      >
        <div className="cursor-pointer flex items-center baby-pink-blue-glow-hover transition-all duration-300">
          <span className="mr-2 text-pink-400 icon-font">{expandedSections.threads ? "▼" : "▸"}</span>
          <span className="text-pink-400">threads</span>
        </div>
      </div>

      {expandedSections.threads && (
        <div className="pl-4 space-y-1">
          {filteredThreads.map((thread) => (
            <div key={thread.id} className="cursor-pointer relative" tabIndex={-1}>
              <div
                className={`${activeItem === thread.id ? "text-white" : "text-gray-500"} hover:text-white transition-colors duration-150 hover-drip-effect magenta-purple-glow-hover px-2 py-1 rounded select-none`}
                onClick={(e) => handleItemClick(e, thread.id)}
                onContextMenu={(e) => handleItemContextMenu(e, thread.id)}
                role="button"
                aria-selected={activeItem === thread.id}
                aria-label={`Thread: ${thread.title}`}
                onMouseDown={(e) => {
                  if (e.button === 2) e.preventDefault();
                }}
              >
                {safeRename.isRenaming && safeRename.itemId === thread.id && (safeRename.isThread ?? true) ? (
                  <input
                    ref={renameInputRef}
                    value={safeRename.newName}
                    onChange={onRenameChange}
                    onKeyDown={onRenameKeyDown}
                    onBlur={onRenameBlur}
                    className="bg-zinc-900 text-white px-1 w-full border border-zinc-700 rounded"
                    autoFocus
                    aria-label="Rename thread"
                    onContextMenu={(e) => e.stopPropagation()}
                  />
                ) : (
                  <div className="relative">
                    <span id={`thread-${thread.id}`} className="truncate block max-w-full">
                      {thread.title}
                    </span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
