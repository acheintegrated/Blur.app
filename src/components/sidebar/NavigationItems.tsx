import React, { RefObject } from 'react';

// REFINEMENT: The component's props are updated to receive state and a ref directly from its parent, SideBar.tsx.
// This ensures the parent is in full control of the search box's visibility.
interface NavigationItemsProps {
  onNewConversation: () => void;
  onSearch: () => void;
  searchContainerRef: RefObject<HTMLDivElement>; // This ref is crucial for detecting outside clicks.
  searchState: { showOptions: boolean; query: string }; // This state tells the component whether to show the input.
}

export const NavigationItems: React.FC<NavigationItemsProps> = ({
  onNewConversation,
  onSearch,
  searchContainerRef,
  searchState,
}) => {
  // REFINEMENT: All local state (`searchActive`, `searchQuery`) has been removed.
  // The component now relies entirely on props from its parent, making it a "controlled component".
  // This is the key to fixing the bug.

  const handleSearchClick = (e: React.MouseEvent) => {
    e.stopPropagation(); // Prevents the click from bubbling up and immediately closing the input.
    onSearch();
  };

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // The actual state update for the query will be handled by the parent component (SideBar.tsx).
    // This function can be expanded later to pass the new value up.
  };

  return (
    <div className="border-b border-zinc-900 p-4 space-y-1">
      <div
        className="flex items-center space-x-2 cursor-pointer hover:text-white px-2 py-1.5 rounded"
        onClick={onNewConversation}
        role="button"
        aria-label="Start new chat"
      >
        <span className="text-lg icon-font">∞</span>
        <span>new</span>
      </div>

      {/* REFINEMENT: The `ref` from the parent is now attached to this container.
          The parent's click-away logic is watching this specific element. */}
      <div
        ref={searchContainerRef}
        id="search-threads-button"
        className="flex items-center space-x-2 cursor-pointer hover:text-white px-2 py-1.5 rounded"
        onClick={handleSearchClick}
        role="button"
        aria-label="Search threads"
      >
        {/* REFINEMENT: The decision to show the input is now driven by the parent's state (`searchState.showOptions`). */}
        {searchState.showOptions ? (
          <input
            type="text"
            placeholder="Search..."
            value={searchState.query} // The input's value is also controlled by the parent.
            onChange={handleSearchChange}
            className="bg-zinc-900 text-white px-2 py-1 w-full outline-none rounded"
            autoFocus
            aria-label="Search input"
            onClick={(e) => e.stopPropagation()} // Prevents clicking inside the input from closing it.
          />
        ) : (
          <>
            <span className="text-lg icon-font">∃</span>
            <span>search</span>
          </>
        )}
      </div>
    </div>
  );
};
