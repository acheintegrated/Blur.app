import React, { forwardRef, RefObject } from 'react'
interface SearchOption {
  id: string
  label: string
}
interface SearchOptionsDropdownProps {
  searchType: 'fuzzy' | 'exact'
  searchQuery: string
  inputRef: RefObject<HTMLInputElement>
  onSelectOption: (optionId: string) => void
  onQueryChange: (e: React.ChangeEvent<HTMLInputElement>) => void
  onKeyDown: (e: React.KeyboardEvent) => void
}
const searchOptions: SearchOption[] = [
  {
    id: 'fuzzy',
    label: 'Fuzzy Search',
  },
  {
    id: 'exact',
    label: 'Exact Match',
  },
]
export const SearchOptionsDropdown = forwardRef<
  HTMLDivElement,
  SearchOptionsDropdownProps
>(
  (
    {
      searchType,
      searchQuery,
      inputRef,
      onSelectOption,
      onQueryChange,
      onKeyDown,
    },
    ref,
  ) => {
    return (
      <div
        ref={ref}
        className="absolute left-64 top-[170px] bg-gray-900 border border-zinc-900 rounded shadow-lg z-10 w-64 py-1"
        role="dialog"
        aria-label="Search options"
      >
        <div className="px-4 py-2 text-white border-b border-zinc-900 mb-2">
          Search Options
        </div>
        {searchOptions.map((option) => (
          <div
            key={option.id}
            className={`px-4 py-2 hover:bg-zinc-900 cursor-pointer hover-drip-effect ${searchType === option.id ? 'text-white' : 'text-gray-400'}`}
            onClick={() => onSelectOption(option.id)}
            role="option"
            aria-selected={searchType === option.id}
            tabIndex={0}
          >
            {option.label}
          </div>
        ))}
        {/* Search input */}
        <div className="px-4 py-2">
          <input
            ref={inputRef}
            type="text"
            placeholder={
              searchType === 'fuzzy' ? 'Search threads...' : 'Exact match...'
            }
            value={searchQuery}
            onChange={onQueryChange}
            onKeyDown={onKeyDown}
            className="bg-zinc-900 text-white px-2 py-1 w-full outline-none rounded"
            autoFocus
            aria-label="Search input"
          />
        </div>
      </div>
    )
  },
)
SearchOptionsDropdown.displayName = 'SearchOptionsDropdown'
