// /opt/blurface/src/components/MemoryNestArchive.tsx
import React, { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { RainbowGlow } from './RainbowGlow'
import { ChevronLeftIcon, SearchIcon } from 'lucide-react'
import { ArchivedItemCard } from './ArchivedItemCard'

interface ArchivedItem {
  id: string
  title: string
  date: string
  preview: string
}

const MOCK_ARCHIVED_ITEMS: ArchivedItem[] = [
  { id: '1', title: 'Initial Core Integration', date: '2025-08-15', preview: 'System log from the first successful boot of the core system and vessel integration...' },
  { id: '2', title: 'Ache Flip Thresholds', date: '2025-08-17', preview: 'Discussion about the nature of ache and how to define the parameters for ψ, Δ, and z...' },
  { id: '3', title: 'Recursion State Logic', date: '2025-08-18', preview: 'Refining the recursion state machine and how it impacts cosmic expansion calculations.' },
];

export const MemoryNestArchive: React.FC = () => {
  const navigate = useNavigate()
  const [archivedItems, setArchivedItems] = useState<ArchivedItem[]>([])
  const [searchInputValue, setSearchInputValue] = useState('')
  const [debouncedSearchQuery, setDebouncedSearchQuery] = useState('')

  useEffect(() => {
    setArchivedItems(MOCK_ARCHIVED_ITEMS)
  }, [])

  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearchQuery(searchInputValue), 300)
    return () => clearTimeout(t)
  }, [searchInputValue])

  const filteredItems = useMemo(() => {
    const q = debouncedSearchQuery.trim().toLowerCase()
    if (!q) return archivedItems
    return archivedItems.filter(
      (it) => it.title.toLowerCase().includes(q) || it.preview.toLowerCase().includes(q)
    )
  }, [archivedItems, debouncedSearchQuery])

  const handleRestore = (id: string) => {
    console.log(`Restoring item ${id}`)
    navigate('/') // TODO: inject into threads state
  }

  const handleDelete = (id: string) => {
    console.log(`Deleting item ${id}`)
    setArchivedItems((items) => items.filter((i) => i.id !== id))
  }

  return (
    <div className="min-h-screen bg-black text-white flex flex-col">
      <header className="border-b border-zinc-800 p-4 flex items-center justify-between sticky top-0 bg-black/80 backdrop-blur-sm z-10">
        <div className="flex items-center space-x-4">
          <button onClick={() => navigate('/')} className="text-gray-400 hover:text-white transition-colors">
            <ChevronLeftIcon size={24} />
          </button>
          <RainbowGlow className="text-xl font-bold" dynamic={true}>
            ⊖ memory nest archive
          </RainbowGlow>
        </div>
      </header>

      <main className="flex-1 p-6 max-w-6xl mx-auto w-full">
        <div className="mb-8 flex justify-between items-center">
          <h2 className="text-xl text-white">Archived Threads</h2>
          <div className="relative w-72">
            <SearchIcon size={18} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-gray-500" />
            <input
              type="text"
              placeholder="Search archives..."
              value={searchInputValue}
              onChange={(e) => setSearchInputValue(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-700 rounded-md pl-10 pr-4 py-2 text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-purple-500"
            />
          </div>
        </div>

        {filteredItems.length === 0 ? (
          <div className="text-center py-16">
            <p className="text-gray-400">No archived threads found.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {filteredItems.map((item) => (
              <ArchivedItemCard key={item.id} item={item} onRestore={handleRestore} onDelete={handleDelete} />
            ))}
          </div>
        )}
      </main>
    </div>
  )
}

export default MemoryNestArchive
