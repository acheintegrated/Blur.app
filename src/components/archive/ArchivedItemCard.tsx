// /opt/blurface/src/components/ArchivedItemCard.tsx

import React, { memo } from 'react'
import { RotateCcwIcon, TrashIcon } from 'lucide-react'

// Corrected Interface: Removed the 'type' property
interface ArchivedItem {
  id: string
  title: string
  date: string
  preview: string
}

interface ArchivedItemCardProps {
  item: ArchivedItem
  onRestore: (id: string) => void
  onDelete: (id: string) => void
}

export const ArchivedItemCard: React.FC<ArchivedItemCardProps> = memo(({
  item,
  onRestore,
  onDelete,
}) => {
  const handleRestoreClick = () => onRestore(item.id)
  const handleDeleteClick = () => onDelete(item.id)

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 hover:border-zinc-700 transition-colors duration-200">
      <div className="flex justify-between items-start">
        <div className="flex-1 min-w-0">
          <div className="flex items-center space-x-3 mb-2">
            {/* Corrected Display: Always shows the 'blur' thread icon */}
            <span
              className={`text-sm font-mono px-2 py-0.5 rounded bg-purple-900/40 text-purple-300`}
            >
              ∞
            </span>
            <h3 className="text-white font-medium truncate">{item.title}</h3>
          </div>
          <p className="text-gray-400 text-sm mb-3 line-clamp-2">
            {item.preview}
          </p>
          <p className="text-gray-500 text-xs font-mono">{item.date}</p>
        </div>
        <div className="flex items-center space-x-1 ml-4">
          <button
            onClick={handleRestoreClick}
            className="p-2 text-gray-400 hover:text-green-400 hover:bg-zinc-800 rounded-full transition-colors duration-150"
            title="Restore"
          >
            <RotateCcwIcon size={16} />
          </button>
          <button
            onClick={handleDeleteClick}
            className="p-2 text-gray-400 hover:text-red-400 hover:bg-zinc-800 rounded-full transition-colors duration-150"
            title="Delete permanently"
          >
            <TrashIcon size={16} />
          </button>
        </div>
      </div>
    </div>
  )
})