import React, { memo } from 'react'
import { XIcon } from 'lucide-react'
import { RainbowGlow } from '../RainbowGlow'
import { useNavigate } from 'react-router-dom'
interface ArchiveSelectionModalProps {
  onClose: () => void
}
export const ArchiveSelectionModal: React.FC<ArchiveSelectionModalProps> = ({
  onClose,
}) => {
  const navigate = useNavigate()
  const handleSelection = () => {
    navigate('/archive')
    onClose()
  }
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop with blur effect */}
      <div className="absolute inset-0 backdrop-blur-md bg-black bg-opacity-30"></div>
      <div className="relative bg-opacity-90 backdrop-blur-md border border-gray-700 rounded-lg w-full max-w-md shadow-2xl overflow-hidden glass-panel">
        {/* Header */}
        <div className="border-b border-gray-800 p-4 flex justify-between items-center">
          <RainbowGlow className="text-white text-xl font-bold" dynamic={true}>
            ⊖ memory nest
          </RainbowGlow>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors duration-150"
          >
            <XIcon size={20} />
          </button>
        </div>
        {/* Content */}
        <div className="p-8 space-y-6">
          <h2 className="text-white text-lg mb-6 text-center">
            View Archived Conversations
          </h2>
          <div className="flex justify-center">
            <button
              className="p-6 bg-gray-800 hover:bg-gray-700 rounded-lg flex flex-col items-center justify-center space-y-4 transition-colors duration-150 border border-gray-700 hover:border-pink-300 w-full max-w-xs"
              onClick={handleSelection}
            >
              <span className="text-3xl text-pink-300 icon-font magenta-purple-glow-hover">
                ∞
              </span>
              <span className="text-white">Archived Conversations</span>
              <span className="text-gray-400 text-sm text-center">
                View your archived blur conversations
              </span>
            </button>
          </div>
        </div>
        {/* Footer */}
        <div className="border-t border-gray-800 p-4 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded bg-gray-800 text-gray-300 hover:bg-gray-700 transition-colors duration-150"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  )
}
