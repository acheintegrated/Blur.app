import React from 'react'
interface TooltipDisplayProps {
  text: string
  x: number
  y: number
}
export const TooltipDisplay: React.FC<TooltipDisplayProps> = ({
  text,
  x,
  y,
}) => {
  return (
    <div
      className="fixed bg-gray-900 px-2 py-1 rounded text-xs text-white z-50 pointer-events-none"
      style={{
        left: `${x}px`,
        top: `${y}px`,
        transition: 'opacity 0.2s',
      }}
      role="tooltip"
    >
      {text}
    </div>
  )
}
