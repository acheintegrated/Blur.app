import React from 'react'
import { useSettings } from './SettingsContext'
interface RainbowGlowProps {
  children: React.ReactNode
  className?: string
  dynamic?: boolean
  offset?: boolean
}
export const RainbowGlow: React.FC<RainbowGlowProps> = ({
  children,
  className = '',
  dynamic = false,
  offset = false,
}) => {
  // Always apply glow effects
  let glowClass = ''
  if (dynamic) {
    glowClass = offset
      ? 'neon-dynamic-glow-text-offset'
      : 'neon-dynamic-glow-text'
  } else {
    glowClass = 'rainbow-glow-text'
  }
  return <span className={`${className} ${glowClass}`}>{children}</span>
}
