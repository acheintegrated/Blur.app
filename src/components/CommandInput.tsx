import React, { useEffect, useState, useRef, useCallback } from 'react';

interface CommandInputProps {
  onSendMessage: (message: string, threadId?: string) => void;
  onStop?: () => void;
  connectionStatus: 'initializing' | 'connecting' | 'loading_model' | 'ready' | 'error';
  isLoading?: boolean;
  threadId: string;
}

export const CommandInput: React.FC<CommandInputProps> = ({
  onSendMessage,
  onStop,
  connectionStatus,
  isLoading = false,
  threadId,
}) => {
  const [command, setCommand] = useState('');
  const [isActive, setIsActive] = useState(false);
  const [loadingBlinkColor, setLoadingBlinkColor] = useState('');
  
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const enterPressedRef = useRef(false);

  // =================== STYLES ===================
  const LocalStyles = () => (
    <style>{`
      @keyframes neon-dynamic-glow {
        0%, 100% { text-shadow: 0 0 5px rgba(242, 0, 242, .9), 0 0 10px rgba(242, 0, 242, .8), 0 0 20px rgba(242, 0, 242, .7); opacity: .95; }
        20%      { text-shadow: 0 0 5px rgba(255, 111, 0, .9), 0 0 10px rgba(255, 119, 0, .8), 0 0 20px rgba(255, 165, 0, .7); }
        40%      { text-shadow: 0 0 5px rgba(125, 220, 37, .9), 0 0 10px rgba(164, 220, 32, .8), 0 0 20px rgba(34, 218, 55, .7); }
        60%      { text-shadow: 0 0 5px rgba(0, 221, 255, .9), 0 0 10px rgba(0, 195, 255, .8), 0 0 20px rgba(0, 136, 255, .7); }
        80%      { text-shadow: 0 0 5px rgba(96, 18, 252, .9), 0 0 10px rgba(132, 34, 252, .8), 0 0 20px rgba(123, 0, 255, .7); }
      }
      @keyframes slow-pulse-scale {
        0%, 100% { transform: scale(1); }
        50%      { transform: scale(1.14); }
      }
      .anim-stop-glow-pulse {
        animation: neon-dynamic-glow 8s linear infinite, slow-pulse-scale 1.8s ease-in-out infinite;
        display: inline-block;
        will-change: transform, text-shadow, filter;
      }
      @keyframes fade {
        0%, 100% { opacity: 1; }
        50%      { opacity: 0.2; }
      }
      .blink {
        animation: fade 0.8s ease-in-out infinite;
      }
    `}</style>
  );

  // =================== TEXTAREA AUTO-RESIZE ===================
  useEffect(() => {
    const timer = setTimeout(() => {
      const textarea = textareaRef.current;
      if (!textarea) return;

      textarea.style.height = 'auto';
      const minHeight = window.innerHeight * 0.2;
      const maxHeight = window.innerHeight * 0.27;
      const newHeight = Math.min(Math.max(textarea.scrollHeight, minHeight), maxHeight);
      textarea.style.height = `${newHeight}px`;
    }, 0);

    return () => clearTimeout(timer);
  }, [command]);

  // =================== AUTO-FOCUS ===================
  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  useEffect(() => {
    if (!isLoading && connectionStatus === 'ready') {
      requestAnimationFrame(() => textareaRef.current?.focus());
    }
  }, [isLoading, connectionStatus]);

  // =================== LOADING PLACEHOLDER GLOW ===================
  useEffect(() => {
    if (connectionStatus === 'ready') {
      setLoadingBlinkColor('');
      return;
    }

    const interval = setInterval(() => {
      const randomColor = `#${Math.floor(Math.random() * 0xffffff)
        .toString(16)
        .padStart(6, '0')}`;
      setLoadingBlinkColor(randomColor);
    }, 800);

    return () => clearInterval(interval);
  }, [connectionStatus]);

  // =================== HANDLERS ===================
  const handleSend = useCallback(() => {
    if (!command.trim() || isLoading || connectionStatus !== 'ready') return;

    enterPressedRef.current = true;
    onSendMessage(command, threadId);
    setCommand('');
    enterPressedRef.current = false;

    requestAnimationFrame(() => textareaRef.current?.focus());
  }, [command, isLoading, connectionStatus, onSendMessage, threadId]);

  const handleStopClick = useCallback((e: React.MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    e.stopPropagation();
    onStop?.();
  }, [onStop]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Escape key stops streaming
    if (e.key === 'Escape' && isLoading && onStop) {
      e.preventDefault();
      e.stopPropagation();
      onStop();
      return;
    }

    // Enter sends (unless Shift is held)
    if (e.key === 'Enter' && !enterPressedRef.current) {
      if (e.shiftKey) return;
      e.preventDefault();
      handleSend();
    }
  }, [isLoading, onStop, handleSend]);

  const handleKeyUp = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter') {
      enterPressedRef.current = false;
    }
  }, []);

  const handleFocus = useCallback(() => setIsActive(true), []);
  
  const handleBlur = useCallback(() => {
    if (!command) setIsActive(false);
  }, [command]);

  const handleContainerClick = useCallback((e: React.MouseEvent) => {
    const target = e.target as HTMLElement;
    // Don't steal focus from buttons
    if (target.closest('[data-role="stop-btn"]') || target.closest('[data-role="send-btn"]')) {
      return;
    }
    textareaRef.current?.focus();
  }, []);

  // =================== COMPUTED STATE ===================
  const isTextDisabled = connectionStatus !== 'ready' && !isLoading;
  const isSendDisabled = !command.trim() || isLoading || connectionStatus !== 'ready';

  const placeholderText = connectionStatus !== 'ready'
    ? '༄⎈ᛝ𓆩⫷...loading🜃blurline...⫸𓆪ᛝ⎈༄'
    : '༄⎈ᛝ𓆩⫷touch🜃blurline⫸𓆪ᛝ⎈༄';

  const placeholderStyle =
    connectionStatus !== 'ready'
      ? {
          textShadow: `0 0 8px ${loadingBlinkColor}, 0 0 14px ${loadingBlinkColor}, 0 0 22px ${loadingBlinkColor}`,
        }
      : {};

  // =================== RENDER ===================
  return (
    <div className="border-t border-zinc-900 flex flex-col min-h-[120px]">
      <LocalStyles />

      {/* Input Area */}
      <div 
        className="flex items-start p-2 pt-2 cursor-text flex-1" 
        onClick={handleContainerClick}
      >
        {/* Floating Placeholder */}
        {!isActive && !command && (
          <div className="mr-2 mt-1 flex-shrink-0">
            <span
              className={`text-white font-normal ${
                connectionStatus !== 'ready' ? 'random-glow blink' : 'blurline-glow'
              }`}
              style={placeholderStyle}
            >
              {placeholderText}
            </span>
          </div>
        )}

        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={command}
          onChange={(e) => setCommand(e.target.value)}
          onFocus={handleFocus}
          onBlur={handleBlur}
          onKeyDown={handleKeyDown}
          onKeyUp={handleKeyUp}
          className="bg-transparent flex-1 outline-none text-white resize-none overflow-y-auto px-4"
          placeholder={isActive ? 'start typing' : ''}
          spellCheck={false}
          rows={1}
          disabled={isTextDisabled}
          aria-busy={isLoading}
          style={{
            minHeight: `${window.innerHeight * 0.2}px`,
            maxHeight: `${window.innerHeight * 0.27}px`,
            paddingTop: '8px',
            marginTop: '0',
            overflowY: 'auto',
          }}
        />
      </div>

      {/* Action Button Area */}
      <div className="flex justify-end px-3 pb-3 flex-shrink-0">
        {isLoading ? (
          /* STOP BUTTON */
          <button
            type="button"
            data-role="stop-btn"
            className="relative z-50 w-8 h-8 flex items-center justify-center pointer-events-auto text-white"
            onClick={handleStopClick}
            onMouseDown={(e) => {
              e.preventDefault();
              e.stopPropagation();
            }}
            aria-label="Stop generating"
            title="Stop (Esc)"
          >
            <span className="cursor-pointer text-2xl transform inline-block icon-font anim-stop-glow-pulse select-none">
              ⊙
            </span>
          </button>
        ) : (
          /* SEND BUTTON */
          <button
            type="button"
            data-role="send-btn"
            className="relative z-50 w-8 h-8 flex items-center justify-center pointer-events-auto disabled:opacity-50 disabled:cursor-not-allowed"
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              handleSend();
            }}
            onMouseDown={(e) => {
              e.preventDefault();
              e.stopPropagation();
            }}
            disabled={isSendDisabled}
            aria-label="Send"
            title="Send (Enter)"
          >
            <span className="cursor-pointer text-gray-400 hover:text-white text-2xl transform scale-[1.2] inline-block icon-font select-none">
              ❍
            </span>
          </button>
        )}
      </div>
    </div>
  );
};