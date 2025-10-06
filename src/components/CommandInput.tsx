// CommandInput.tsx
import React, { useEffect, useState, useRef } from 'react';

interface CommandInputProps {
  onSendMessage: (message: string, threadId?: string) => void;
  connectionStatus: 'initializing' | 'connecting' | 'loading_model' | 'ready' | 'error';
  isLoading?: boolean;
  threadId: string;
}

export const CommandInput: React.FC<CommandInputProps> = ({
  onSendMessage,
  connectionStatus,
  isLoading = false,
  threadId,
}) => {
  const [command, setCommand] = useState('');
  const [isActive, setIsActive] = useState(false);
  const [loadingBlinkColor, setLoadingBlinkColor] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const enterPressedRef = useRef(false);

  // Debounced auto-resize for smooth typing performance
  useEffect(() => {
    const resizeTimeout = setTimeout(() => {
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
        const minHeight = window.innerHeight * 0.2;
        const maxHeight = window.innerHeight * 0.27;
        const newHeight = Math.min(
          Math.max(textareaRef.current.scrollHeight, minHeight),
          maxHeight,
        );
        textareaRef.current.style.height = `${newHeight}px`;
      }
    }, 100);

    return () => clearTimeout(resizeTimeout);
  }, [command]);

  // Auto-focus on mount and when connection is ready
  useEffect(() => { textareaRef.current?.focus(); }, []);
  useEffect(() => {
    if (!isLoading && connectionStatus === 'ready') {
      requestAnimationFrame(() => textareaRef.current?.focus());
    }
  }, [isLoading, connectionStatus]);

  // Random color for loading placeholder blink
  useEffect(() => {
    if (connectionStatus === 'ready') { setLoadingBlinkColor(''); return; }
    const interval = setInterval(() => {
      const random = `#${Math.floor(Math.random() * 0xffffff).toString(16).padStart(6, '0')}`;
      setLoadingBlinkColor(random);
    }, 800);
    return () => clearInterval(interval);
  }, [connectionStatus]);

  const handleSend = () => {
    if (!command.trim() || isLoading || connectionStatus !== 'ready') return;
    enterPressedRef.current = true;
    onSendMessage(command, threadId);
    setCommand('');
    enterPressedRef.current = false;
    requestAnimationFrame(() => textareaRef.current?.focus());
  };

  const handleFocus = () => setIsActive(true);
  const handleBlur = () => { if (!command) setIsActive(false); };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !enterPressedRef.current) {
      if (e.shiftKey) return;
      e.preventDefault();
      handleSend();
    }
  };
  const handleKeyUp = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter') enterPressedRef.current = false;
  };

  const handleContainerClick = () => textareaRef.current?.focus();

  const isDisabled = connectionStatus !== 'ready';
  const isSendDisabled = !command.trim() || isLoading || connectionStatus !== 'ready';

  return (
    <div className="border-t border-zinc-900 flex flex-col">
      <style>{`
        @keyframes fade { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }
        .blink { animation: fade 0.8s ease-in-out infinite; }
      `}</style>

      <div className="flex items-start p-2 pt-0 cursor-text" onClick={handleContainerClick}>
        {!isActive && !command && (
          <div className="mr-2 mt-1">
            <span
              className={`text-white font-normal ${connectionStatus !== 'ready' ? 'random-glow blink' : 'blurline-glow'}`}
              style={connectionStatus !== 'ready'
                ? { textShadow: `0 0 8px ${loadingBlinkColor}, 0 0 14px ${loadingBlinkColor}, 0 0 22px ${loadingBlinkColor}` }
                : {}}
            >
              {connectionStatus !== 'ready'
                ? '༄⎈ᛝ𓆩⫷...loading🜃blurline...⫸𓆪ᛝ⎈༄'
                : '༄⎈ᛝ𓆩⫷touch🜃blurline⫸𓆪ᛝ⎈༄'}
            </span>
          </div>
        )}

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
          spellCheck="false"
          rows={1}
          disabled={isDisabled}
          aria-busy={isLoading}
          style={{
            minHeight: `${window.innerHeight * 0.2}px`,
            maxHeight: `${window.innerHeight * 0.27}px`,
            paddingTop: '8px',
            marginTop: '0',
            overflowY: 'auto', // ensures scrollbars appear when needed
          }}
        />

      </div>

      <div className="flex justify-end px-3 pb-3">
        {/* The mic button has been removed, leaving only the send button in a simple flex-end container */}
        <button
          className="w-8 h-8 flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed"
          onMouseDown={(e) => e.preventDefault()}
          onClick={handleSend}
          disabled={isSendDisabled}
          aria-label="Send"
        >
          <span className="cursor-pointer text-gray-400 hover:text-white text-2xl transform scale-[1.2] inline-block icon-font">
            ❍
          </span>
        </button>
      </div>
    </div>
  );
};