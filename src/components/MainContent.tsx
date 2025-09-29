
// file: /opt/blurface/src/components/MainContent.tsx

import React, { useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import type { Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useSmartAutoscroll } from './hooks/useSmartAutoscroll';

interface Message {
  sender: 'Blur' | 'You' | 'System';
  text: string;
  systemType?: 'announcement' | 'normal';
}
interface MainContentProps {
  messages: Message[];
  streamingToken?: number;
}

const markdownComponents: Components = {
  // Headings
  h1: ({ children }) => (
    <h1 className="mt-3 mb-2 text-2xl md:text-3xl font-semibold leading-tight">
      {children}
    </h1>
  ),
  h2: ({ children }) => (
    <h2 className="mt-3 mb-2 text-xl md:text-2xl font-semibold leading-tight">
      {children}
    </h2>
  ),
  h3: ({ children }) => (
    <h3 className="mt-3 mb-2 text-lg md:text-xl font-semibold">
      {children}
    </h3>
  ),
  h4: ({ children }) => (
    <h4 className="mt-2 mb-1 text-base md:text-lg font-semibold">
      {children}
    </h4>
  ),
  h5: ({ children }) => (
    <h5 className="mt-2 mb-1 text-base font-semibold">
      {children}
    </h5>
  ),
  h6: ({ children }) => (
    <h6 className="mt-2 mb-1 text-sm font-semibold uppercase tracking-wide text-zinc-400">
      {children}
    </h6>
  ),

  // Paragraphs
  p: ({ children }) => (
    <p className="my-2 whitespace-pre-wrap break-words leading-relaxed">
      {children}
    </p>
  ),

  // Lists
  ul: ({ children }) => <ul className="my-2 ml-5 list-disc space-y-1">{children}</ul>,
  ol: ({ children }) => <ol className="my-2 ml-5 list-decimal space-y-1">{children}</ol>,
  li: ({ children }) => <li className="whitespace-pre-wrap break-words">{children}</li>,

  // Links
  a: ({ href, children }) => (
    <a
      href={href || '#'}
      target="_blank"
      rel="noreferrer"
      className="underline text-purple-300 hover:text-purple-200"
    >
      {children}
    </a>
  ),

  // Blockquote
  blockquote: ({ children }) => (
    <blockquote className="my-2 border-l-2 border-zinc-700 pl-3 italic text-zinc-300">
      {children}
    </blockquote>
  ),

  // Code (inline & blocks)
  code: ({ inline, children }) =>
    inline ? (
      <code className="px-1 py-0.5 rounded bg-zinc-800 text-[0.95em]">{children}</code>
    ) : (
      <pre className="my-2 p-3 rounded bg-black/60 border border-zinc-800 overflow-x-auto">
        <code>{children}</code>
      </pre>
    ),

  // Horizontal rule
  hr: () => <div className="my-4 border-t border-zinc-800" />,
};

export const MainContent: React.FC<MainContentProps> = ({ messages, streamingToken = 0 }) => {
  const { ref, isPinned, onScroll, onContentAppended, scrollToBottom } =
    useSmartAutoscroll<HTMLDivElement>();

  useEffect(() => {
    onContentAppended();
  }, [messages, streamingToken, onContentAppended]);

  return (
    <div className="relative h-full">
      {!isPinned && (
        <button
          className="absolute bottom-3 right-3 rounded-2xl px-3 py-1 shadow-lg bg-black/70 text-white text-sm z-10 hover:bg-black transition-colors duration-200"
          onClick={() => scrollToBottom(true)}
        >
          jump ↓
        </button>
      )}

      <div
        ref={ref}
        onScroll={onScroll}
        className="h-full overflow-y-auto p-4 flex flex-col space-y-4 scroll-smooth body-font"
        style={{ WebkitOverflowScrolling: 'touch', overscrollBehaviorY: 'contain', contain: 'content' }}
      >
        {messages.map((msg, idx) => {
          if (msg.sender === 'Blur' && !msg.text.trim()) return null;

          if (msg.sender === 'System') {
            return (
              <div key={`msg-${idx}`} className="w-full text-center text-xs text-zinc-500 my-2 italic">
                {msg.text}
              </div>
            );
          }

          const isYou = msg.sender === 'You';
          const wrapperClasses = isYou ? 'flex justify-end' : 'flex justify-start';
          const bubbleClasses = isYou ? 'bg-purple-950 text-white rounded-lg' : 'bg-zinc-900 text-white rounded-lg';

          return (
            <div key={`msg-${idx}`} className={`w-full ${wrapperClasses}`}>
              <div
                className={`max-w-[80%] px-4 py-2 whitespace-pre-wrap break-words overflow-hidden ${bubbleClasses}`}
                style={{ overflowWrap: 'anywhere', wordBreak: 'break-word' }}
              >
                <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                  {msg.text}
                </ReactMarkdown>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
