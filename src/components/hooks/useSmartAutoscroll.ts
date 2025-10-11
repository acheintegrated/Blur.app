// file: /opt/blurface/src/components/hooks/useSmartAutoscroll.ts

import { useRef, useState, useCallback } from 'react';

interface SmartAutoscrollReturn<T extends HTMLElement> {
  ref: React.RefObject<T>;
  isPinned: boolean;
  onScroll: () => void;
  onContentAppended: () => void;
  scrollToBottom: (smooth?: boolean) => void;
}

export const useSmartAutoscroll = <T extends HTMLElement>(): SmartAutoscrollReturn<T> => {
  const ref = useRef<T>(null);
  const [isPinned, setIsPinned] = useState(true);
  const isPinnedRef = useRef(true);
  const lastScrollHeightRef = useRef(0);

  const onScroll = useCallback(() => {
    if (!ref.current) return;
    
    const { scrollTop, scrollHeight, clientHeight } = ref.current;
    const atBottom = scrollHeight - scrollTop - clientHeight < 10; // Increased threshold
    
    setIsPinned(atBottom);
    isPinnedRef.current = atBottom;
  }, []);

  const scrollToBottom = useCallback((smooth = false) => {
    if (!ref.current) return;
    
    const { scrollHeight } = ref.current;
    
    ref.current.scrollTo({
      top: scrollHeight,
      behavior: smooth ? 'smooth' : 'auto',
    });
    
    setIsPinned(true);
    isPinnedRef.current = true;
    lastScrollHeightRef.current = scrollHeight;
  }, []);

  const onContentAppended = useCallback(() => {
    if (!ref.current) return;
    
    const currentScrollHeight = ref.current.scrollHeight;
    
    // Only scroll if pinned AND content actually changed
    if (isPinnedRef.current && currentScrollHeight !== lastScrollHeightRef.current) {
      requestAnimationFrame(() => {
        scrollToBottom(true);
      });
    }
  }, [scrollToBottom]);

  return {
    ref,
    isPinned,
    onScroll,
    onContentAppended,
    scrollToBottom,
  };
};