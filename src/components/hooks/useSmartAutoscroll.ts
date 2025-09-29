// file: /opt/blurface/src/components/hooks/useSmartAutoscroll.ts
// FINAL VERSION — Clean, simple, and robust logic for scroll-locking.

import { useRef, useState, useEffect, useCallback } from 'react';

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

  const onScroll = useCallback(() => {
    if (ref.current) {
      const { scrollTop, scrollHeight, clientHeight } = ref.current;
      const atBottom = scrollHeight - scrollTop - clientHeight < 5;
      setIsPinned(atBottom);
    }
  }, []);

  const scrollToBottom = useCallback((smooth = false) => {
    if (ref.current) {
      ref.current.scrollTo({
        top: ref.current.scrollHeight,
        behavior: smooth ? 'smooth' : 'auto',
      });
      setIsPinned(true);
    }
  }, []);

  const onContentAppended = useCallback(() => {
    if (isPinned) {
      scrollToBottom(true);
    }
  }, [isPinned, scrollToBottom]);

  useEffect(() => {
    scrollToBottom();
  }, [scrollToBottom]);

  return {
    ref,
    isPinned,
    onScroll,
    onContentAppended,
    scrollToBottom,
  };
};