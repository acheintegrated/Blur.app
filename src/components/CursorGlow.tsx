// src/components/CursorGlow.tsx
import React, { useEffect, useRef } from 'react';

interface CursorGlowProps {
  isEmphatic?: boolean;   // stronger glow + more dust while streaming
  scopeSelector?: string; // where to hide the OS cursor. default: 'body'
  cursorSize?: number;    // px diameter of the star core (default 3)
  pool?: number;          // max confetti particles (default 160)
}

type P = {
  alive: boolean;
  x: number; y: number;
  vx: number; vy: number;
  life: number; maxLife: number;
  size: number;
  hue: number;
  el: HTMLDivElement | null;
};

export const CursorGlow: React.FC<CursorGlowProps> = ({
  isEmphatic = false,
  scopeSelector = 'body',
  cursorSize = 3,
  pool = 160,
}) => {
  const rootRef = useRef<HTMLDivElement>(null);
  const cursorRef = useRef<HTMLDivElement>(null);

  const particles = useRef<P[]>([]);
  const els = useRef<Array<HTMLDivElement | null>>([]);
  const raf = useRef<number | null>(null);

  const target = useRef({ x: 0, y: 0 });
  const pos = useRef({ x: 0, y: 0 });
  const last = useRef({ x: 0, y: 0 });

  // init pool
  useEffect(() => {
    particles.current = Array.from({ length: pool }, () => ({
      alive: false, x: 0, y: 0, vx: 0, vy: 0, life: 0, maxLife: 0, size: 0, hue: 0, el: null,
    }));
  }, [pool]);

  // create DOM for particles
  useEffect(() => {
    const root = rootRef.current!;
    els.current = particles.current.map(() => {
      const d = document.createElement('div');
      d.className = 'cg2-dust';
      d.style.opacity = '0';
      d.style.transform = 'translate(-9999px,-9999px)';
      root.appendChild(d);
      return d;
    });
    // bind elements to particles
    particles.current.forEach((p, i) => (p.el = els.current[i]));

    return () => {
      els.current.forEach((d) => d && d.remove());
      els.current = [];
    };
  }, []);

  // input + loop
  useEffect(() => {
    const onFirst = (e: MouseEvent) => {
      target.current.x = e.clientX; target.current.y = e.clientY;
      pos.current.x = e.clientX; pos.current.y = e.clientY;
      last.current.x = e.clientX; last.current.y = e.clientY;
      window.removeEventListener('mousemove', onFirst);
    };
    window.addEventListener('mousemove', onFirst, { passive: true });

    const onMove = (e: MouseEvent) => {
      target.current.x = e.clientX;
      target.current.y = e.clientY;
    };
    window.addEventListener('mousemove', onMove, { passive: true });

    let tPrev = performance.now();
    const tick = (t: number) => {
      const dt = Math.min(0.05, (t - tPrev) / 1000); // clamp
      tPrev = t;

      const ease = 0.22;
      pos.current.x += (target.current.x - pos.current.x) * ease;
      pos.current.y += (target.current.y - pos.current.y) * ease;

      const speed = Math.hypot(target.current.x - last.current.x, target.current.y - last.current.y);
      last.current.x = target.current.x;
      last.current.y = target.current.y;

      const baseHue = (t * 0.06 + speed * 2) % 360;

      if (cursorRef.current) {
        cursorRef.current.style.transform =
          `translate(${pos.current.x}px, ${pos.current.y}px) translate(-50%, -50%)`;
        cursorRef.current.style.setProperty('--h', String(baseHue));
        cursorRef.current.style.setProperty('--emph', isEmphatic ? '1' : '0');
      }

      const spawnBase = isEmphatic ? 8 : 8;
      let toSpawn = Math.min(24, Math.floor((spawnBase + speed * 0.25)));
      while (toSpawn-- > 0) {
        const p = particles.current.find((q) => !q.alive);
        if (!p) break;
        p.alive = true;
        p.x = pos.current.x;
        p.y = pos.current.y;
        const ang = Math.random() * Math.PI * 2;
        const mag = (Math.random() * 60 + 30) * (0.4 + Math.min(1, speed / 24)); // px/s
        p.vx = Math.cos(ang) * mag;
        p.vy = Math.sin(ang) * mag;
        p.size = Math.random() * 2.5 + 1; // smaller dust
        p.maxLife = (isEmphatic ? 1.0 : 0.85) + Math.random() * 0.4;
        p.life = p.maxLife;
        p.hue = (baseHue + Math.random() * 80 - 40 + 360) % 360;

        const el = p.el!;
        el.style.width = `${p.size}px`;
        el.style.height = `${p.size}px`;
        // DIMMER: Significantly reduced lightness
        const c = `hsl(${p.hue} 90% 35%)`;
        // UNCHANGED: Kept the white background as requested
        el.style.background = 'white';
        el.style.boxShadow = `0 0 4px ${c}, 0 0 8px ${c}`;
      }

      const drag = 0.9;
      particles.current.forEach((p) => {
        if (!p.alive) return;
        p.life -= dt;
        if (p.life <= 0) {
          p.alive = false;
          if (p.el) p.el.style.opacity = '0';
          return;
        }
        p.vx *= drag; p.vy *= drag;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        const k = p.life / p.maxLife;
        const el = p.el!;
        // TRANSLUCENT: Fade out is slightly faster
        el.style.opacity = String(Math.max(0, k * 1.25));
        el.style.transform = `translate(${p.x}px, ${p.y}px) translate(-50%, -50%) rotate(${(1 - k) * 180}deg)`;
      });

      raf.current = requestAnimationFrame(tick);
    };

    raf.current = requestAnimationFrame(tick);
    return () => {
      window.removeEventListener('mousemove', onMove);
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [isEmphatic, cursorSize]);

  return (
    <>
      <style>{`${scopeSelector}, ${scopeSelector} * { cursor: none !important; }`}</style>
      <style>{`
        @keyframes cg2-star-pulse {
          0%, 100% {
            transform: translate(-50%, -50%) scale(1);
            filter: brightness(1);
          }
          50% {
            transform: translate(-50%, -50%) scale(1.15);
            filter: brightness(1.25);
          }
        }

        .cg2-root { position: fixed; inset: 0; z-index: 2147483647; pointer-events: none; }
        .cg2-cursor, .cg2-dust {
          position: absolute; left: 0; top: 0;
          will-change: transform, filter, opacity, box-shadow;
          pointer-events: none;
        }

        .cg2-cursor {
          transform: translate(-50%,-50%);
          width: ${cursorSize}px; height: ${cursorSize}px; border-radius: 9999px;
          /* UNCHANGED: Kept the white background as requested */
          background: white;
          /* DIMMER/TRANSLUCENT: Greatly reduced lightness and opacity */
          box-shadow:
            0 0 3px hsl(var(--h, 280) 95% 65% / 0.7),
            0 0 7px hsl(var(--h, 280) 90% 55% / 0.4),
            0 0 12px hsl(var(--h, 280) 85% 45% / 0.15);
        }

        .cg2-dust {
          transform: translate(-50%,-50%);
          border-radius: 2px;
          mix-blend-mode: screen;
        }
      `}</style>

      <div ref={rootRef} className="cg2-root" aria-hidden>
        <div ref={cursorRef} className="cg2-cursor" />
      </div>
    </>
  );
};