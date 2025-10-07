// src/components/CursorGlow.tsx
import React, { useEffect, useRef } from 'react';

interface CursorGlowProps {
  isEmphatic?: boolean;   // stronger glow + more dust while streaming
  scopeSelector?: string; // where to hide the OS cursor. default: 'body'
  cursorSize?: number;    // px edge of the square (default 6)
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
  cursorSize = 6,
  pool = 160,
}) => {
  const rootRef = useRef<HTMLDivElement>(null);
  const cursorRef = useRef<HTMLDivElement>(null);
  const haloRef = useRef<HTMLDivElement>(null);

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
    // seed position at first move
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

      // ease cursor toward target
      const ease = 0.22;
      pos.current.x += (target.current.x - pos.current.x) * ease;
      pos.current.y += (target.current.y - pos.current.y) * ease;

      const vx = target.current.x - last.current.x;
      const vy = target.current.y - last.current.y;
      const speed = Math.hypot(vx, vy);
      last.current.x = target.current.x;
      last.current.y = target.current.y;

      // hue over time + speed
      const baseHue = (t * 0.06 + speed * 2) % 360;

      // place tiny square cursor
      if (cursorRef.current) {
        cursorRef.current.style.transform =
          `translate(${pos.current.x}px, ${pos.current.y}px) translate(-50%, -50%)`;
        cursorRef.current.style.setProperty('--h', String(baseHue));
      }
      // halo glow lags slightly (soft)
      if (haloRef.current) {
        const lag = 0.12;
        const hx = pos.current.x + (target.current.x - pos.current.x) * lag;
        const hy = pos.current.y + (target.current.y - pos.current.y) * lag;
        haloRef.current.style.transform =
          `translate(${hx}px, ${hy}px) translate(-50%, -50%)`;
        haloRef.current.style.setProperty('--h', String((baseHue + 20) % 360));
        haloRef.current.style.opacity = String(Math.min(1, 0.35 + speed / 40) * (isEmphatic ? 1.2 : 1));
        haloRef.current.style.filter = `blur(${10 + Math.min(30, speed)}px)`;
      }

      // spawn “digital dust” squares based on speed
      const spawnBase = isEmphatic ? 12 : 8;
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
        p.size = Math.random() * 3 + 1.5; // tiny squares
        p.maxLife = (isEmphatic ? 0.9 : 0.75) + Math.random() * 0.4;
        p.life = p.maxLife;
        p.hue = (baseHue + Math.random() * 80 - 40 + 360) % 360;

        // style the square once
        const el = p.el!;
        el.style.width = `${p.size}px`;
        el.style.height = `${p.size}px`;
        const c = `hsl(${p.hue} 100% 60%)`;
        el.style.background = 'white';
        el.style.boxShadow = `0 0 6px ${c}, 0 0 12px ${c}, 0 0 20px ${c}`;
      }

      // update particles
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
        el.style.opacity = String(Math.max(0, k));
        el.style.transform = `translate(${p.x}px, ${p.y}px) translate(-50%, -50%) rotate(${(1 - k) * 180}deg)`;
      });

      raf.current = requestAnimationFrame(tick);
    };

    raf.current = requestAnimationFrame(tick);
    return () => {
      window.removeEventListener('mousemove', onMove);
      if (raf.current) cancelAnimationFrame(raf.current);
      raf.current = null;
    };
  }, [isEmphatic, cursorSize]);

  return (
    <>
      {/* hide OS cursor inside scope only */}
      <style>{`${scopeSelector}, ${scopeSelector} * { cursor: none !important; }`}</style>
      {/* component styles */}
      <style>{`
        .cg2-root { position: fixed; inset: 0; z-index: 2147483647; pointer-events: none; }
        .cg2-cursor, .cg2-halo, .cg2-dust {
          position: absolute; left: 0; top: 0; transform: translate(-50%,-50%);
          will-change: transform, filter, opacity, box-shadow;
          pointer-events: none;
        }
        /* tiny square cursor */
        .cg2-cursor {
          width: ${cursorSize}px; height: ${cursorSize}px; border-radius: 1px;
          background: white;
          /* rainbow neon: hue via --h */
          box-shadow:
            0 0 6px hsl(var(--h, 280) 100% 60% / .95),
            0 0 14px hsl(var(--h, 280) 100% 60% / .75),
            0 0 26px hsl(calc((var(--h, 280) + 40)) 100% 60% / .55),
            0 0 40px hsl(calc((var(--h, 280) + 80)) 100% 60% / .35);
        }
        /* soft rainbow halo that lags */
        .cg2-halo {
          width: ${cursorSize * 12}px; height: ${cursorSize * 12}px; border-radius: 9999px;
          background: radial-gradient(closest-side,
            hsl(calc((var(--h, 280) + 0)) 100% 60% / .22),
            hsl(calc((var(--h, 280) + 40)) 100% 60% / .18),
            hsl(calc((var(--h, 280) + 80)) 100% 60% / .08),
            transparent 70%
          );
          filter: blur(18px);
          opacity: .4;
          mix-blend-mode: screen;
        }
        .cg2-dust {
          border-radius: 2px;
          mix-blend-mode: screen;
        }
      `}</style>

      <div ref={rootRef} className="cg2-root" aria-hidden>
        <div ref={haloRef} className="cg2-halo" />
        <div ref={cursorRef} className="cg2-cursor" />
        {/* dust squares are appended dynamically */}
      </div>
    </>
  );
};
