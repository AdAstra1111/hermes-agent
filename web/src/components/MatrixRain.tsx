import { useEffect, useRef } from "react";

/**
 * Digital-rain canvas backdrop for the Oracle page.
 *
 * Renders the classic falling-glyph effect (half-width katakana + digits)
 * behind its siblings. Honors `prefers-reduced-motion` by drawing a single
 * static frame instead of animating, and pauses while the tab is hidden.
 */
const GLYPHS =
  "ﾊﾐﾋｰｳｼﾅﾓﾆｻﾜﾂｵﾘｱﾎﾃﾏｹﾒｴｶｷﾑﾕﾗｾﾈｽﾀﾇﾍ0123456789Z:・.\"=*+-<>¦|";
const FONT_SIZE = 14;
const FRAME_MS = 66; // ~15fps — plenty for rain, easy on the battery

export function MatrixRain({ opacity = 0.18 }: { opacity?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    let drops: number[] = [];
    let raf = 0;
    let last = 0;

    const resize = () => {
      const parent = canvas.parentElement;
      if (!parent) return;
      canvas.width = parent.clientWidth;
      canvas.height = parent.clientHeight;
      const columns = Math.ceil(canvas.width / FONT_SIZE);
      drops = Array.from({ length: columns }, () =>
        Math.floor((Math.random() * canvas.height) / FONT_SIZE),
      );
      ctx.fillStyle = "rgba(1, 4, 1, 1)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    };

    const drawFrame = () => {
      // Fade the previous frame instead of clearing — this produces the
      // glyph trails.
      ctx.fillStyle = "rgba(1, 4, 1, 0.08)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.font = `${FONT_SIZE}px monospace`;
      for (let i = 0; i < drops.length; i++) {
        const glyph = GLYPHS[Math.floor(Math.random() * GLYPHS.length)];
        const x = i * FONT_SIZE;
        const y = drops[i] * FONT_SIZE;
        // Head glyph is brighter than the trail.
        ctx.fillStyle = Math.random() < 0.1 ? "#d8ffe5" : "#00ff66";
        ctx.fillText(glyph, x, y);
        if (y > canvas.height && Math.random() > 0.975) {
          drops[i] = 0;
        }
        drops[i]++;
      }
    };

    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      if (document.hidden) return;
      if (now - last < FRAME_MS) return;
      last = now;
      drawFrame();
    };

    resize();
    if (reducedMotion) {
      // A single sparse frame so the page still reads "Matrix" without motion.
      for (let f = 0; f < 30; f++) drawFrame();
    } else {
      raf = requestAnimationFrame(tick);
    }

    const observer = new ResizeObserver(resize);
    if (canvas.parentElement) observer.observe(canvas.parentElement);

    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="pointer-events-none absolute inset-0"
      style={{ opacity }}
    />
  );
}
