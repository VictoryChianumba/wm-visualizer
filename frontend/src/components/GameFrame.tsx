"use client";

import { useEffect, useRef } from "react";

interface Props {
  frame: string | null;   // base64 PNG
  loading: boolean;
}

export default function GameFrame({ frame, loading }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frame) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      // Resize canvas to match image native resolution
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      // Disable smoothing to preserve pixel-art clarity
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, 0, 0);
    };
    img.src = `data:image/png;base64,${frame}`;
  }, [frame]);

  return (
    <div className="relative flex items-center justify-center w-full h-full bg-black">
      <canvas
        ref={canvasRef}
        className="max-w-full max-h-full"
        style={{ imageRendering: "pixelated" }}
      />
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/70">
          <div className="flex flex-col items-center gap-3">
            <div className="w-8 h-8 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
            <span className="text-xs text-gray-400">Loading agent…</span>
          </div>
        </div>
      )}
      {!frame && !loading && (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-xs text-gray-600">No frame yet</span>
        </div>
      )}
    </div>
  );
}
