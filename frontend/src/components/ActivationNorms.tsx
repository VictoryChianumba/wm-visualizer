"use client";

import { useMemo } from "react";
import { Group } from "@visx/group";
import { Bar } from "@visx/shape";
import { scaleBand, scaleLinear } from "@visx/scale";
import { AxisBottom, AxisLeft } from "@visx/axis";

interface Props {
  norms: number[] | null;
  numLayers: number;
}

const MARGIN = { top: 10, right: 8, bottom: 28, left: 36 };

// Plasma-ish colour from normalised value [0, 1]
function plasmaColor(t: number): string {
  const r = Math.round(13 + t * 234);
  const g = Math.round(8 + t * 93);
  const b = Math.round(135 - t * 27);
  return `rgb(${r},${g},${b})`;
}

export default function ActivationNorms({ norms, numLayers }: Props) {
  const width = 280;
  const height = 160;
  const innerW = width - MARGIN.left - MARGIN.right;
  const innerH = height - MARGIN.top - MARGIN.bottom;

  const layers = useMemo(
    () => Array.from({ length: numLayers }, (_, i) => i),
    [numLayers]
  );

  const displayNorms = useMemo(
    () => norms ?? new Array(numLayers).fill(0),
    [norms, numLayers]
  );

  const maxNorm = useMemo(
    () => Math.max(...displayNorms, 0.001),
    [displayNorms]
  );

  const xScale = useMemo(
    () =>
      scaleBand<number>({
        domain: layers,
        range: [0, innerW],
        padding: 0.15,
      }),
    [layers, innerW]
  );

  const yScale = useMemo(
    () =>
      scaleLinear<number>({
        domain: [0, maxNorm * 1.1],
        range: [innerH, 0],
        clamp: true,
      }),
    [maxNorm, innerH]
  );

  if (norms === null) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-[#0d0d20]">
        <span className="text-xs text-gray-600">Waiting for data…</span>
      </div>
    );
  }

  return (
    <div className="w-full h-full flex flex-col bg-[#0d0d20] p-1 overflow-hidden">
      <div className="text-[10px] text-gray-400 px-1 pb-0.5 flex-shrink-0">
        Residual Norms
      </div>
      <div className="flex-1 flex items-center justify-center overflow-hidden">
        <svg width={width} height={height} style={{ overflow: "visible" }}>
          <Group top={MARGIN.top} left={MARGIN.left}>
            {/* Bars */}
            {layers.map((i) => {
              const x = xScale(i) ?? 0;
              const bw = xScale.bandwidth();
              const norm = displayNorms[i] ?? 0;
              const yTop = yScale(norm);
              const barH = innerH - yTop;
              const color = plasmaColor(norm / maxNorm);
              return (
                <Bar
                  key={i}
                  x={x}
                  y={yTop}
                  width={bw}
                  height={Math.max(barH, 0)}
                  fill={color}
                />
              );
            })}

            {/* Y axis */}
            <AxisLeft
              scale={yScale}
              numTicks={4}
              stroke="#374151"
              tickStroke="#374151"
              tickLabelProps={() => ({
                fill: "#6b7280",
                fontSize: 7,
                textAnchor: "end",
                dy: "0.3em",
              })}
            />

            {/* X axis */}
            <AxisBottom
              top={innerH}
              scale={xScale}
              tickFormat={(v) => String(v)}
              stroke="#374151"
              tickStroke="#374151"
              tickValues={
                numLayers <= 10
                  ? layers
                  : layers.filter((i) => i % Math.ceil(numLayers / 5) === 0)
              }
              tickLabelProps={() => ({
                fill: "#6b7280",
                fontSize: 7,
                textAnchor: "middle",
              })}
            />
          </Group>
        </svg>
      </div>
    </div>
  );
}
