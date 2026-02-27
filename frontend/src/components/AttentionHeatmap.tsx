"use client";

import { useMemo } from "react";
import { Group } from "@visx/group";
import { HeatmapRect } from "@visx/heatmap";
import { scaleLinear } from "@visx/scale";
import { Text } from "@visx/text";
import type { TokenLayout } from "@/hooks/useVisualizerSocket";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Props {
  attention: Record<string, number[][][]> | null;
  selectedLayer: number;
  tokenLayout: TokenLayout | null;
  numHeads: number;
}

// ---------------------------------------------------------------------------
// Colour scale  (white → indigo → dark)
// ---------------------------------------------------------------------------

function attnColor(t: number): string {
  // t in [0, 1]
  const r = Math.round(13 + t * 79);
  const g = Math.round(13 + t * 52);
  const b = Math.round(32 + t * 160);
  return `rgb(${r},${g},${b})`;
}

// ---------------------------------------------------------------------------
// Single head heatmap
// ---------------------------------------------------------------------------

interface HeadHeatmapProps {
  matrix: number[][];          // [T_q][T_k]
  labels: string[];
  headIdx: number;
  width: number;
  height: number;
}

function HeadHeatmap({ matrix, labels, headIdx, width, height }: HeadHeatmapProps) {
  const T_q = matrix.length;
  const T_k = matrix[0]?.length ?? 0;

  const showLabels = T_k <= 24;
  const labelMargin = showLabels ? 28 : 4;

  const innerW = width - labelMargin - 4;
  const innerH = height - labelMargin - 20; // 20 for title

  const cellW = T_k > 0 ? innerW / T_k : 0;
  const cellH = T_q > 0 ? innerH / T_q : 0;

  return (
    <svg width={width} height={height} style={{ background: "#0d0d20" }}>
      {/* Head title */}
      <Text
        x={labelMargin + innerW / 2}
        y={14}
        textAnchor="middle"
        fill="#9ca3af"
        fontSize={9}
      >
        {`H${headIdx}`}
      </Text>

      <Group top={20} left={labelMargin}>
        {/* Y-axis labels */}
        {showLabels &&
          Array.from({ length: T_q }, (_, r) => (
            <text
              key={`yl-${r}`}
              x={-3}
              y={r * cellH + cellH / 2 + 4}
              textAnchor="end"
              fontSize={5}
              fill="#6b7280"
            >
              {labels[r] ?? r}
            </text>
          ))}

        {/* X-axis labels */}
        {showLabels &&
          Array.from({ length: T_k }, (_, c) => (
            <text
              key={`xl-${c}`}
              x={c * cellW + cellW / 2}
              y={innerH + 9}
              textAnchor="middle"
              fontSize={5}
              fill="#6b7280"
              transform={`rotate(-45, ${c * cellW + cellW / 2}, ${innerH + 9})`}
            >
              {labels[c] ?? c}
            </text>
          ))}

        {/* Cells */}
        {matrix.map((row, r) =>
          row.map((val, c) => (
            <rect
              key={`${r}-${c}`}
              x={c * cellW}
              y={r * cellH}
              width={Math.max(cellW - 0.5, 1)}
              height={Math.max(cellH - 0.5, 1)}
              fill={attnColor(val)}
            />
          ))
        )}
      </Group>
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function AttentionHeatmap({
  attention,
  selectedLayer,
  tokenLayout,
  numHeads,
}: Props) {
  const layerKey = String(selectedLayer);
  const layerData = attention?.[layerKey];      // [nh][T_q][T_k]

  const labels = useMemo(() => {
    if (!tokenLayout) return [];
    return tokenLayout.labels;
  }, [tokenLayout]);

  if (!layerData) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-[#0d0d20]">
        <span className="text-xs text-gray-600">
          {attention === null ? "Waiting for data…" : `No data for layer ${selectedLayer}`}
        </span>
      </div>
    );
  }

  // Lay out heads in a 2-column grid
  const gridCols = Math.min(numHeads, 2);
  const gridRows = Math.ceil(numHeads / gridCols);

  return (
    <div
      className="w-full h-full flex flex-col bg-[#0d0d20] p-1"
      style={{ overflow: "hidden" }}
    >
      {/* Title row */}
      <div className="text-[10px] text-gray-400 px-1 pb-1 flex-shrink-0">
        Attention · Layer {selectedLayer}
      </div>

      {/* Head grid */}
      <div
        className="flex-1 grid gap-0.5"
        style={{
          gridTemplateColumns: `repeat(${gridCols}, 1fr)`,
          gridTemplateRows: `repeat(${gridRows}, 1fr)`,
          overflow: "hidden",
        }}
      >
        {Array.from({ length: numHeads }, (_, h) => {
          const matrix = layerData[h] ?? [];
          return (
            <div key={h} className="relative overflow-hidden">
              <HeadHeatmapFill
                matrix={matrix}
                labels={labels}
                headIdx={h}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Responsive wrapper that fills its container
// ---------------------------------------------------------------------------

function HeadHeatmapFill({
  matrix,
  labels,
  headIdx,
}: {
  matrix: number[][];
  labels: string[];
  headIdx: number;
}) {
  return (
    <div className="w-full h-full">
      <HeadHeatmap
        matrix={matrix}
        labels={labels}
        headIdx={headIdx}
        width={200}
        height={200}
      />
    </div>
  );
}
