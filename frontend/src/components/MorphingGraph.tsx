"use client";

/**
 * MorphingGraph — placeholder scaffold for future D3.js morphing graph view.
 *
 * This component will eventually render a dynamic graph visualisation of the
 * transformer's token-to-token information flow, animated to morph between
 * timesteps as the agent plays.
 *
 * TODO:
 *   - Import d3-force for layout
 *   - Render nodes for each token position (colour-coded by token type)
 *   - Draw edges weighted by attention magnitude from the selected layer/head
 *   - Animate transitions with d3-transition between steps
 */

interface Props {
  attention: Record<string, number[][][]> | null;
  selectedLayer: number;
}

export default function MorphingGraph({ attention: _a, selectedLayer: _l }: Props) {
  return (
    <div className="w-full h-full flex items-center justify-center bg-[#0d0d20] border border-dashed border-gray-700 rounded">
      <div className="text-center">
        <p className="text-xs text-gray-500 font-medium">Morphing Graph</p>
        <p className="text-[10px] text-gray-700 mt-1">D3.js — coming soon</p>
      </div>
    </div>
  );
}
