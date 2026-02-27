"use client";

/**
 * MorphingGraph — D3 force-directed graph of token-to-token attention flow.
 *
 * Nodes  = token positions derived from tokenLayout (never hardcoded).
 * Edges  = attention weights averaged across all heads for the selected layer.
 * Layout = D3 force simulation; re-runs softly on each frame so the graph
 *          morphs smoothly as the agent plays.
 *
 * The action token ("act") is pinned to the right side of the SVG so the
 * graph doesn't rotate arbitrarily between frames.
 */

import { useEffect, useRef } from "react";
import { select } from "d3-selection";
import "d3-transition"; // side-effect: extends Selection prototype with .transition()
import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  type ForceLink,
  type Simulation,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from "d3-force";
import type { TokenLayout } from "@/hooks/useVisualizerSocket";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const OBS_COLOR = "#818cf8";  // indigo-400 — matches heatmap colour scheme
const ACT_COLOR = "#f472b6";  // pink-400 — action token
const EDGE_COLOR = "#6366f1"; // indigo-500
const BG_COLOR = "#0d0d20";   // matches all other panes

const EDGE_THRESHOLD = 0.02;  // edges with avg attention below this are suppressed
const TRANSITION_MS = 350;    // slightly shorter than ~500 ms frame interval @ 2 fps
const NODE_R_MIN = 4;         // px — minimum node radius
const NODE_R_MAX = 14;        // px — maximum (highest incoming attention)

// ---------------------------------------------------------------------------
// D3 data types
// ---------------------------------------------------------------------------

interface NodeDatum extends SimulationNodeDatum {
  id: number;
  label: string;
  isAction: boolean;
  radius: number;
}

interface EdgeDatum extends SimulationLinkDatum<NodeDatum> {
  weight: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Average attention across all heads → [T_q][T_k]. */
function avgAttentionMatrix(layerData: number[][][]): number[][] {
  const numHeads = layerData.length;
  if (numHeads === 0) return [];
  const Tq = layerData[0].length;
  const Tk = layerData[0][0]?.length ?? 0;
  const avg: number[][] = Array.from({ length: Tq }, () =>
    new Array<number>(Tk).fill(0),
  );
  for (const head of layerData) {
    for (let r = 0; r < Tq; r++) {
      for (let c = 0; c < Tk; c++) {
        avg[r][c] += (head[r]?.[c] ?? 0) / numHeads;
      }
    }
  }
  return avg;
}

/**
 * Stable key for D3 edge data join.
 * Handles both numeric and resolved-object source/target:
 * d3-forceLink replaces indices with node objects in-place after .links() is called.
 */
function edgeKey(d: EdgeDatum): string {
  const s =
    typeof d.source === "number" ? d.source : (d.source as NodeDatum).id;
  const t =
    typeof d.target === "number" ? d.target : (d.target as NodeDatum).id;
  return `${s}-${t}`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface Props {
  attention: Record<string, number[][][]> | null;
  selectedLayer: number;
  tokenLayout: TokenLayout | null;
}

export default function MorphingGraph({
  attention,
  selectedLayer,
  tokenLayout,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const simRef = useRef<Simulation<NodeDatum, EdgeDatum> | null>(null);
  // nodesRef keeps mutable node objects alive across both effects so the
  // simulation can update their x/y, and Effect 2 can update their radius.
  const nodesRef = useRef<NodeDatum[]>([]);

  // -------------------------------------------------------------------------
  // Effect 1: rebuild simulation and SVG skeleton when tokenLayout changes.
  //           Runs on initial mount and on every agent switch.
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (!svgRef.current || !tokenLayout) return;

    const el = svgRef.current;
    // getBoundingClientRect() returns 0 in jsdom — fall back to sensible defaults
    const { width: rawW, height: rawH } = el.getBoundingClientRect();
    const W = rawW || 600;
    const H = rawH || 400;
    const cx = W / 2;
    const cy = H / 2;
    const ringR = Math.min(cx, cy) * 0.62;

    // ---- Build nodes --------------------------------------------------------
    const numTokens = tokenLayout.labels.length;
    const nodes: NodeDatum[] = tokenLayout.labels.map((label, i) => {
      const angle = (2 * Math.PI * i) / numTokens - Math.PI / 2;
      const isAction = label === "act";
      return {
        id: i,
        label,
        isAction,
        radius: 6,
        x: cx + ringR * Math.cos(angle),
        y: cy + ringR * Math.sin(angle),
        // Pin the action token to the right — prevents graph from rotating
        fx: isAction ? cx + ringR * 0.85 : undefined,
        fy: isAction ? cy : undefined,
      };
    });
    nodesRef.current = nodes;

    // ---- Clear and rebuild SVG structure ------------------------------------
    const svg = select(el);
    svg.selectAll("*").remove();
    svg.append("g").attr("class", "edges");
    const nodesG = svg.append("g").attr("class", "nodes");

    // ---- Render node groups (circle + label per token) ----------------------
    const nodeEnter = nodesG
      .selectAll<SVGGElement, NodeDatum>("g")
      .data(nodes, (d) => String(d.id))
      .enter()
      .append("g");

    nodeEnter
      .append("circle")
      .attr("r", (d) => d.radius)
      .attr("fill", (d) => (d.isAction ? ACT_COLOR : OBS_COLOR))
      .attr("stroke", "#ffffff18")
      .attr("stroke-width", 1);

    nodeEnter
      .append("text")
      .attr("dy", "0.32em")
      .attr("text-anchor", "middle")
      .attr("font-size", 5)
      .attr("fill", "#ffffffaa")
      .attr("pointer-events", "none")
      .text((d) => d.label);

    // ---- Stop previous simulation before creating a new one -----------------
    simRef.current?.stop();

    // ---- Create force simulation (no edges yet — added in Effect 2) ---------
    const sim = forceSimulation<NodeDatum>(nodes)
      .force("charge", forceManyBody<NodeDatum>().strength(-60))
      .force("center", forceCenter(cx, cy).strength(0.05))
      .force(
        "collide",
        forceCollide<NodeDatum>().radius((d) => d.radius + 5),
      )
      .force(
        "link",
        forceLink<NodeDatum, EdgeDatum>([])
          .id((d) => d.id)
          .strength((d) => (d.weight ?? 0) * 0.2)
          .distance(60),
      )
      .alphaDecay(0.015)
      .on("tick", () => {
        svg
          .select<SVGGElement>(".edges")
          .selectAll<SVGLineElement, EdgeDatum>("line")
          .attr("x1", (d) => (d.source as NodeDatum).x ?? 0)
          .attr("y1", (d) => (d.source as NodeDatum).y ?? 0)
          .attr("x2", (d) => (d.target as NodeDatum).x ?? 0)
          .attr("y2", (d) => (d.target as NodeDatum).y ?? 0);

        svg
          .select<SVGGElement>(".nodes")
          .selectAll<SVGGElement, NodeDatum>("g")
          .attr("transform", (d) => `translate(${d.x ?? 0},${d.y ?? 0})`);
      });

    simRef.current = sim;

    return () => {
      sim.stop();
    };
  }, [tokenLayout]);

  // -------------------------------------------------------------------------
  // Effect 2: update edge weights and node sizes on each new attention frame.
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (!svgRef.current || !attention || !tokenLayout) return;

    const layerData = attention[String(selectedLayer)];
    if (!layerData || nodesRef.current.length === 0) return;

    const avgAttn = avgAttentionMatrix(layerData);
    const numTokens = tokenLayout.labels.length;

    // Guard: attention shape doesn't match current tokenLayout (agent-switch lag).
    // Wait for the next frame when both tokenLayout and attention are consistent.
    if (avgAttn.length !== numTokens) return;

    // ---- Incoming attention per node (column sums) --------------------------
    const incomingAttn = new Array<number>(numTokens).fill(0);
    for (let r = 0; r < numTokens; r++) {
      for (let c = 0; c < numTokens; c++) {
        incomingAttn[c] += avgAttn[r]?.[c] ?? 0;
      }
    }
    const maxIncoming = Math.max(...incomingAttn, 1);

    // Update mutable radius on each node object (simulation uses these in tick)
    nodesRef.current.forEach((node) => {
      node.radius =
        NODE_R_MIN +
        (incomingAttn[node.id] / maxIncoming) * (NODE_R_MAX - NODE_R_MIN);
    });

    // ---- Build edge list above threshold ------------------------------------
    const edges: EdgeDatum[] = [];
    for (let r = 0; r < numTokens; r++) {
      for (let c = 0; c < numTokens; c++) {
        if (r === c) continue;
        const w = avgAttn[r]?.[c] ?? 0;
        if (w > EDGE_THRESHOLD) {
          edges.push({ source: r, target: c, weight: w });
        }
      }
    }

    // ---- Update SVG edges (data join keyed by source-target pair) -----------
    const svg = select(svgRef.current);
    const edgeSel = svg
      .select<SVGGElement>(".edges")
      .selectAll<SVGLineElement, EdgeDatum>("line")
      .data(edges, edgeKey);

    edgeSel.exit().remove();

    edgeSel
      .enter()
      .append("line")
      .attr("stroke", EDGE_COLOR)
      .attr("stroke-width", 0)
      .attr("stroke-opacity", 0)
      .merge(edgeSel)
      .transition()
      .duration(TRANSITION_MS)
      .attr("stroke-width", (d) => Math.max(d.weight * 5, 0.3))
      .attr("stroke-opacity", (d) => Math.min(d.weight * 2.5, 0.9));

    // ---- Transition node circles to updated radii ---------------------------
    svg
      .select<SVGGElement>(".nodes")
      .selectAll<SVGGElement, NodeDatum>("g")
      .data(nodesRef.current, (d) => String(d.id))
      .select("circle")
      .transition()
      .duration(TRANSITION_MS)
      .attr("r", (d) => d.radius);

    // ---- Update force link and soft-restart → morph to new equilibrium ------
    const sim = simRef.current;
    if (sim) {
      (sim.force("link") as ForceLink<NodeDatum, EdgeDatum>).links(edges);
      sim.alpha(0.12).restart();
    }
  }, [attention, selectedLayer, tokenLayout]);

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  if (!tokenLayout) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-[#0d0d20]">
        <span className="text-xs text-gray-600">Waiting for data…</span>
      </div>
    );
  }

  return (
    <div className="w-full h-full bg-[#0d0d20] relative overflow-hidden">
      {/* Legend */}
      <div className="absolute top-1 left-2 flex items-center gap-3 z-10 pointer-events-none">
        <span className="text-[9px] text-gray-600">
          Avg attn · Layer {selectedLayer}
        </span>
        <span className="flex items-center gap-1">
          <span
            className="inline-block w-2 h-2 rounded-full"
            style={{ background: OBS_COLOR }}
          />
          <span className="text-[9px] text-gray-500">obs</span>
        </span>
        <span className="flex items-center gap-1">
          <span
            className="inline-block w-2 h-2 rounded-full"
            style={{ background: ACT_COLOR }}
          />
          <span className="text-[9px] text-gray-500">act (pinned)</span>
        </span>
      </div>
      <svg
        ref={svgRef}
        className="w-full h-full"
        style={{ background: BG_COLOR }}
      />
    </div>
  );
}
