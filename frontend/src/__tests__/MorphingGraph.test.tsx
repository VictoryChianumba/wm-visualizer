/**
 * Tests for MorphingGraph — D3 force-directed attention graph.
 *
 * Coverage:
 *   - Placeholder when tokenLayout is null
 *   - SVG rendered when tokenLayout is provided
 *   - Node (circle) count matches tokenLayout.labels.length
 *   - Node count updates when tokenLayout changes (agent switch)
 *   - Edges appear when attention data is above the threshold
 *   - No crash when attention token count mismatches tokenLayout (agent-switch lag)
 *   - No crash when attention is null
 *   - Selected layer number appears in the legend
 */

import React from "react";
import { render, screen, act } from "@testing-library/react";
import MorphingGraph from "@/components/MorphingGraph";
import type { TokenLayout } from "@/hooks/useVisualizerSocket";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Build a TokenLayout whose labels follow the IRIS token pattern:
 *   [ o0, o1, …, o(tpb-2), act ] repeating.
 */
function makeTokenLayout(
  numTokens: number,
  tokensPerBlock = 5,
): TokenLayout {
  const labels: string[] = [];
  for (let i = 0; i < numTokens; i++) {
    labels.push(
      i % tokensPerBlock === tokensPerBlock - 1 ? "act" : `o${i % tokensPerBlock}`,
    );
  }
  return { labels };
}

/**
 * Build a uniform attention dict for a single layer.
 * Each head has a (numTokens × numTokens) matrix of value 1/numTokens,
 * which is 0.2 for numTokens=5 — well above EDGE_THRESHOLD (0.02).
 */
function makeUniformAttention(
  numTokens: number,
  numHeads = 4,
  layerKey = "5",
): Record<string, number[][][]> {
  const val = 1 / numTokens;
  const layer: number[][][] = Array.from({ length: numHeads }, () =>
    Array.from({ length: numTokens }, () =>
      new Array<number>(numTokens).fill(val),
    ),
  );
  return { [layerKey]: layer };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("MorphingGraph", () => {
  // ── Placeholder ──────────────────────────────────────────────────────────

  it("renders a waiting placeholder when tokenLayout is null", () => {
    render(
      <MorphingGraph attention={null} selectedLayer={5} tokenLayout={null} />,
    );
    expect(screen.getByText(/waiting for data/i)).toBeInTheDocument();
  });

  it("does not render the SVG when tokenLayout is null", () => {
    const { container } = render(
      <MorphingGraph attention={null} selectedLayer={5} tokenLayout={null} />,
    );
    expect(container.querySelector("svg")).toBeNull();
  });

  // ── Node count ───────────────────────────────────────────────────────────

  it("renders an SVG element when tokenLayout is provided", () => {
    const { container } = render(
      <MorphingGraph
        attention={null}
        selectedLayer={5}
        tokenLayout={makeTokenLayout(5)}
      />,
    );
    expect(container.querySelector("svg")).toBeInTheDocument();
  });

  it("creates exactly one circle per token label", async () => {
    const numTokens = 5;
    const { container } = render(
      <MorphingGraph
        attention={null}
        selectedLayer={5}
        tokenLayout={makeTokenLayout(numTokens)}
      />,
    );
    await act(async () => {});
    expect(container.querySelectorAll("svg circle")).toHaveLength(numTokens);
  });

  it("creates one circle per token for the IRIS default 17-token layout", async () => {
    const numTokens = 17;
    const { container } = render(
      <MorphingGraph
        attention={null}
        selectedLayer={5}
        tokenLayout={makeTokenLayout(numTokens, 17)}
      />,
    );
    await act(async () => {});
    expect(container.querySelectorAll("svg circle")).toHaveLength(numTokens);
  });

  it("updates node count when tokenLayout changes (agent switch)", async () => {
    const { container, rerender } = render(
      <MorphingGraph
        attention={null}
        selectedLayer={5}
        tokenLayout={makeTokenLayout(5)}
      />,
    );
    await act(async () => {});
    expect(container.querySelectorAll("svg circle")).toHaveLength(5);

    rerender(
      <MorphingGraph
        attention={null}
        selectedLayer={5}
        tokenLayout={makeTokenLayout(17, 17)}
      />,
    );
    await act(async () => {});
    expect(container.querySelectorAll("svg circle")).toHaveLength(17);
  });

  // ── Edges ─────────────────────────────────────────────────────────────────

  it("adds edge elements when attention weights exceed the threshold", async () => {
    const numTokens = 5;
    const tokenLayout = makeTokenLayout(numTokens);
    const attention = makeUniformAttention(numTokens);

    const { container } = render(
      <MorphingGraph
        attention={attention}
        selectedLayer={5}
        tokenLayout={tokenLayout}
      />,
    );
    await act(async () => {});

    // Uniform attention (0.2 per cell) > EDGE_THRESHOLD (0.02), so edges should appear.
    // Off-diagonal entries: numTokens*(numTokens-1) = 20 edges for 5 tokens.
    const lines = container.querySelectorAll("svg line");
    expect(lines.length).toBeGreaterThan(0);
  });

  it("renders zero edges when no attention data is provided", async () => {
    const { container } = render(
      <MorphingGraph
        attention={null}
        selectedLayer={5}
        tokenLayout={makeTokenLayout(5)}
      />,
    );
    await act(async () => {});
    expect(container.querySelectorAll("svg line")).toHaveLength(0);
  });

  // ── Robustness ────────────────────────────────────────────────────────────

  it("does not crash when attention token count mismatches tokenLayout (agent-switch lag)", async () => {
    // During an agent switch, tokenLayout may update before attention does
    // (or vice-versa). Effect 2 guards against this mismatch.
    const tokenLayout = makeTokenLayout(5);
    const attention = makeUniformAttention(17); // wrong size

    expect(() =>
      render(
        <MorphingGraph
          attention={attention}
          selectedLayer={5}
          tokenLayout={tokenLayout}
        />,
      ),
    ).not.toThrow();
    await act(async () => {});
  });

  it("does not crash when attention is null after tokenLayout loads", async () => {
    const tokenLayout = makeTokenLayout(5);
    expect(() =>
      render(
        <MorphingGraph
          attention={null}
          selectedLayer={5}
          tokenLayout={tokenLayout}
        />,
      ),
    ).not.toThrow();
    await act(async () => {});
  });

  // ── Layer display ─────────────────────────────────────────────────────────

  it("shows the selected layer number in the legend", () => {
    render(
      <MorphingGraph
        attention={null}
        selectedLayer={7}
        tokenLayout={makeTokenLayout(5)}
      />,
    );
    // Legend reads "Avg attn · Layer 7"
    expect(screen.getByText(/layer 7/i)).toBeInTheDocument();
  });

  it("updates the legend when selectedLayer prop changes", () => {
    const { rerender } = render(
      <MorphingGraph
        attention={null}
        selectedLayer={3}
        tokenLayout={makeTokenLayout(5)}
      />,
    );
    expect(screen.getByText(/layer 3/i)).toBeInTheDocument();

    rerender(
      <MorphingGraph
        attention={null}
        selectedLayer={9}
        tokenLayout={makeTokenLayout(5)}
      />,
    );
    expect(screen.getByText(/layer 9/i)).toBeInTheDocument();
  });
});
