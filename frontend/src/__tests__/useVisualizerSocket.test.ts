/**
 * Tests for useVisualizerSocket:
 *   - Reconnects on disconnect
 *   - Correctly parses incoming messages
 *   - Token layout updates when token_layout changes between agents
 *   - Agent switch UI: state clears immediately, loading indicator appears
 */

import { act, renderHook, waitFor } from "@testing-library/react";
import { useVisualizerSocket, type ControlCommand } from "../hooks/useVisualizerSocket";

// ---------------------------------------------------------------------------
// Mock WebSocket
// ---------------------------------------------------------------------------

type WsHandler = (ev: { data?: string; code?: number }) => void;

interface MockWs {
  onopen: WsHandler | null;
  onclose: WsHandler | null;
  onerror: WsHandler | null;
  onmessage: WsHandler | null;
  close: () => void;
  _trigger: (event: string, payload?: unknown) => void;
}

let _latestWs: MockWs | null = null;

class MockWebSocket {
  onopen: WsHandler | null = null;
  onclose: WsHandler | null = null;
  onerror: WsHandler | null = null;
  onmessage: WsHandler | null = null;

  constructor(_url: string) {
    _latestWs = this as unknown as MockWs;
    (this as unknown as MockWs)._trigger = (event: string, payload?: unknown) => {
      const handler = (this as Record<string, unknown>)[`on${event}`] as WsHandler | null;
      handler?.(payload as { data?: string; code?: number });
    };
  }

  close() {
    (this as unknown as MockWs)._trigger("close", { code: 1000 });
  }
}

(MockWebSocket as unknown as { CONNECTING: number; OPEN: number; CLOSING: number; CLOSED: number }).CONNECTING = 0;
(MockWebSocket as unknown as { OPEN: number }).OPEN = 1;
(MockWebSocket as unknown as { CLOSING: number }).CLOSING = 2;
(MockWebSocket as unknown as { CLOSED: number }).CLOSED = 3;

const originalWebSocket = global.WebSocket;

beforeAll(() => {
  global.WebSocket = MockWebSocket as unknown as typeof WebSocket;
  global.fetch = jest.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({}) });
});

afterAll(() => {
  global.WebSocket = originalWebSocket;
});

beforeEach(() => {
  _latestWs = null;
  jest.useFakeTimers();
});

afterEach(() => {
  jest.useRealTimers();
  jest.clearAllMocks();
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function sendToHook(msg: object) {
  _latestWs?._trigger("message", { data: JSON.stringify(msg) });
}

function connectHook() {
  _latestWs?._trigger("open", {});
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("useVisualizerSocket", () => {
  it("starts disconnected", () => {
    const { result } = renderHook(() => useVisualizerSocket());
    expect(result.current.state.connected).toBe(false);
  });

  it("becomes connected after WebSocket open", async () => {
    const { result } = renderHook(() => useVisualizerSocket());
    act(() => connectHook());
    await waitFor(() => expect(result.current.state.connected).toBe(true));
  });

  it("reconnects after disconnect", async () => {
    const { result } = renderHook(() => useVisualizerSocket());
    act(() => connectHook());
    await waitFor(() => expect(result.current.state.connected).toBe(true));

    const firstWs = _latestWs;
    act(() => firstWs?._trigger("close", { code: 1006 }));
    await waitFor(() => expect(result.current.state.connected).toBe(false));

    // Advance past reconnect delay
    act(() => jest.advanceTimersByTime(3000));
    act(() => _latestWs?._trigger("open", {}));
    await waitFor(() => expect(result.current.state.connected).toBe(true));

    expect(_latestWs).not.toBe(firstWs); // new socket created
  });

  it("parses frame messages and updates frame/attention/norms/metrics", async () => {
    const { result } = renderHook(() => useVisualizerSocket());
    act(() => connectHook());

    const frameMsg = {
      type: "frame",
      frame: "abc123",
      attention: { "0": [[[0.1, 0.9], [0.5, 0.5]]] },
      norms: [1.0, 2.0],
      metrics: {
        infer_fps: 15,
        step: 42,
        episode: 1,
        queue_depth: 0,
        hook_latency_ms: 2.1,
        drop_rate: 0,
        return: 5.0,
      },
      token_layout: {
        tokens_per_block: 5,
        obs_per_block: 4,
        labels: ["o0", "o1", "o2", "o3", "act"],
      },
    };

    act(() => sendToHook(frameMsg));

    await waitFor(() => {
      expect(result.current.state.frame).toBe("abc123");
      expect(result.current.state.norms).toEqual([1.0, 2.0]);
      expect(result.current.state.metrics?.step).toBe(42);
      expect(result.current.state.token_layout?.tokens_per_block).toBe(5);
    });
  });

  it("parses config messages and stores config + agents list", async () => {
    const { result } = renderHook(() => useVisualizerSocket());
    act(() => connectHook());

    const configMsg = {
      type: "config",
      num_layers: 10,
      num_heads: 4,
      embed_dim: 256,
      tokens_per_block: 17,
      max_blocks: 20,
      agents: [
        { id: "Breakout", name: "Breakout", path: "/ckpts/Breakout.pt", env_id: "BreakoutNoFrameskip-v4" },
      ],
    };

    act(() => sendToHook(configMsg));

    await waitFor(() => {
      expect(result.current.state.config?.num_layers).toBe(10);
      expect(result.current.state.config?.agents).toHaveLength(1);
      expect(result.current.state.config?.agents[0].id).toBe("Breakout");
    });
  });

  it("token_layout updates when a new agent sends different token_layout", async () => {
    const { result } = renderHook(() => useVisualizerSocket());
    act(() => connectHook());

    // First agent: tpb=17
    act(() =>
      sendToHook({
        type: "frame",
        frame: "f1",
        attention: {},
        norms: [],
        metrics: { infer_fps: 10, step: 1, episode: 1, queue_depth: 0, hook_latency_ms: 1, drop_rate: 0, return: 0 },
        token_layout: { tokens_per_block: 17, obs_per_block: 16, labels: ["o0"] },
      })
    );
    await waitFor(() => expect(result.current.state.token_layout?.tokens_per_block).toBe(17));

    // Switch agent: tpb=5
    act(() =>
      sendToHook({
        type: "frame",
        frame: "f2",
        attention: {},
        norms: [],
        metrics: { infer_fps: 10, step: 2, episode: 1, queue_depth: 0, hook_latency_ms: 1, drop_rate: 0, return: 0 },
        token_layout: { tokens_per_block: 5, obs_per_block: 4, labels: ["o0", "o1", "o2", "o3", "act"] },
      })
    );
    await waitFor(() => expect(result.current.state.token_layout?.tokens_per_block).toBe(5));
  });

  it("clears frame/attention/norms immediately on switch_agent control command", async () => {
    const { result } = renderHook(() => useVisualizerSocket());
    act(() => connectHook());

    // Populate state
    act(() =>
      sendToHook({
        type: "frame",
        frame: "initial",
        attention: { "0": [[[1]]] },
        norms: [1.5],
        metrics: { infer_fps: 15, step: 1, episode: 1, queue_depth: 0, hook_latency_ms: 1, drop_rate: 0, return: 0 },
        token_layout: { tokens_per_block: 17, obs_per_block: 16, labels: [] },
      })
    );
    await waitFor(() => expect(result.current.state.frame).toBe("initial"));

    // Send switch_agent
    await act(async () => {
      await result.current.sendControl({
        command: "switch_agent",
        payload: { checkpoint_path: "/ckpts/Alien.pt", env_id: "AlienNoFrameskip-v4" },
      });
    });

    // Visualisation state should be cleared and loading should be true
    expect(result.current.state.frame).toBeNull();
    expect(result.current.state.attention).toBeNull();
    expect(result.current.state.norms).toBeNull();
    expect(result.current.state.loading).toBe(true);
  });

  it("shows loading indicator on agent_loaded event", async () => {
    const { result } = renderHook(() => useVisualizerSocket());
    act(() => connectHook());

    // Populate with initial frame
    act(() =>
      sendToHook({
        type: "frame",
        frame: "old_frame",
        attention: {},
        norms: [],
        metrics: { infer_fps: 10, step: 1, episode: 1, queue_depth: 0, hook_latency_ms: 1, drop_rate: 0, return: 0 },
        token_layout: { tokens_per_block: 17, obs_per_block: 16, labels: [] },
      })
    );
    await waitFor(() => expect(result.current.state.frame).toBe("old_frame"));

    // Backend signals agent loaded
    act(() => sendToHook({ type: "event", event: "agent_loaded", data: { agent: "Alien" } }));

    await waitFor(() => {
      expect(result.current.state.loading).toBe(true);
      expect(result.current.state.frame).toBeNull();
    });
  });

  it("appends events to event log", async () => {
    const { result } = renderHook(() => useVisualizerSocket());
    act(() => connectHook());

    act(() =>
      sendToHook({ type: "event", event: "episode_start", data: { episode: 1 } })
    );
    act(() =>
      sendToHook({ type: "event", event: "episode_end", data: { episode: 1, return: 42 } })
    );

    await waitFor(() => {
      expect(result.current.state.events.length).toBeGreaterThanOrEqual(2);
    });
  });

  it("does not schedule duplicate reconnect when agentName changes (intentional close)", async () => {
    // Render with one agent name; connect and confirm connected.
    const { result, rerender } = renderHook(
      ({ agent }: { agent?: string }) => useVisualizerSocket(agent),
      { initialProps: { agent: "Breakout" } }
    );
    act(() => connectHook());
    await waitFor(() => expect(result.current.state.connected).toBe(true));

    const firstWs = _latestWs;
    const eventCountBefore = result.current.state.events.length;

    // Switch to a different agent — triggers cleanup of old effect and new connect()
    rerender({ agent: "Alien" });

    // The old socket's onclose fires (intentional) — no extra "disconnected" event logged
    // The new effect body calls connect() immediately, creating a fresh socket.
    act(() => _latestWs?._trigger("open", {}));
    await waitFor(() => expect(result.current.state.connected).toBe(true));

    // A new socket must have been created
    expect(_latestWs).not.toBe(firstWs);

    // No spurious "disconnected" event should have been added for the intentional close
    const disconnectedEvents = result.current.state.events.filter(
      (e) => e.event === "disconnected"
    );
    expect(disconnectedEvents).toHaveLength(0);
  });
});
