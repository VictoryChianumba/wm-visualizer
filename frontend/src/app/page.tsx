"use client";

import { useEffect, useState } from "react";
import ControlBar from "@/components/ControlBar";
import GameFrame from "@/components/GameFrame";
import AttentionHeatmap from "@/components/AttentionHeatmap";
import ActivationNorms from "@/components/ActivationNorms";
import LogPane from "@/components/LogPane";
import type { AgentInfo } from "@/hooks/useVisualizerSocket";
import { useVisualizerSocket } from "@/hooks/useVisualizerSocket";

const API_BASE =
  typeof window !== "undefined"
    ? `http://${window.location.hostname}:8000`
    : "http://localhost:8000";

export default function Home() {
  const [selectedAgent, setSelectedAgent] = useState<string>("");
  const [selectedLayer, setSelectedLayer] = useState<number>(5);
  const [selectedDevice, setSelectedDevice] = useState<string>("cpu");
  const [availableDevices, setAvailableDevices] = useState<string[]>(["cpu"]);

  // Fetch available devices from the backend once on mount
  useEffect(() => {
    fetch(`${API_BASE}/devices`)
      .then((r) => r.json())
      .then((data: { available: string[]; default: string }) => {
        setAvailableDevices(data.available);
        setSelectedDevice(data.default);
      })
      .catch(() => {
        // Backend not ready yet — keep cpu default, will retry on reconnect
      });
  }, []);

  const { state, sendControl } = useVisualizerSocket(
    selectedAgent || undefined,
    undefined,
    selectedDevice,
  );

  // Once we receive the agent list from the backend, default-select the first
  useEffect(() => {
    if (state.config?.agents && state.config.agents.length > 0 && !selectedAgent) {
      setSelectedAgent(state.config.agents[0].id);
    }
  }, [state.config, selectedAgent]);

  // Clamp layer to valid range when config updates
  useEffect(() => {
    const numLayers = state.config?.num_layers;
    if (numLayers != null && numLayers > 0) {
      setSelectedLayer((l) => Math.max(0, Math.min(isNaN(l) ? 5 : l, numLayers - 1)));
    }
  }, [state.config]);

  const agents: AgentInfo[] = state.config?.agents ?? [];
  const numLayers = state.config?.num_layers ?? 10;
  const numHeads = state.config?.num_heads ?? 4;

  function handleAgentChange(agent: AgentInfo) {
    setSelectedAgent(agent.id);
    // sendControl is called by ControlBar
  }

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      {/* ── Top bar ─────────────────────────────────────────────────── */}
      <ControlBar
        agents={agents}
        selectedAgent={selectedAgent}
        onAgentChange={handleAgentChange}
        availableDevices={availableDevices}
        selectedDevice={selectedDevice}
        onDeviceChange={setSelectedDevice}
        selectedLayer={selectedLayer}
        maxLayer={numLayers - 1}
        onLayerChange={setSelectedLayer}
        connected={state.connected}
        loading={state.loading}
        sendControl={sendControl}
      />

      {/* ── Three-pane layout ────────────────────────────────────────── */}
      <main className="flex flex-1 overflow-hidden divide-x divide-gray-800">
        {/* Left: game frame */}
        <div className="flex-1 flex flex-col min-w-0">
          <PaneTitle>Game</PaneTitle>
          <div className="flex-1 overflow-hidden">
            <GameFrame frame={state.frame} loading={state.loading} />
          </div>
        </div>

        {/* Middle: attention heatmap */}
        <div className="flex-1 flex flex-col min-w-0">
          <PaneTitle>Attention · Layer {selectedLayer}</PaneTitle>
          <div className="flex-1 overflow-hidden">
            <AttentionHeatmap
              attention={state.attention}
              selectedLayer={selectedLayer}
              tokenLayout={state.token_layout}
              numHeads={numHeads}
            />
          </div>
        </div>

        {/* Right: norms (top) + log (bottom) */}
        <div className="flex-1 flex flex-col min-w-0 divide-y divide-gray-800">
          <div className="flex-1 flex flex-col overflow-hidden">
            <PaneTitle>Activation Norms</PaneTitle>
            <div className="flex-1 overflow-hidden">
              <ActivationNorms norms={state.norms} numLayers={numLayers} />
            </div>
          </div>
          <div className="flex-1 flex flex-col overflow-hidden">
            <PaneTitle>Log</PaneTitle>
            <div className="flex-1 overflow-hidden">
              <LogPane metrics={state.metrics} events={state.events} />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

function PaneTitle({ children }: { children: React.ReactNode }) {
  return (
    <div className="px-2 py-0.5 text-[10px] text-gray-500 bg-gray-900 border-b border-gray-800 flex-shrink-0 uppercase tracking-wider">
      {children}
    </div>
  );
}
