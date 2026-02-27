"use client";

import type { AgentInfo, ControlCommand, ModelConfig } from "@/hooks/useVisualizerSocket";

interface Props {
  agents: AgentInfo[];
  selectedAgent: string;
  onAgentChange: (agent: AgentInfo) => void;
  availableDevices: string[];
  selectedDevice: string;
  onDeviceChange: (device: string) => void;
  selectedLayer: number;
  maxLayer: number;
  onLayerChange: (layer: number) => void;
  connected: boolean;
  loading: boolean;
  sendControl: (cmd: ControlCommand) => Promise<void>;
}

export default function ControlBar({
  agents,
  selectedAgent,
  onAgentChange,
  availableDevices,
  selectedDevice,
  onDeviceChange,
  selectedLayer,
  maxLayer,
  onLayerChange,
  connected,
  loading,
  sendControl,
}: Props) {
  function handleAgentChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const agent = agents.find((a) => a.id === e.target.value);
    if (!agent) return;
    onAgentChange(agent);
    sendControl({
      command: "switch_agent",
      payload: {
        checkpoint_path: agent.path,
        env_id: agent.env_id,
        device: selectedDevice,
      },
    });
  }

  function handleDeviceChange(e: React.ChangeEvent<HTMLSelectElement>) {
    onDeviceChange(e.target.value);
    // If an agent is already selected, reload it on the new device
    const agent = agents.find((a) => a.id === selectedAgent);
    if (agent) {
      sendControl({
        command: "switch_agent",
        payload: {
          checkpoint_path: agent.path,
          env_id: agent.env_id,
          device: e.target.value,
        },
      });
    }
  }

  const btn =
    "px-3 py-1 rounded text-xs font-medium border border-gray-700 hover:border-indigo-500 hover:text-indigo-300 transition-colors disabled:opacity-40";

  return (
    <header className="flex items-center gap-4 px-4 py-2 bg-gray-900 border-b border-gray-800 flex-shrink-0 flex-wrap">
      {/* Agent selector */}
      <div className="flex items-center gap-2">
        <label className="text-xs text-gray-400 whitespace-nowrap">Agent</label>
        <select
          className="bg-gray-800 text-gray-100 text-xs rounded px-2 py-1 border border-gray-700 focus:outline-none focus:border-indigo-500"
          value={selectedAgent}
          onChange={handleAgentChange}
          disabled={loading}
        >
          {agents.length === 0 && (
            <option value="">No agents found</option>
          )}
          {agents.map((a) => (
            <option key={a.id} value={a.id}>
              {a.name}
            </option>
          ))}
        </select>
      </div>

      {/* Device selector */}
      <div className="flex items-center gap-2">
        <label className="text-xs text-gray-400 whitespace-nowrap">Device</label>
        <select
          className="bg-gray-800 text-gray-100 text-xs rounded px-2 py-1 border border-gray-700 focus:outline-none focus:border-indigo-500"
          value={selectedDevice}
          onChange={handleDeviceChange}
          disabled={loading}
        >
          {availableDevices.map((d) => (
            <option key={d} value={d}>
              {d}
            </option>
          ))}
        </select>
      </div>

      {/* Episode controls */}
      <div className="flex items-center gap-1.5">
        <button
          className={btn}
          onClick={() => sendControl({ command: "restart" })}
          title="Restart episode"
        >
          Restart
        </button>
        <button
          className={btn}
          onClick={() => sendControl({ command: "pause" })}
          title="Pause inference"
        >
          Pause
        </button>
        <button
          className={btn}
          onClick={() => sendControl({ command: "resume" })}
          title="Resume inference"
        >
          Resume
        </button>
        <button
          className={btn}
          onClick={() => sendControl({ command: "loop", payload: { enabled: true } })}
          title="Enable episode looping"
        >
          Loop
        </button>
      </div>

      {/* Layer slider */}
      <div className="flex items-center gap-2 flex-1 min-w-40 max-w-64">
        <label className="text-xs text-gray-400 whitespace-nowrap">
          Layer
        </label>
        <input
          type="range"
          min={0}
          max={maxLayer}
          value={selectedLayer}
          onChange={(e) => onLayerChange(Number(e.target.value))}
          className="flex-1 accent-indigo-500"
        />
        <span className="text-xs text-gray-300 w-6 text-right font-mono">
          {selectedLayer}
        </span>
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Connection indicator */}
      <div className="flex items-center gap-1.5">
        <div
          className={`w-2 h-2 rounded-full ${
            connected ? "bg-green-400 animate-pulse" : "bg-red-500"
          }`}
        />
        <span className="text-xs text-gray-400">
          {loading ? "Loading…" : connected ? "Connected" : "Disconnected"}
        </span>
      </div>
    </header>
  );
}
