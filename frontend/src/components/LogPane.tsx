"use client";

import { useEffect, useRef } from "react";
import type { EventMessage, Metrics } from "@/hooks/useVisualizerSocket";

interface Props {
  metrics: Metrics | null;
  events: EventMessage[];
}

const EVENT_COLORS: Record<string, string> = {
  agent_loaded:   "text-green-400",
  episode_start:  "text-blue-400",
  episode_end:    "text-yellow-400",
  error:          "text-red-400",
  connected:      "text-emerald-400",
  disconnected:   "text-orange-400",
};

function eventColor(event: string): string {
  return EVENT_COLORS[event] ?? "text-gray-400";
}

function MetricRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex justify-between text-[10px]">
      <span className="text-gray-500">{label}</span>
      <span className="text-gray-200 font-mono">{value}</span>
    </div>
  );
}

export default function LogPane({ metrics, events }: Props) {
  const logRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to top (newest events are prepended)
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = 0;
  }, [events.length]);

  return (
    <div className="w-full h-full flex flex-col bg-[#0d0d20] border-t border-gray-800 overflow-hidden">
      {/* Metrics grid */}
      {metrics && (
        <div className="px-2 pt-1.5 pb-1 border-b border-gray-800 flex-shrink-0 grid grid-cols-2 gap-x-4 gap-y-0.5">
          <MetricRow label="infer fps"     value={`${metrics.infer_fps.toFixed(1)} fps`} />
          <MetricRow label="episode"       value={metrics.episode} />
          <MetricRow label="step"          value={metrics.step} />
          <MetricRow label="return"        value={metrics.return.toFixed(1)} />
          <MetricRow label="queue depth"   value={metrics.queue_depth} />
          <MetricRow label="drop rate"     value={`${(metrics.drop_rate * 100).toFixed(1)}%`} />
          <MetricRow label="hook latency"  value={`${metrics.hook_latency_ms.toFixed(1)} ms`} />
        </div>
      )}

      {/* Event stream */}
      <div
        ref={logRef}
        className="flex-1 overflow-y-auto scrollbar-thin px-2 py-1 space-y-0.5"
      >
        {events.map((e) => (
          <div key={e.id} className="flex gap-1.5 items-baseline">
            <span className="text-[9px] text-gray-600 flex-shrink-0 font-mono">
              {e.timestamp}
            </span>
            <span className={`text-[9px] flex-shrink-0 font-medium ${eventColor(e.event)}`}>
              {e.event}
            </span>
            <span className="text-[9px] text-gray-500 truncate font-mono">
              {JSON.stringify(e.data)}
            </span>
          </div>
        ))}
        {events.length === 0 && (
          <div className="text-[10px] text-gray-700 pt-2">No events yet</div>
        )}
      </div>
    </div>
  );
}
