"use client";

import React, { useEffect, useState } from "react";
import { checkBackendHealth } from "@/lib/api";

export default function ApiStatus() {
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null);
  const [isChecking, setIsChecking] = useState<boolean>(false);

  const verifyHealth = async () => {
    setIsChecking(true);
    const online = await checkBackendHealth();
    setIsHealthy(online);
    setIsChecking(false);
  };

  useEffect(() => {
    verifyHealth();
    const interval = setInterval(verifyHealth, 15000); // Check every 15s
    return () => clearInterval(interval);
  }, []);

  return (
    <div
      className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#131A2A] border border-[#263245] text-xs font-mono transition-colors shadow-sm"
      role="status"
      aria-live="polite"
    >
      <span className="relative flex h-2 w-2">
        {isHealthy === true && (
          <>
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#22C55E] opacity-75 motion-reduce:hidden" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-[#22C55E]" />
          </>
        )}
        {isHealthy === false && (
          <span className="relative inline-flex rounded-full h-2 w-2 bg-[#EF4444]" />
        )}
        {isHealthy === null && (
          <span className="relative inline-flex rounded-full h-2 w-2 bg-[#F59E0B] animate-pulse motion-reduce:animate-none" />
        )}
      </span>
      <span className="text-[#94A3B8] text-[11px] font-medium">
        {isHealthy === true && <span className="text-[#22C55E]">Inference API Online</span>}
        {isHealthy === false && <span className="text-[#EF4444]">Inference API Offline</span>}
        {isHealthy === null && (isChecking ? "Checking API..." : "Connecting...")}
      </span>
    </div>
  );
}
