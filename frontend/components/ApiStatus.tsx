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
    <div className="flex items-center gap-2 bg-slate-900/80 px-3 py-1.5 rounded-full border border-slate-800 text-xs">
      <span className="relative flex h-2.5 w-2.5">
        {isHealthy === true && (
          <>
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
          </>
        )}
        {isHealthy === false && (
          <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-rose-500"></span>
        )}
        {isHealthy === null && (
          <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-amber-500 animate-pulse"></span>
        )}
      </span>
      <span className="text-slate-300 font-medium">
        {isHealthy === true && "API Online"}
        {isHealthy === false && "API Offline"}
        {isHealthy === null && (isChecking ? "Checking..." : "Connecting...")}
      </span>
    </div>
  );
}
