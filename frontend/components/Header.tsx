"use client";

import React from "react";
import ApiStatus from "./ApiStatus";
import { Brain, Sparkles } from "lucide-react";

export default function Header() {
  return (
    <header className="w-full border-b border-[#263245] bg-[#0A0F1C]/90 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-[#1A2336] to-[#131A2A] border border-[#263245] flex items-center justify-center shadow-lg shadow-black/40 text-[#22D3EE] relative overflow-hidden group">
            <div className="absolute inset-0 bg-gradient-to-tr from-[#3B82F6]/20 to-[#22D3EE]/20 opacity-0 group-hover:opacity-100 transition-opacity" />
            <Brain className="w-5 h-5 text-[#22D3EE] relative z-10" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base sm:text-lg font-black tracking-tight text-[#F8FAFC]">
                NEUROSCAN <span className="text-[#3B82F6]">AI</span>
              </h1>
              <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-[#1A2336] text-[#22D3EE] border border-[#263245]">
                ResNet-50
              </span>
            </div>
            <p className="text-[11px] text-[#94A3B8] font-mono hidden sm:block">
              Deep Learning Brain Tumor MRI Classification
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <ApiStatus />
        </div>
      </div>
    </header>
  );
}
