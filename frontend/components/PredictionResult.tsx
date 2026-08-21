"use client";

import React from "react";
import { PredictionResponse, TumorClass } from "@/types/api";
import BrainViewer from "@/components/BrainViewer";
import { Activity, BarChart3, ShieldAlert, Sparkles } from "lucide-react";

interface PredictionResultProps {
  result: PredictionResponse;
}

const CLASS_CONFIG: Record<
  TumorClass,
  {
    title: string;
    description: string;
    color: string;
    bgColor: string;
    borderColor: string;
    badgeBg: string;
    progressColor: string;
  }
> = {
  glioma: {
    title: "Glioma",
    description: "Tumor arising from glial cells within brain tissue",
    color: "text-rose-400",
    bgColor: "bg-rose-500/10",
    borderColor: "border-rose-500/40",
    badgeBg: "bg-rose-500/20 text-rose-300 border-rose-500/40",
    progressColor: "bg-rose-500",
  },
  meningioma: {
    title: "Meningioma",
    description: "Tumor originating from the meningeal brain layer",
    color: "text-amber-400",
    bgColor: "bg-amber-500/10",
    borderColor: "border-amber-500/40",
    badgeBg: "bg-amber-500/20 text-amber-300 border-amber-500/40",
    progressColor: "bg-amber-500",
  },
  pituitary: {
    title: "Pituitary Tumor",
    description: "Growth formed within the pituitary gland region",
    color: "text-purple-400",
    bgColor: "bg-purple-500/10",
    borderColor: "border-purple-500/40",
    badgeBg: "bg-purple-500/20 text-purple-300 border-purple-500/40",
    progressColor: "bg-purple-500",
  },
  notumor: {
    title: "No Tumor Detected",
    description: "Normal brain MRI scan with no detectable tumor mass",
    color: "text-[#22C55E]",
    bgColor: "bg-[#22C55E]/10",
    borderColor: "border-[#22C55E]/40",
    badgeBg: "bg-[#22C55E]/20 text-[#22C55E] border-[#22C55E]/40",
    progressColor: "bg-[#22C55E]",
  },
};

const ALL_CLASSES: TumorClass[] = ["glioma", "meningioma", "pituitary", "notumor"];

export default function PredictionResult({ result }: PredictionResultProps) {
  const predictedClass = result.predicted_class as TumorClass;
  const config = CLASS_CONFIG[predictedClass] || CLASS_CONFIG.notumor;
  const confidencePercent = (result.confidence * 100).toFixed(2);

  return (
    <div className="w-full flex flex-col gap-6 animate-fade-in font-sans">
      {/* Primary Result Banner */}
      <div
        className={`w-full rounded-3xl border ${config.borderColor} ${config.bgColor} p-6 sm:p-7 flex flex-col sm:flex-row items-center justify-between gap-6 shadow-2xl backdrop-blur-xl`}
      >
        <div className="flex flex-col items-center sm:items-start text-center sm:text-left">
          <span
            className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-bold uppercase border ${config.badgeBg} mb-3`}
          >
            <Activity className="w-3.5 h-3.5" />
            PREDICTED DIAGNOSIS
          </span>
          <h2 className={`text-3xl sm:text-4xl font-extrabold tracking-tight ${config.color} mb-1.5`}>
            {config.title}
          </h2>
          <p className="text-xs text-[#94A3B8] max-w-md leading-relaxed">
            {config.description}
          </p>
        </div>

        <div className="flex flex-col items-center sm:items-end justify-center bg-[#0A0F1C]/90 p-4 sm:p-5 rounded-2xl border border-[#263245] shrink-0 min-w-[150px] shadow-inner font-mono">
          <span className="text-[10px] uppercase tracking-widest font-bold text-[#94A3B8] mb-1">
            CONFIDENCE
          </span>
          <span className={`text-3xl sm:text-4xl font-black ${config.color}`}>
            {confidencePercent}%
          </span>
        </div>
      </div>

      {/* 3D Volumetric Brain Reconstruction Viewer Panel */}
      <div className="w-full">
        <BrainViewer
          prediction={config.title}
          confidence={Number(confidencePercent)}
        />
      </div>

      {/* Class Probability Distribution Breakdown (4 Classes) */}
      <div className="w-full bg-[#131A2A] border border-[#263245] rounded-3xl p-6 shadow-xl">
        <div className="flex items-center justify-between border-b border-[#263245] pb-3 mb-4">
          <h3 className="text-xs font-mono font-bold text-[#F8FAFC] uppercase tracking-wider flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-[#22D3EE]" />
            Class Probability Distribution (4 Classes)
          </h3>
          <span className="text-[10px] font-mono text-[#94A3B8] bg-[#1A2336] border border-[#263245] px-2 py-0.5 rounded">
            ResNet-50 Output
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
          {ALL_CLASSES.map((clsKey) => {
            const rawProb = result.probabilities[clsKey] ?? 0;
            const probPercent = (rawProb * 100).toFixed(2);
            const clsCfg = CLASS_CONFIG[clsKey];
            const isTopClass = clsKey === predictedClass;

            return (
              <div
                key={clsKey}
                className={`p-4 rounded-2xl border transition-all ${
                  isTopClass
                    ? `${clsCfg.borderColor} ${clsCfg.bgColor} shadow-md`
                    : "border-[#263245] bg-[#0A0F1C]/60 hover:border-[#3B82F6]/40"
                }`}
              >
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs font-bold text-[#F8FAFC] flex items-center gap-2 font-mono">
                    {clsCfg.title}
                    {isTopClass && (
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-[#22D3EE]/20 text-[#22D3EE] font-mono font-bold border border-[#22D3EE]/30">
                        TOP
                      </span>
                    )}
                  </span>
                  <span className={`text-xs font-mono font-bold ${isTopClass ? clsCfg.color : "text-[#94A3B8]"}`}>
                    {probPercent}%
                  </span>
                </div>

                {/* Progress Bar */}
                <div className="w-full h-2 rounded-full bg-[#0A0F1C] border border-[#263245] overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${clsCfg.progressColor}`}
                    style={{ width: `${Math.max(Number(probPercent), 1)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
