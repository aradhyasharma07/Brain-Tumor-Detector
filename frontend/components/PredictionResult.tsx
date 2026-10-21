"use client";

import React from "react";
import { PredictionResponse, TumorClass } from "@/types/api";
import { Activity, BarChart3, AlertTriangle, ShieldCheck } from "lucide-react";

interface PredictionResultProps {
  result: PredictionResponse;
}

const CLASS_CONFIG: Record<
  TumorClass,
  {
    title: string;
    shortLabel: string;
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
    shortLabel: "Glioma",
    description: "Intra-axial tumor originating from glial cells within brain tissue",
    color: "text-rose-400",
    bgColor: "bg-rose-500/10",
    borderColor: "border-rose-500/40",
    badgeBg: "bg-rose-500/20 text-rose-300 border-rose-500/40",
    progressColor: "bg-rose-500",
  },
  meningioma: {
    title: "Meningioma",
    shortLabel: "Meningioma",
    description: "Extra-axial tumor arising from the meningeal membranes covering the brain",
    color: "text-amber-400",
    bgColor: "bg-amber-500/10",
    borderColor: "border-amber-500/40",
    badgeBg: "bg-amber-500/20 text-amber-300 border-amber-500/40",
    progressColor: "bg-amber-500",
  },
  pituitary: {
    title: "Pituitary Tumor",
    shortLabel: "Pituitary",
    description: "Neuroendocrine tumor developing within the sella turcica / pituitary gland region",
    color: "text-purple-400",
    bgColor: "bg-purple-500/10",
    borderColor: "border-purple-500/40",
    badgeBg: "bg-purple-500/20 text-purple-300 border-purple-500/40",
    progressColor: "bg-purple-500",
  },
  notumor: {
    title: "No Tumor Detected",
    shortLabel: "No Tumor",
    description: "Normal neuroanatomy with no detectable intracranial tumor mass in this scan slice",
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
  const isLowConfidence = result.confidence < 0.70;

  return (
    <div className="w-full flex flex-col gap-5 animate-fade-in font-sans">
      {/* Primary Result Banner */}
      <div
        className={`w-full rounded-2xl border ${config.borderColor} ${config.bgColor} p-5 sm:p-6 flex flex-col sm:flex-row items-center justify-between gap-5 shadow-2xl backdrop-blur-xl transition-all`}
      >
        <div className="flex flex-col items-center sm:items-start text-center sm:text-left">
          <span
            className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-mono font-bold uppercase border ${config.badgeBg} mb-2.5`}
          >
            <Activity className="w-3.5 h-3.5" />
            PRIMARY CLASSIFICATION
          </span>
          <h2 className={`text-2xl sm:text-3xl font-extrabold tracking-tight ${config.color} mb-1`}>
            {config.title}
          </h2>
          <p className="text-xs text-[#94A3B8] max-w-md leading-relaxed">
            {config.description}
          </p>
        </div>

        <div className="flex flex-col items-center sm:items-end justify-center bg-[#0A0F1C]/90 p-4 sm:p-5 rounded-xl border border-[#263245] shrink-0 min-w-[140px] shadow-inner font-mono">
          <span className="text-[10px] uppercase tracking-widest font-bold text-[#94A3B8] mb-0.5">
            CONFIDENCE
          </span>
          <span className={`text-3xl sm:text-4xl font-black ${config.color} tracking-tight`}>
            {confidencePercent}%
          </span>
        </div>
      </div>

      {/* Cautious Low-Confidence Advisory Callout (Requirement 9) */}
      {isLowConfidence && (
        <div
          className="w-full p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs flex items-start gap-3 shadow-md animate-fade-in"
          role="alert"
        >
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="font-bold font-mono uppercase text-[11px] tracking-wide mb-0.5 text-amber-200">
              ELEVATED CLASSIFICATION UNCERTAINTY (&lt;70.0%)
            </p>
            <p className="text-[11px] text-amber-300/90 leading-relaxed">
              The model confidence for this slice is {confidencePercent}%. When diagnostic confidence is below 70%, radiological ambiguity is heightened. Review the alternative class distribution below and evaluate adjacent axial slices or multi-planar views.
            </p>
          </div>
        </div>
      )}

      {/* Class Probability Distribution Breakdown (4 Classes with Short Labels) */}
      <div className="w-full bg-[#131A2A] border border-[#263245] rounded-2xl p-5 shadow-xl">
        <div className="flex items-center justify-between border-b border-[#263245] pb-3 mb-4">
          <h3 className="text-xs font-mono font-bold text-[#F8FAFC] uppercase tracking-wider flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-[#22D3EE]" />
            Probability Distribution
          </h3>
          <span className="text-[10px] font-mono text-[#94A3B8] bg-[#1A2336] border border-[#263245] px-2.5 py-0.5 rounded">
            Softmax Output
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {ALL_CLASSES.map((clsKey) => {
            const rawProb = result.probabilities[clsKey] ?? 0;
            const probPercent = (rawProb * 100).toFixed(2);
            const clsCfg = CLASS_CONFIG[clsKey];
            const isTopClass = clsKey === predictedClass;

            return (
              <div
                key={clsKey}
                className={`p-3.5 rounded-xl border transition-all ${
                  isTopClass
                    ? `${clsCfg.borderColor} ${clsCfg.bgColor} shadow-sm`
                    : "border-[#263245] bg-[#0A0F1C]/60 hover:border-[#3B82F6]/30"
                }`}
              >
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs font-bold text-[#F8FAFC] flex items-center gap-1.5 font-mono">
                    {/* Consistent Short Label in Probability Grid (Requirement 8) */}
                    <span>{clsCfg.shortLabel}</span>
                    {isTopClass && (
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-[#22D3EE]/20 text-[#22D3EE] font-mono font-bold border border-[#22D3EE]/30 tracking-tight">
                        TOP
                      </span>
                    )}
                  </span>
                  <span className={`text-xs font-mono font-bold ${isTopClass ? clsCfg.color : "text-[#94A3B8]"}`}>
                    {probPercent}%
                  </span>
                </div>

                {/* Progress Bar with smooth transition */}
                <div className="w-full h-2 rounded-full bg-[#0A0F1C] border border-[#263245] overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-700 ease-out motion-reduce:transition-none ${clsCfg.progressColor}`}
                    style={{ width: `${Math.max(Number(probPercent), 0.5)}%` }}
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
