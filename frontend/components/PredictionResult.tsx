"use client";

import React from "react";
import { PredictionResponse, TumorClass } from "@/types/api";

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
    description: "Tumor that arises from glial cells in the brain tissue",
    color: "text-rose-400",
    bgColor: "bg-rose-500/10",
    borderColor: "border-rose-500/30",
    badgeBg: "bg-rose-500/20 text-rose-300 border-rose-500/30",
    progressColor: "bg-rose-500",
  },
  meningioma: {
    title: "Meningioma",
    description: "Tumor originating from the meninges surrounding the brain",
    color: "text-amber-400",
    bgColor: "bg-amber-500/10",
    borderColor: "border-amber-500/30",
    badgeBg: "bg-amber-500/20 text-amber-300 border-amber-500/30",
    progressColor: "bg-amber-500",
  },
  pituitary: {
    title: "Pituitary Tumor",
    description: "Growth formed in the pituitary gland at the base of the brain",
    color: "text-purple-400",
    bgColor: "bg-purple-500/10",
    borderColor: "border-purple-500/30",
    badgeBg: "bg-purple-500/20 text-purple-300 border-purple-500/30",
    progressColor: "bg-purple-500",
  },
  notumor: {
    title: "No Tumor Detected",
    description: "Normal brain MRI scan with no detectable tumor mass",
    color: "text-emerald-400",
    bgColor: "bg-emerald-500/10",
    borderColor: "border-emerald-500/30",
    badgeBg: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
    progressColor: "bg-emerald-500",
  },
};

const ALL_CLASSES: TumorClass[] = ["glioma", "meningioma", "pituitary", "notumor"];

export default function PredictionResult({ result }: PredictionResultProps) {
  const predictedClass = result.predicted_class as TumorClass;
  const config = CLASS_CONFIG[predictedClass] || CLASS_CONFIG.notumor;
  const confidencePercent = (result.confidence * 100).toFixed(2);

  return (
    <div className="w-full flex flex-col gap-6 animate-fade-in">
      {/* Primary Result Banner */}
      <div
        className={`w-full rounded-2xl border ${config.borderColor} ${config.bgColor} p-6 sm:p-8 flex flex-col sm:flex-row items-center justify-between gap-6 shadow-xl backdrop-blur-sm`}
      >
        <div className="flex flex-col items-center sm:items-start text-center sm:text-left">
          <span
            className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold border ${config.badgeBg} mb-3`}
          >
            Predicted Diagnosis
          </span>
          <h2 className={`text-3xl sm:text-4xl font-extrabold tracking-tight ${config.color} mb-1`}>
            {config.title}
          </h2>
          <p className="text-xs text-slate-300 max-w-md">{config.description}</p>
        </div>

        <div className="flex flex-col items-center sm:items-end justify-center bg-slate-950/60 p-4 rounded-xl border border-slate-800 shrink-0 min-w-[140px]">
          <span className="text-[10px] uppercase tracking-wider font-semibold text-slate-400 mb-1">
            Confidence
          </span>
          <span className={`text-3xl font-black ${config.color}`}>{confidencePercent}%</span>
        </div>
      </div>

      {/* Class Probability Distribution Breakdown */}
      <div className="w-full bg-slate-900/60 rounded-2xl border border-slate-800 p-6 shadow-lg">
        <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider mb-4 flex items-center gap-2">
          <svg className="w-4 h-4 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          Probability Distribution (4 Classes)
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {ALL_CLASSES.map((clsKey) => {
            const rawProb = result.probabilities[clsKey] ?? 0;
            const probPercent = (rawProb * 100).toFixed(2);
            const clsCfg = CLASS_CONFIG[clsKey];
            const isTopClass = clsKey === predictedClass;

            return (
              <div
                key={clsKey}
                className={`p-4 rounded-xl border transition-all ${
                  isTopClass
                    ? `${clsCfg.borderColor} ${clsCfg.bgColor} shadow-md`
                    : "border-slate-800/80 bg-slate-950/40 hover:border-slate-700"
                }`}
              >
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs font-bold text-slate-200 flex items-center gap-2">
                    {clsCfg.title}
                    {isTopClass && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 font-semibold">
                        TOP
                      </span>
                    )}
                  </span>
                  <span className={`text-xs font-mono font-bold ${isTopClass ? clsCfg.color : "text-slate-400"}`}>
                    {probPercent}%
                  </span>
                </div>

                {/* Progress bar */}
                <div className="w-full h-2 rounded-full bg-slate-950 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${clsCfg.progressColor}`}
                    style={{ width: `${Math.max(Number(probPercent), 1)}%` }}
                  ></div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
