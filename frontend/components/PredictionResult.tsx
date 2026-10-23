"use client";

import React, { useState } from "react";
import { PredictionResponse, TumorClass, ClassScore } from "@/types/api";
import { downloadPdfReport } from "@/lib/api";
import {
  Activity,
  BarChart3,
  AlertTriangle,
  FileDown,
  Eye,
  Info,
  CheckCircle2,
  Loader2,
  AlertCircle,
  Sparkles,
} from "lucide-react";

interface PredictionResultProps {
  result: PredictionResponse;
  selectedFile?: File | null;
  imagePreview?: string | null;
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

const CANONICAL_CLASSES: TumorClass[] = ["glioma", "meningioma", "pituitary", "notumor"];

export default function PredictionResult({
  result,
  selectedFile,
  imagePreview,
}: PredictionResultProps) {
  const [isGeneratingPdf, setIsGeneratingPdf] = useState(false);
  const [pdfSuccess, setPdfSuccess] = useState(false);
  const [pdfError, setPdfError] = useState<string | null>(null);

  const predictedClass = (result?.predicted_class || "notumor") as TumorClass;
  const config = CLASS_CONFIG[predictedClass] || CLASS_CONFIG.notumor;
  const rawConfidence = typeof result?.confidence === "number" ? result.confidence : 0;
  const confidencePercent = (rawConfidence * 100).toFixed(2);
  const isLowConfidence = rawConfidence < 0.70;

  // Build canonical scores list with safe fallback
  const scores: ClassScore[] = React.useMemo(() => {
    if (result?.class_scores && Array.isArray(result.class_scores) && result.class_scores.length > 0) {
      return result.class_scores;
    }
    // Fallback using probabilities dict
    const probs = result?.probabilities || {};
    return CANONICAL_CLASSES.map((cls) => {
      const rawScore = typeof probs[cls] === "number" ? probs[cls] : 0;
      return {
        class_name: cls,
        display_name: CLASS_CONFIG[cls]?.title || cls,
        score: rawScore,
        percentage: Number((rawScore * 100).toFixed(2)),
        is_highest: cls === predictedClass,
      };
    });
  }, [result, predictedClass]);

  // Handle PDF report generation
  const handleDownloadPdf = async () => {
    if (!selectedFile || isGeneratingPdf) return;

    setIsGeneratingPdf(true);
    setPdfError(null);
    setPdfSuccess(false);

    try {
      const blob = await downloadPdfReport(selectedFile);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      const reportName = `neuroscan_report_${result?.prediction_id || Date.now()}.pdf`;
      link.setAttribute("download", reportName);
      document.body.appendChild(link);
      link.click();
      link.parentNode?.removeChild(link);
      window.URL.revokeObjectURL(url);
      setPdfSuccess(true);
      setTimeout(() => setPdfSuccess(false), 4000);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to download PDF report.";
      setPdfError(msg);
    } finally {
      setIsGeneratingPdf(false);
    }
  };

  return (
    <div className="w-full flex flex-col gap-6 animate-fade-in font-sans">
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

        <div className="flex flex-col items-center sm:items-end justify-center bg-[#0A0F1C]/90 p-4 sm:p-5 rounded-xl border border-[#263245] shrink-0 min-w-[150px] shadow-inner font-mono">
          <span className="text-[10px] uppercase tracking-widest font-bold text-[#94A3B8] mb-0.5">
            MODEL CONFIDENCE
          </span>
          <span className={`text-3xl sm:text-4xl font-black ${config.color} tracking-tight`}>
            {confidencePercent}%
          </span>
          {result?.prediction_id && (
            <span className="text-[9px] text-[#64748B] mt-1 font-mono">
              ID: {result.prediction_id}
            </span>
          )}
        </div>
      </div>

      {/* Cautious Low-Confidence Advisory Callout */}
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
              The model confidence for this slice is {confidencePercent}%. When diagnostic confidence is below 70%, radiological ambiguity is heightened. Review the alternative class distribution below and evaluate adjacent axial slices.
            </p>
          </div>
        </div>
      )}

      {/* Feature 1: Responsive Horizontal Confidence Bars Breakdown */}
      <div className="w-full bg-[#131A2A] border border-[#263245] rounded-2xl p-5 shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-[#263245] pb-3 mb-4 gap-2">
          <div>
            <h3 className="text-xs font-mono font-bold text-[#F8FAFC] uppercase tracking-wider flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-[#22D3EE]" />
              Multi-Class Confidence Score Distribution
            </h3>
            <p className="text-[11px] text-[#94A3B8] mt-0.5">
              Softmax probability distribution computed by ResNet-50 across all four diagnostic categories
            </p>
          </div>
          <span className="text-[10px] font-mono text-[#94A3B8] bg-[#1A2336] border border-[#263245] px-2.5 py-1 rounded shrink-0 self-start sm:self-auto">
            Softmax (Normalized to 100%)
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
          {scores.map((item) => {
            const clsKey = item.class_name as TumorClass;
            const clsCfg = CLASS_CONFIG[clsKey] || CLASS_CONFIG.notumor;
            const isTopClass = item.is_highest;
            const pct = Math.max(0, Math.min(100, isNaN(item.percentage) ? 0 : item.percentage));

            return (
              <div
                key={clsKey}
                className={`p-4 rounded-xl border transition-all ${
                  isTopClass
                    ? `${clsCfg.borderColor} ${clsCfg.bgColor} ring-1 ring-[#22D3EE]/30 shadow-lg`
                    : "border-[#263245] bg-[#0A0F1C]/60 hover:border-[#3B82F6]/30"
                }`}
              >
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs font-bold text-[#F8FAFC] flex items-center gap-2 font-mono">
                    <span>{clsCfg.title}</span>
                    {isTopClass && (
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-[#22D3EE]/20 text-[#22D3EE] font-mono font-bold border border-[#22D3EE]/30 tracking-tight flex items-center gap-1">
                        <Sparkles className="w-2.5 h-2.5" />
                        TOP SCORE
                      </span>
                    )}
                  </span>
                  <span className={`text-xs font-mono font-bold ${isTopClass ? clsCfg.color : "text-[#94A3B8]"}`}>
                    {pct.toFixed(2)}%
                  </span>
                </div>

                {/* Accessible Progress Bar */}
                <div
                  className="w-full h-2.5 rounded-full bg-[#0A0F1C] border border-[#263245] overflow-hidden"
                  role="progressbar"
                  aria-valuenow={pct}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-label={`${clsCfg.title} confidence: ${pct.toFixed(2)}%`}
                >
                  <div
                    className={`h-full rounded-full transition-all duration-700 ease-out motion-reduce:transition-none ${clsCfg.progressColor}`}
                    style={{ width: `${Math.max(pct, 0.5)}%` }}
                  />
                </div>

                <p className="text-[10px] text-[#64748B] mt-2 leading-tight">
                  {clsCfg.description}
                </p>
              </div>
            );
          })}
        </div>

        {/* Feature 1 Required Disclaimer */}
        <div className="mt-4 pt-3 border-t border-[#263245]/60 flex items-start gap-2 text-[11px] text-[#94A3B8]">
          <Info className="w-3.5 h-3.5 text-[#22D3EE] shrink-0 mt-0.5" />
          <p>
            <strong className="text-[#F8FAFC]">Notice:</strong> Model confidence reflects pattern correlation learned by the ResNet-50 backbone across training slices. It is not an empirical clinical probability, patient prognosis, or medical diagnosis.
          </p>
        </div>
      </div>

      {/* Feature 2: Side-by-Side Visual Explainability & Grad-CAM Heatmap */}
      <div className="w-full bg-[#131A2A] border border-[#263245] rounded-2xl p-5 shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-[#263245] pb-3 mb-4 gap-2">
          <div>
            <h3 className="text-xs font-mono font-bold text-[#F8FAFC] uppercase tracking-wider flex items-center gap-2">
              <Eye className="w-4 h-4 text-[#3B82F6]" />
              Visual Explainability: Grad-CAM Heatmap
            </h3>
            <p className="text-[11px] text-[#94A3B8] mt-0.5">
              Gradient-weighted Class Activation Mapping from target layer <code className="text-[#22D3EE]">layer4[-1]</code>
            </p>
          </div>
          <span className="text-[10px] font-mono text-[#22D3EE] bg-[#1A2336] border border-[#263245] px-2.5 py-1 rounded shrink-0 self-start sm:self-auto">
            Target Layer: ResNet-50 layer4[-1]
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Left Column: Original MRI Scan */}
          <div className="flex flex-col items-center bg-[#0A0F1C]/80 border border-[#263245] rounded-xl p-4">
            <span className="text-[11px] font-mono font-bold text-[#F8FAFC] mb-2 uppercase tracking-wide">
              Original Brain MRI Scan
            </span>
            <div className="w-full aspect-square max-w-[260px] rounded-lg overflow-hidden border border-[#263245] bg-[#0A0F1C] flex items-center justify-center relative">
              {imagePreview ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={imagePreview}
                  alt="Original Brain MRI Scan"
                  className="w-full h-full object-contain"
                />
              ) : (
                <div className="text-xs text-[#64748B] p-4 text-center">
                  Original image preview unavailable
                </div>
              )}
            </div>
            <p className="text-[10px] text-[#64748B] mt-2.5 text-center">
              Input axial slice normalized to 224 × 224 px
            </p>
          </div>

          {/* Right Column: Grad-CAM Heatmap Overlay */}
          <div className="flex flex-col items-center bg-[#0A0F1C]/80 border border-[#263245] rounded-xl p-4">
            <span className="text-[11px] font-mono font-bold text-[#22D3EE] mb-2 uppercase tracking-wide flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5 text-[#22D3EE]" />
              Grad-CAM Heatmap Overlay
            </span>
            <div className="w-full aspect-square max-w-[260px] rounded-lg overflow-hidden border border-[#263245] bg-[#0A0F1C] flex items-center justify-center relative">
              {result?.gradcam_image ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={result.gradcam_image}
                  alt="Grad-CAM Activation Heatmap Overlay"
                  className="w-full h-full object-contain"
                />
              ) : (
                <div className="text-xs text-[#64748B] p-4 text-center flex flex-col items-center gap-1.5">
                  <AlertCircle className="w-5 h-5 text-amber-400" />
                  <span>Grad-CAM heatmap unavailable</span>
                </div>
              )}
            </div>

            {/* Heatmap color gradient legend */}
            <div className="w-full max-w-[260px] mt-2.5 flex items-center justify-between text-[9px] font-mono text-[#94A3B8]">
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-blue-500 inline-block" />
                Baseline (Low)
              </span>
              <div className="h-1.5 flex-1 mx-2 rounded-full bg-gradient-to-r from-blue-600 via-green-500 via-yellow-400 to-red-600" />
              <span className="flex items-center gap-1">
                Salient (High)
                <span className="w-2 h-2 rounded-full bg-red-500 inline-block" />
              </span>
            </div>
          </div>
        </div>

        {/* Feature 2 Required Explainability Disclaimer */}
        <div className="mt-4 pt-3 border-t border-[#263245]/60 flex items-start gap-2 text-[11px] text-[#94A3B8]">
          <Info className="w-3.5 h-3.5 text-[#3B82F6] shrink-0 mt-0.5" />
          <p>
            <strong className="text-[#F8FAFC]">Explainability Note:</strong> The Grad-CAM heatmap highlights spatial regions and feature activations within <code className="text-[#22D3EE]">layer4[-1]</code> that most strongly influenced the neural network&apos;s classification. It illustrates computational focus, not confirmed histological boundaries or anatomical margins of a tumor.
          </p>
        </div>
      </div>

      {/* Feature 3: Downloadable PDF Clinical Research Report */}
      <div className="w-full bg-[#131A2A] border border-[#263245] rounded-2xl p-5 shadow-xl flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex flex-col text-center sm:text-left">
          <h4 className="text-xs font-mono font-bold text-[#F8FAFC] uppercase tracking-wider flex items-center justify-center sm:justify-start gap-2">
            <FileDown className="w-4 h-4 text-[#22C55E]" />
            Download Medical Research Report
          </h4>
          <p className="text-[11px] text-[#94A3B8] mt-1 max-w-lg leading-relaxed">
            Generate an authoritative, server-rendered PDF report containing model classification metrics, complete class score distribution, original MRI scan, Grad-CAM activation visualization, and regulatory disclaimers.
          </p>
        </div>

        <div className="flex flex-col items-center sm:items-end gap-2 shrink-0">
          <button
            onClick={handleDownloadPdf}
            disabled={!selectedFile || isGeneratingPdf}
            className={`px-5 py-2.5 rounded-xl font-mono font-bold text-xs flex items-center gap-2 transition-all shadow-lg ${
              !selectedFile || isGeneratingPdf
                ? "bg-[#1A2336] text-[#64748B] border border-[#263245] cursor-not-allowed"
                : "bg-[#22C55E] hover:bg-[#16A34A] text-[#0A0F1C] border border-[#22C55E] hover:shadow-[#22C55E]/20 hover:scale-[1.02] active:scale-[0.98]"
            }`}
          >
            {isGeneratingPdf ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin text-[#0A0F1C]" />
                <span>Generating PDF...</span>
              </>
            ) : pdfSuccess ? (
              <>
                <CheckCircle2 className="w-4 h-4 text-[#0A0F1C]" />
                <span>Report Downloaded!</span>
              </>
            ) : (
              <>
                <FileDown className="w-4 h-4 text-[#0A0F1C]" />
                <span>Download PDF Report</span>
              </>
            )}
          </button>

          {pdfError && (
            <span className="text-[10px] text-rose-400 font-mono text-center sm:text-right">
              {pdfError}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
