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
  ArrowLeft,
  FileText,
  Calendar,
  Cpu,
  Layers,
  ShieldAlert,
} from "lucide-react";

interface PredictionResultProps {
  result: PredictionResponse;
  selectedFile?: File | null;
  imagePreview?: string | null;
  onNewAnalysis?: () => void;
}

type TabType = "overview" | "explainability" | "report";

const CLASS_CONFIG: Record<
  TumorClass,
  {
    title: string;
    shortLabel: string;
    category: string;
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
    category: "Intra-axial Neoplasm",
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
    category: "Extra-axial Neoplasm",
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
    category: "Sellar / Endocrine Mass",
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
    category: "Normal Neuroanatomy",
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
  onNewAnalysis,
}: PredictionResultProps) {
  const [activeTab, setActiveTab] = useState<TabType>("overview");
  const [isGeneratingPdf, setIsGeneratingPdf] = useState(false);
  const [pdfSuccess, setPdfSuccess] = useState(false);
  const [pdfError, setPdfError] = useState<string | null>(null);

  const predictedClass = (result?.predicted_class || "notumor") as TumorClass;
  const config = CLASS_CONFIG[predictedClass] || CLASS_CONFIG.notumor;
  const rawConfidence = typeof result?.confidence === "number" ? result.confidence : 0;
  const confidencePercent = (rawConfidence * 100).toFixed(2);
  const isLowConfidence = rawConfidence < 0.70;

  // Safely convert authoritative UTC timestamp to browser local timezone
  const localTimestamp = React.useMemo(() => {
    try {
      const raw = result?.timestamp;
      const date = raw ? new Date(raw) : new Date();
      if (isNaN(date.getTime())) {
        return new Date().toLocaleString(undefined, {
          year: "numeric",
          month: "short",
          day: "numeric",
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          timeZoneName: "short",
        });
      }
      return date.toLocaleString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        timeZoneName: "short",
      });
    } catch {
      return new Date().toLocaleString();
    }
  }, [result?.timestamp]);

  // Build canonical scores list with safe fallback
  const scores: ClassScore[] = React.useMemo(() => {
    if (result?.class_scores && Array.isArray(result.class_scores) && result.class_scores.length > 0) {
      return result.class_scores;
    }
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
    <div className="w-full flex flex-col gap-5 animate-fade-in font-sans">
      {/* Compact Dashboard Top Bar */}
      <div className="w-full bg-[#131A2A] border border-[#263245] rounded-2xl p-4 sm:p-5 shadow-xl flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4">
        {/* Left: Back Action & Title */}
        <div className="flex items-center gap-3">
          {onNewAnalysis && (
            <button
              onClick={onNewAnalysis}
              className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl bg-[#1A2336] hover:bg-[#223049] text-[#94A3B8] hover:text-[#F8FAFC] border border-[#263245] hover:border-[#3B82F6]/50 text-xs font-mono font-semibold transition-all shadow-sm shrink-0"
              title="Return to upload screen for another MRI scan"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>New Analysis</span>
            </button>
          )}

          <div className="flex flex-col">
            <div className="flex items-center gap-2">
              <h2 className="text-sm sm:text-base font-mono font-bold uppercase tracking-wider text-[#F8FAFC]">
                Analysis Results
              </h2>
              {result?.prediction_id && (
                <span className="hidden sm:inline-block text-[10px] font-mono px-2 py-0.5 rounded bg-[#1A2336] text-[#64748B] border border-[#263245]">
                  {result.prediction_id}
                </span>
              )}
            </div>
            <p className="text-[11px] text-[#94A3B8]">
              Automated ResNet-50 diagnostic evaluation &amp; explainability
            </p>
          </div>
        </div>

        {/* Right: Primary Result Summary & Quick PDF Action */}
        <div className="flex items-center justify-between md:justify-end gap-3 flex-wrap">
          {/* Classification Verdict Pill */}
          <div className={`px-3 py-1.5 rounded-xl border ${config.borderColor} ${config.bgColor} flex items-center gap-2`}>
            <span className="text-[10px] font-mono uppercase text-[#94A3B8]">Class:</span>
            <span className={`text-xs font-bold font-mono ${config.color}`}>
              {config.title}
            </span>
          </div>

          {/* Confidence Score Pill */}
          <div className="px-3 py-1.5 rounded-xl bg-[#0A0F1C]/90 border border-[#263245] flex items-center gap-2 font-mono">
            <span className="text-[10px] uppercase text-[#94A3B8]">Confidence:</span>
            <span className={`text-xs sm:text-sm font-black ${config.color}`}>
              {confidencePercent}%
            </span>
          </div>

          {/* Quick PDF Download Button */}
          <button
            onClick={handleDownloadPdf}
            disabled={!selectedFile || isGeneratingPdf}
            className={`px-3 py-1.5 rounded-xl text-xs font-mono font-bold flex items-center gap-1.5 transition-all shadow-md ${
              !selectedFile || isGeneratingPdf
                ? "bg-[#1A2336] text-[#64748B] border border-[#263245] cursor-not-allowed"
                : "bg-[#22C55E] hover:bg-[#16A34A] text-[#0A0F1C] border border-[#22C55E] hover:shadow-[#22C55E]/20"
            }`}
            title="Download full PDF clinical research report"
          >
            {isGeneratingPdf ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span className="hidden sm:inline">Generating...</span>
              </>
            ) : pdfSuccess ? (
              <>
                <CheckCircle2 className="w-3.5 h-3.5 text-[#0A0F1C]" />
                <span className="hidden sm:inline">Downloaded!</span>
              </>
            ) : (
              <>
                <FileDown className="w-3.5 h-3.5" />
                <span>PDF Report</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Accessible Navigation Tabs */}
      <div
        className="w-full flex items-center gap-2 border-b border-[#263245] pb-2 overflow-x-auto no-scrollbar"
        role="tablist"
        aria-label="Result View Tabs"
      >
        <button
          role="tab"
          aria-selected={activeTab === "overview"}
          aria-controls="panel-overview"
          id="tab-overview"
          onClick={() => setActiveTab("overview")}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-mono font-bold transition-all border shrink-0 ${
            activeTab === "overview"
              ? "bg-[#1A2336] text-[#22D3EE] border-[#22D3EE]/40 shadow-sm"
              : "text-[#94A3B8] hover:text-[#F8FAFC] border-transparent hover:bg-[#131A2A]"
          }`}
        >
          <BarChart3 className="w-3.5 h-3.5" />
          <span>Overview</span>
        </button>

        <button
          role="tab"
          aria-selected={activeTab === "explainability"}
          aria-controls="panel-explainability"
          id="tab-explainability"
          onClick={() => setActiveTab("explainability")}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-mono font-bold transition-all border shrink-0 ${
            activeTab === "explainability"
              ? "bg-[#1A2336] text-[#22D3EE] border-[#22D3EE]/40 shadow-sm"
              : "text-[#94A3B8] hover:text-[#F8FAFC] border-transparent hover:bg-[#131A2A]"
          }`}
        >
          <Eye className="w-3.5 h-3.5" />
          <span>Explainability (Grad-CAM)</span>
        </button>

        <button
          role="tab"
          aria-selected={activeTab === "report"}
          aria-controls="panel-report"
          id="tab-report"
          onClick={() => setActiveTab("report")}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-mono font-bold transition-all border shrink-0 ${
            activeTab === "report"
              ? "bg-[#1A2336] text-[#22D3EE] border-[#22D3EE]/40 shadow-sm"
              : "text-[#94A3B8] hover:text-[#F8FAFC] border-transparent hover:bg-[#131A2A]"
          }`}
        >
          <FileText className="w-3.5 h-3.5" />
          <span>Clinical Report &amp; PDF</span>
        </button>
      </div>

      {/* TAB 1: OVERVIEW */}
      {activeTab === "overview" && (
        <div
          id="panel-overview"
          role="tabpanel"
          aria-labelledby="tab-overview"
          className="w-full flex flex-col gap-5 animate-fade-in"
        >
          {/* Low Confidence Advisory */}
          {isLowConfidence && (
            <div
              className="w-full p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs flex items-start gap-3 shadow-md"
              role="alert"
            >
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="font-bold font-mono uppercase text-[11px] tracking-wide mb-0.5 text-amber-200">
                  Elevated Classification Uncertainty (&lt;70.0%)
                </p>
                <p className="text-[11px] text-amber-300/90 leading-relaxed">
                  The model confidence for this scan is {confidencePercent}%. When diagnostic confidence is below 70%, radiological ambiguity is heightened. Review the alternative class distribution and evaluate adjacent axial slices.
                </p>
              </div>
            </div>
          )}

          {/* Responsive 2-Column Overview */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
            {/* Left Column: Verdict Card + Original Image Thumbnail */}
            <div className="lg:col-span-5 flex flex-col gap-4">
              {/* Primary Verdict Card */}
              <div className={`w-full rounded-2xl border ${config.borderColor} ${config.bgColor} p-5 shadow-xl backdrop-blur-md`}>
                <div className="flex items-center justify-between mb-2">
                  <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold uppercase border ${config.badgeBg}`}>
                    <Activity className="w-3 h-3" />
                    PRIMARY CLASSIFICATION
                  </span>
                  <span className="text-[10px] font-mono text-[#94A3B8]">
                    {config.category}
                  </span>
                </div>

                <h3 className={`text-2xl sm:text-3xl font-black tracking-tight ${config.color} mb-1.5`}>
                  {config.title}
                </h3>
                <p className="text-xs text-[#94A3B8] leading-relaxed mb-4">
                  {config.description}
                </p>

                <div className="flex items-center justify-between p-3 rounded-xl bg-[#0A0F1C]/90 border border-[#263245] font-mono">
                  <span className="text-[10px] text-[#94A3B8] uppercase">
                    Prediction Confidence
                  </span>
                  <span className={`text-xl font-black ${config.color}`}>
                    {confidencePercent}%
                  </span>
                </div>
              </div>

              {/* Original MRI Preview Box */}
              <div className="w-full bg-[#131A2A] border border-[#263245] rounded-2xl p-4 shadow-xl">
                <div className="flex items-center justify-between border-b border-[#263245] pb-2.5 mb-3 text-[11px] font-mono text-[#94A3B8]">
                  <span className="font-bold text-[#F8FAFC]">Input MRI Scan Slice</span>
                  <span>224 × 224 px</span>
                </div>

                <div className="w-full aspect-square max-w-[260px] mx-auto rounded-xl overflow-hidden border border-[#263245] bg-[#0A0F1C] flex items-center justify-center relative">
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

                {selectedFile?.name && (
                  <p className="text-[10px] font-mono text-[#64748B] mt-2.5 text-center truncate">
                    File: {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
                  </p>
                )}
              </div>
            </div>

            {/* Right Column: Multi-Class Confidence Bars */}
            <div className="lg:col-span-7 flex flex-col gap-4">
              <div className="w-full bg-[#131A2A] border border-[#263245] rounded-2xl p-5 shadow-xl">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-[#263245] pb-3 mb-4 gap-2">
                  <div>
                    <h3 className="text-xs font-mono font-bold text-[#F8FAFC] uppercase tracking-wider flex items-center gap-2">
                      <BarChart3 className="w-4 h-4 text-[#22D3EE]" />
                      Multi-Class Confidence Distribution
                    </h3>
                    <p className="text-[11px] text-[#94A3B8] mt-0.5">
                      Softmax probability output across all 4 diagnostic classes
                    </p>
                  </div>
                  <span className="text-[10px] font-mono text-[#94A3B8] bg-[#1A2336] border border-[#263245] px-2.5 py-1 rounded shrink-0 self-start sm:self-auto">
                    Normalized (100%)
                  </span>
                </div>

                <div className="flex flex-col gap-3">
                  {scores.map((item) => {
                    const clsKey = item.class_name as TumorClass;
                    const clsCfg = CLASS_CONFIG[clsKey] || CLASS_CONFIG.notumor;
                    const isTopClass = item.is_highest;
                    const pct = Math.max(0, Math.min(100, isNaN(item.percentage) ? 0 : item.percentage));

                    return (
                      <div
                        key={clsKey}
                        className={`p-3.5 rounded-xl border transition-all ${
                          isTopClass
                            ? `${clsCfg.borderColor} ${clsCfg.bgColor} ring-1 ring-[#22D3EE]/30 shadow-md`
                            : "border-[#263245] bg-[#0A0F1C]/60 hover:border-[#3B82F6]/30"
                        }`}
                      >
                        <div className="flex justify-between items-center mb-1.5">
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

                        <p className="text-[10px] text-[#64748B] mt-1.5 leading-tight">
                          {clsCfg.description}
                        </p>
                      </div>
                    );
                  })}
                </div>

                {/* Feature 1 Disclaimer */}
                <div className="mt-4 pt-3 border-t border-[#263245]/60 flex items-start gap-2 text-[11px] text-[#94A3B8]">
                  <Info className="w-3.5 h-3.5 text-[#22D3EE] shrink-0 mt-0.5" />
                  <p>
                    <strong className="text-[#F8FAFC]">Notice:</strong> Model confidence reflects pattern correlation learned by the ResNet-50 backbone across training slices. It is not an empirical clinical probability, patient prognosis, or medical diagnosis.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: EXPLAINABILITY (GRAD-CAM) */}
      {activeTab === "explainability" && (
        <div
          id="panel-explainability"
          role="tabpanel"
          aria-labelledby="tab-explainability"
          className="w-full flex flex-col gap-5 animate-fade-in"
        >
          <div className="w-full bg-[#131A2A] border border-[#263245] rounded-2xl p-5 shadow-xl">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-[#263245] pb-3 mb-4 gap-2">
              <div>
                <h3 className="text-xs font-mono font-bold text-[#F8FAFC] uppercase tracking-wider flex items-center gap-2">
                  <Eye className="w-4 h-4 text-[#3B82F6]" />
                  Visual Explainability: Grad-CAM Activation Heatmap
                </h3>
                <p className="text-[11px] text-[#94A3B8] mt-0.5">
                  Gradient-weighted Class Activation Mapping computed from final layer <code className="text-[#22D3EE]">layer4[-1]</code>
                </p>
              </div>
              <span className="text-[10px] font-mono text-[#22D3EE] bg-[#1A2336] border border-[#263245] px-2.5 py-1 rounded shrink-0 self-start sm:self-auto">
                Target Layer: ResNet-50 layer4[-1]
              </span>
            </div>

            {/* Side-by-Side Visual Comparison */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {/* Left Column: Original MRI Scan */}
              <div className="flex flex-col items-center bg-[#0A0F1C]/80 border border-[#263245] rounded-xl p-4">
                <span className="text-[11px] font-mono font-bold text-[#F8FAFC] mb-2 uppercase tracking-wide">
                  Input Brain MRI Scan
                </span>
                <div className="w-full aspect-square max-w-[280px] rounded-lg overflow-hidden border border-[#263245] bg-[#0A0F1C] flex items-center justify-center relative shadow-inner">
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
                  Axial T1/T2 scan slice normalized to 224 × 224 px
                </p>
              </div>

              {/* Right Column: Grad-CAM Heatmap Overlay */}
              <div className="flex flex-col items-center bg-[#0A0F1C]/80 border border-[#263245] rounded-xl p-4">
                <span className="text-[11px] font-mono font-bold text-[#22D3EE] mb-2 uppercase tracking-wide flex items-center gap-1.5">
                  <Activity className="w-3.5 h-3.5 text-[#22D3EE]" />
                  Grad-CAM Heatmap Overlay
                </span>
                <div className="w-full aspect-square max-w-[280px] rounded-lg overflow-hidden border border-[#263245] bg-[#0A0F1C] flex items-center justify-center relative shadow-inner">
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
                <div className="w-full max-w-[280px] mt-2.5 flex items-center justify-between text-[9px] font-mono text-[#94A3B8]">
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

            {/* Feature 2 Explainability Disclaimer */}
            <div className="mt-5 pt-3.5 border-t border-[#263245]/60 flex items-start gap-2.5 text-[11px] text-[#94A3B8]">
              <Info className="w-4 h-4 text-[#3B82F6] shrink-0 mt-0.5" />
              <p className="leading-relaxed">
                <strong className="text-[#F8FAFC]">Explainability Note:</strong> The Grad-CAM heatmap highlights spatial regions and feature activations within <code className="text-[#22D3EE]">layer4[-1]</code> that most strongly influenced the neural network&apos;s classification. It illustrates computational focus, not confirmed histological boundaries or anatomical margins of a tumor.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: REPORT & PDF DOWNLOAD */}
      {activeTab === "report" && (
        <div
          id="panel-report"
          role="tabpanel"
          aria-labelledby="tab-report"
          className="w-full flex flex-col gap-5 animate-fade-in"
        >
          {/* Diagnostic Report Summary */}
          <div className="w-full bg-[#131A2A] border border-[#263245] rounded-2xl p-5 shadow-xl">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-[#263245] pb-3 mb-4 gap-2">
              <div>
                <h3 className="text-xs font-mono font-bold text-[#F8FAFC] uppercase tracking-wider flex items-center gap-2">
                  <FileText className="w-4 h-4 text-[#22C55E]" />
                  Clinical Research Report Summary
                </h3>
                <p className="text-[11px] text-[#94A3B8] mt-0.5">
                  Server-generated authoritative summary ready for PDF export
                </p>
              </div>
              <span className="text-[10px] font-mono text-[#94A3B8] bg-[#1A2336] border border-[#263245] px-2.5 py-1 rounded shrink-0 self-start sm:self-auto">
                Model: ResNet-50 Fine-Tuned
              </span>
            </div>

            {/* Report Metadata Strip */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4 text-xs font-mono">
              <div className="bg-[#0A0F1C]/70 p-3 rounded-xl border border-[#263245]">
                <span className="text-[9px] text-[#94A3B8] block mb-0.5">REPORT ID</span>
                <span className="font-bold text-[#F8FAFC]">{result?.prediction_id || "REP-STANDALONE"}</span>
              </div>
              <div className="bg-[#0A0F1C]/70 p-3 rounded-xl border border-[#263245]">
                <span className="text-[9px] text-[#94A3B8] block mb-0.5">LOCAL TIME</span>
                <span className="font-bold text-[#22D3EE] text-[11px] truncate block" title={localTimestamp}>
                  {localTimestamp}
                </span>
              </div>
              <div className="bg-[#0A0F1C]/70 p-3 rounded-xl border border-[#263245]">
                <span className="text-[9px] text-[#94A3B8] block mb-0.5">PRIMARY VERDICT</span>
                <span className={`font-bold ${config.color}`}>{config.title}</span>
              </div>
              <div className="bg-[#0A0F1C]/70 p-3 rounded-xl border border-[#263245]">
                <span className="text-[9px] text-[#94A3B8] block mb-0.5">CONFIDENCE</span>
                <span className="font-bold text-[#22C55E]">{confidencePercent}%</span>
              </div>
            </div>

            {/* Probability Breakdown Table */}
            <div className="w-full overflow-x-auto border border-[#263245] rounded-xl mb-5">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-[#1A2336] text-[#94A3B8] border-b border-[#263245]">
                  <tr>
                    <th className="py-2.5 px-3 font-semibold">Diagnostic Class</th>
                    <th className="py-2.5 px-3 font-semibold">Category</th>
                    <th className="py-2.5 px-3 font-semibold text-right">Probability</th>
                    <th className="py-2.5 px-3 font-semibold text-right">Percentage</th>
                    <th className="py-2.5 px-3 font-semibold text-center">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#263245]/60">
                  {scores.map((s) => {
                    const isTop = s.is_highest;
                    return (
                      <tr
                        key={s.class_name}
                        className={isTop ? "bg-[#22D3EE]/5 text-[#F8FAFC]" : "text-[#94A3B8] hover:bg-[#1A2336]/40"}
                      >
                        <td className="py-2.5 px-3 font-bold">
                          {s.display_name}
                        </td>
                        <td className="py-2.5 px-3 text-[#64748B]">
                          {CLASS_CONFIG[s.class_name as TumorClass]?.category || "Neoplasm"}
                        </td>
                        <td className="py-2.5 px-3 text-right">
                          {s.score.toFixed(5)}
                        </td>
                        <td className={`py-2.5 px-3 text-right font-bold ${isTop ? config.color : ""}`}>
                          {s.percentage.toFixed(2)}%
                        </td>
                        <td className="py-2.5 px-3 text-center">
                          {isTop ? (
                            <span className="inline-block px-2 py-0.5 text-[9px] rounded bg-[#22D3EE]/20 text-[#22D3EE] font-bold border border-[#22D3EE]/30">
                              PREDICTED
                            </span>
                          ) : (
                            <span className="text-[10px] text-[#64748B]">Alternative</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* PDF Download Hub Card */}
            <div className="w-full bg-[#0A0F1C]/80 border border-[#263245] rounded-xl p-4 flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="flex flex-col text-center sm:text-left">
                <span className="text-xs font-mono font-bold text-[#F8FAFC] flex items-center justify-center sm:justify-start gap-1.5">
                  <FileDown className="w-3.5 h-3.5 text-[#22C55E]" />
                  Download Complete PDF Clinical Report
                </span>
                <span className="text-[11px] text-[#94A3B8] mt-0.5">
                  Includes full metric tables, original scan, Grad-CAM activation overlay, and regulatory notice.
                </span>
              </div>

              <div className="flex flex-col items-center sm:items-end gap-1.5 shrink-0">
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
                      <span>Rendering PDF...</span>
                    </>
                  ) : pdfSuccess ? (
                    <>
                      <CheckCircle2 className="w-4 h-4 text-[#0A0F1C]" />
                      <span>PDF Downloaded!</span>
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

            {/* Disclaimer Box in Report Tab */}
            <div className="mt-4 pt-3.5 border-t border-[#263245]/60 flex items-start gap-2.5 text-[11px] text-[#94A3B8]">
              <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <p className="leading-relaxed">
                <strong className="text-amber-200">Regulatory Disclaimer:</strong> This PDF document is generated by an automated computational deep learning model (NeuroScan AI) strictly for educational and academic research evaluation. It is <strong>NOT</strong> an FDA-approved medical diagnostic device and must not be used to direct clinical patient care, treatment planning, or medication.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
