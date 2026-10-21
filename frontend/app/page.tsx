"use client";

import React, { useState } from "react";
import Header from "@/components/Header";
import UploadZone from "@/components/UploadZone";
import PredictionResult from "@/components/PredictionResult";
import { predictMriImage } from "@/lib/api";
import { PredictionResponse } from "@/types/api";
import {
  Brain,
  CheckCircle2,
  FileCheck2,
  Layers,
  Zap,
  Radio,
  AlertTriangle,
  Cpu,
  Shield,
} from "lucide-react";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);

  const handleFileSelected = (file: File) => {
    // Clear stale prediction and errors on new file selection (Requirement 10)
    setSelectedFile(file);
    setError(null);
    setPrediction(null);
    const objectUrl = URL.createObjectURL(file);
    setImagePreview(objectUrl);
  };

  const handleClear = () => {
    if (imagePreview) {
      URL.revokeObjectURL(imagePreview);
    }
    setSelectedFile(null);
    setImagePreview(null);
    setError(null);
    setPrediction(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile || isLoading) return;

    setIsLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const result = await predictMriImage(selectedFile);
      setPrediction(result);
    } catch (err: unknown) {
      const msg =
        err instanceof Error
          ? err.message
          : "An unexpected error occurred while communicating with the inference server.";
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  };

  // Structured benchmark statistics: Test Accuracy primary, Validation secondary (Requirement 1 & 2)
  const stats = [
    {
      value: "95.06%",
      label: "Test Accuracy (Primary)",
      description: "Evaluated on 1,600 unseen test MRI scans",
      badge: "TEST BENCHMARK",
      highlight: true,
      icon: <CheckCircle2 className="w-4 h-4 text-[#22D3EE]" />,
    },
    {
      value: "97.77%",
      label: "Validation Accuracy",
      description: "Peak model epoch performance",
      badge: "VALIDATION",
      highlight: false,
      icon: <Zap className="w-4 h-4 text-[#3B82F6]" />,
    },
    {
      value: "1,600",
      label: "Test MRI Images",
      description: "Balanced test split (400 per class)",
      badge: "TEST SPLIT",
      highlight: false,
      icon: <FileCheck2 className="w-4 h-4 text-[#22D3EE]" />,
    },
    {
      value: "4",
      label: "Diagnostic Classes",
      description: "Glioma, Meningioma, Pituitary & No Tumor",
      badge: "MULTI-CLASS",
      highlight: false,
      icon: <Layers className="w-4 h-4 text-[#22C55E]" />,
    },
  ];

  return (
    <div className="min-h-screen bg-[#0A0F1C] text-[#F8FAFC] flex flex-col font-sans selection:bg-[#22D3EE] selection:text-[#0A0F1C] relative overflow-hidden">
      {/* Subtle Blueprint Mesh Background (2% opacity) */}
      <div
        className="absolute inset-0 bg-[linear-gradient(to_right,#26324512_1px,transparent_1px),linear-gradient(to_bottom,#26324512_1px,transparent_1px)] bg-[size:32px_32px] pointer-events-none z-0"
        aria-hidden="true"
      />

      <Header />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 flex flex-col gap-6 sm:gap-8 relative z-10">
        {/* Research-Only Medical Advisory Notice (Requirement 3) */}
        <div
          className="w-full rounded-2xl bg-[#131A2A]/90 border border-[#263245] px-4 py-3 sm:px-5 sm:py-3.5 flex items-center justify-between gap-3 text-xs shadow-md backdrop-blur-md"
          role="note"
        >
          <div className="flex items-center gap-2.5">
            <Shield className="w-4 h-4 text-[#3B82F6] shrink-0" />
            <p className="text-[#94A3B8] text-[11px] sm:text-xs leading-relaxed">
              <strong className="text-[#F8FAFC] font-semibold">Academic Research & Demonstration System:</strong>{" "}
              NeuroScan AI is designed for computational research and educational demonstrations. It is not an FDA-approved medical diagnostic device.
            </p>
          </div>
          <span className="hidden md:inline-flex text-[10px] font-mono font-semibold px-2.5 py-1 rounded bg-[#1A2336] text-[#22D3EE] border border-[#263245] shrink-0">
            RESEARCH ONLY
          </span>
        </div>

        {/* 2-Column Clinical Workstation Layout */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-6 sm:gap-8 items-start">
          {/* Left Column: Hero Overview + Architecture Strip + Metric Cards */}
          <div className="lg:col-span-5 flex flex-col items-start text-left gap-5">
            {/* Model Badge */}
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#131A2A] border border-[#263245] text-xs font-mono text-[#22D3EE] shadow-sm">
              <Cpu className="w-3.5 h-3.5 text-[#22D3EE]" />
              <span className="tracking-wide uppercase text-[11px]">
                DEEP LEARNING MRI CLASSIFICATION
              </span>
            </div>

            {/* Main Title & Description */}
            <div>
              <h1 className="text-3xl sm:text-4xl lg:text-5xl font-black text-[#F8FAFC] tracking-tight leading-tight mb-2">
                NEUROSCAN <span className="text-[#3B82F6]">AI</span>
              </h1>
              <p className="text-xs sm:text-sm text-[#94A3B8] leading-relaxed">
                Automated multi-class brain tumor classification from MRI scans using ResNet-50 transfer learning with fine-tuned deep feature extraction.
              </p>
            </div>

            {/* Model Architecture Strip (Replaces redundant HUD, Requirement 2 & 11) */}
            <div className="w-full bg-[#131A2A] border border-[#263245] rounded-2xl p-3.5 shadow-lg font-mono text-xs">
              <div className="text-[10px] uppercase tracking-wider text-[#94A3B8] font-bold mb-2 pb-1.5 border-b border-[#263245] flex items-center justify-between">
                <span>MODEL SPECIFICATIONS</span>
                <span className="text-[#22D3EE]">TORCHVISION RESNET-50</span>
              </div>
              <div className="grid grid-cols-3 gap-2 text-center text-[11px]">
                <div className="bg-[#1A2336]/80 p-2 rounded-xl border border-[#263245]">
                  <span className="text-[9px] text-[#94A3B8] block">BACKBONE</span>
                  <span className="font-bold text-[#F8FAFC]">ResNet-50</span>
                </div>
                <div className="bg-[#1A2336]/80 p-2 rounded-xl border border-[#263245]">
                  <span className="text-[9px] text-[#94A3B8] block">INPUT RES</span>
                  <span className="font-bold text-[#22D3EE]">224 × 224</span>
                </div>
                <div className="bg-[#1A2336]/80 p-2 rounded-xl border border-[#263245]">
                  <span className="text-[9px] text-[#94A3B8] block">FINE-TUNED</span>
                  <span className="font-bold text-[#22C55E]">Layer 4 + FC</span>
                </div>
              </div>
            </div>

            {/* Non-Redundant Statistics Grid (2x2) */}
            <div className="w-full grid grid-cols-2 gap-3 sm:gap-3.5">
              {stats.map((stat, idx) => (
                <div
                  key={idx}
                  className={`bg-[#131A2A] border rounded-2xl p-4 transition-all duration-300 shadow-md ${
                    stat.highlight
                      ? "border-[#22D3EE]/50 bg-gradient-to-br from-[#131A2A] to-[#162338]"
                      : "border-[#263245] hover:border-[#3B82F6]/40"
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="p-1.5 rounded-lg bg-[#1A2336] border border-[#263245]">
                      {stat.icon}
                    </div>
                    <span className="text-[9px] font-mono font-bold text-[#94A3B8] bg-[#1A2336] border border-[#263245] px-2 py-0.5 rounded">
                      {stat.badge}
                    </span>
                  </div>
                  <div className="text-xl sm:text-2xl font-black text-[#F8FAFC] tracking-tight mb-0.5">
                    {stat.value}
                  </div>
                  <div className="text-[11px] font-semibold text-[#F8FAFC] mb-0.5 truncate">
                    {stat.label}
                  </div>
                  <p className="text-[10px] text-[#94A3B8] leading-tight">
                    {stat.description}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Right Column: Ingestion Dock & Diagnostic Results */}
          <div className="lg:col-span-7 flex flex-col items-center">
            <div className="w-full bg-[#131A2A] border border-[#263245] rounded-3xl p-5 sm:p-7 shadow-2xl transition-all relative">
              {/* Header Strip for Analysis Panel */}
              <div className="flex items-center justify-between border-b border-[#263245] pb-3.5 mb-5">
                <div className="flex items-center gap-2.5">
                  <div className="p-2 rounded-xl bg-[#1A2336] border border-[#263245] text-[#3B82F6]">
                    <Radio className="w-4 h-4 text-[#3B82F6]" />
                  </div>
                  <div>
                    <h2 className="text-xs sm:text-sm font-mono font-bold uppercase tracking-wider text-[#F8FAFC]">
                      MRI SCAN INGESTION &amp; ANALYSIS
                    </h2>
                    <p className="text-[11px] text-[#94A3B8]">
                      Select an axial brain MRI scan to evaluate tissue characteristics
                    </p>
                  </div>
                </div>
              </div>

              {/* Upload Zone Component */}
              <UploadZone
                onFileSelected={handleFileSelected}
                onAnalyze={handleAnalyze}
                onClear={handleClear}
                selectedFile={selectedFile}
                imagePreview={imagePreview}
                isLoading={isLoading}
                error={error}
              />

              {/* Dynamic Prediction Result Card */}
              {prediction && (
                <div className="mt-5 pt-5 border-t border-[#263245]">
                  <PredictionResult result={prediction} />
                </div>
              )}
            </div>
          </div>
        </section>
      </main>

      {/* Refined Medical Footer (Requirement 1 & 12) */}
      <footer className="w-full border-t border-[#263245] bg-[#0A0F1C] py-5 text-xs text-[#94A3B8] relative z-10 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center justify-between gap-3 sm:gap-4">
          <div className="flex items-center gap-2 font-mono text-[11px]">
            <Brain className="w-4 h-4 text-[#3B82F6]" />
            <span className="font-bold text-[#F8FAFC]">NEUROSCAN AI</span>
            <span className="text-[#263245]">•</span>
            <span>ResNet-50</span>
            <span className="text-[#263245]">•</span>
            <span className="text-[#22D3EE]">95.06% Test Set Accuracy</span>
          </div>

          <div className="flex items-center gap-1.5 font-mono text-[10px]">
            <span className="bg-[#131A2A] border border-[#263245] px-2 py-0.5 rounded text-rose-300">Glioma</span>
            <span className="bg-[#131A2A] border border-[#263245] px-2 py-0.5 rounded text-amber-300">Meningioma</span>
            <span className="bg-[#131A2A] border border-[#263245] px-2 py-0.5 rounded text-purple-300">Pituitary</span>
            <span className="bg-[#131A2A] border border-[#263245] px-2 py-0.5 rounded text-[#22C55E]">No Tumor</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
