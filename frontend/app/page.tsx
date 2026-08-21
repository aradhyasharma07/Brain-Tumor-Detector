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
  Terminal,
} from "lucide-react";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);

  const handleFileSelected = (file: File) => {
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
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const result = await predictMriImage(selectedFile);
      setPrediction(result);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "An unexpected error occurred.";
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  };

  const stats = [
    {
      value: "95.06%",
      label: "Test Accuracy",
      description: "Evaluated on 1,600 test MRI scans",
      badge: "VERIFIED",
      icon: <CheckCircle2 className="w-4 h-4 text-[#22D3EE]" />,
    },
    {
      value: "97.77%",
      label: "Validation Accuracy",
      description: "Peak model epoch performance",
      badge: "OPTIMAL",
      icon: <Zap className="w-4 h-4 text-[#3B82F6]" />,
    },
    {
      value: "1,600",
      label: "Test MRI Images",
      description: "Balanced diagnostic dataset",
      badge: "CLINICAL",
      icon: <FileCheck2 className="w-4 h-4 text-[#22D3EE]" />,
    },
    {
      value: "4",
      label: "Tumor Classes",
      description: "Glioma, Meningioma, Pituitary & No Tumor",
      badge: "MULTI-CLASS",
      icon: <Layers className="w-4 h-4 text-[#22C55E]" />,
    },
  ];

  return (
    <div className="min-h-screen bg-[#0A0F1C] text-[#F8FAFC] flex flex-col font-sans selection:bg-[#22D3EE] selection:text-[#0A0F1C] relative overflow-hidden">
      {/* Brushed Titanium Background Grid (2-3% opacity) */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#26324515_1px,transparent_1px),linear-gradient(to_bottom,#26324515_1px,transparent_1px)] bg-[size:24px_24px] pointer-events-none z-0" />

      <Header />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10 flex flex-col gap-10 relative z-10">
        {/* Desktop Above-The-Fold 2-Column Workstation Layout */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Hero Header + Live Diagnostic HUD + Stats Cards */}
          <div className="lg:col-span-6 flex flex-col items-start text-left gap-6">
            {/* Badge */}
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-[#131A2A] border border-[#263245] text-xs font-mono text-[#22D3EE] shadow-md">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#22C55E] opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-[#22C55E]"></span>
              </span>
              <span className="tracking-wider uppercase">ADVANCED MRI DIAGNOSTIC INTELLIGENCE</span>
            </div>

            {/* Main Title */}
            <div>
              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black text-[#F8FAFC] tracking-tight leading-none mb-3">
                NEUROSCAN <span className="text-[#3B82F6]">AI</span>
              </h1>
              <p className="text-sm sm:text-base text-[#94A3B8] leading-relaxed max-w-xl">
                AI-powered Brain Tumor MRI Classification using ResNet-50 Transfer Learning for fast and reliable diagnostic assistance.
              </p>
            </div>

            {/* Live Diagnostic HUD Panel */}
            <div className="w-full bg-[#131A2A] border border-[#263245] rounded-2xl p-4 shadow-xl relative overflow-hidden">
              <div className="flex items-center justify-between border-b border-[#263245] pb-3 mb-3">
                <div className="flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-[#3B82F6]" />
                  <span className="text-xs font-mono font-bold tracking-widest text-[#F8FAFC] uppercase">
                    LIVE DIAGNOSTIC HUD
                  </span>
                </div>
                <div className="flex items-center gap-1.5 text-[11px] font-mono text-[#22C55E] bg-[#22C55E]/10 border border-[#22C55E]/20 px-2 py-0.5 rounded">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#22C55E] animate-pulse" />
                  <span>AI STATUS: READY</span>
                </div>
              </div>

              <div className="grid grid-cols-4 gap-2 text-center font-mono">
                <div className="bg-[#1A2336] border border-[#263245] p-2 rounded-xl">
                  <span className="text-[10px] text-[#94A3B8] block">MODEL</span>
                  <span className="text-xs font-bold text-[#F8FAFC]">ResNet-50</span>
                </div>
                <div className="bg-[#1A2336] border border-[#263245] p-2 rounded-xl">
                  <span className="text-[10px] text-[#94A3B8] block">TEST ACC</span>
                  <span className="text-xs font-bold text-[#22D3EE]">95.06%</span>
                </div>
                <div className="bg-[#1A2336] border border-[#263245] p-2 rounded-xl">
                  <span className="text-[10px] text-[#94A3B8] block">VALIDATION</span>
                  <span className="text-xs font-bold text-[#3B82F6]">97.77%</span>
                </div>
                <div className="bg-[#1A2336] border border-[#263245] p-2 rounded-xl">
                  <span className="text-[10px] text-[#94A3B8] block">CLASSES</span>
                  <span className="text-xs font-bold text-[#22C55E]">4 Types</span>
                </div>
              </div>
            </div>

            {/* Metallic Statistics Cards Grid (2x2) */}
            <div className="w-full grid grid-cols-2 gap-3.5">
              {stats.map((stat, idx) => (
                <div
                  key={idx}
                  className="bg-[#131A2A] border border-[#263245] hover:border-[#3B82F6]/50 rounded-2xl p-4 transition-all duration-300 hover:-translate-y-0.5 shadow-lg group"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="p-2 rounded-xl bg-[#1A2336] border border-[#263245]">
                      {stat.icon}
                    </div>
                    <span className="text-[9px] font-mono font-bold text-[#94A3B8] bg-[#1A2336] border border-[#263245] px-2 py-0.5 rounded">
                      {stat.badge}
                    </span>
                  </div>
                  <div className="text-2xl font-extrabold text-[#F8FAFC] tracking-tight mb-0.5">
                    {stat.value}
                  </div>
                  <div className="text-xs font-semibold text-[#F8FAFC] mb-1">
                    {stat.label}
                  </div>
                  <p className="text-[11px] text-[#94A3B8]">
                    {stat.description}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Right Column: Upload Dock Station */}
          <div className="lg:col-span-6 flex flex-col items-center">
            <div className="w-full bg-[#131A2A] border border-[#263245] hover:border-[#3B82F6]/40 rounded-3xl p-6 sm:p-7 shadow-2xl shadow-black/80 transition-all duration-300 relative">
              <div className="flex items-center justify-between border-b border-[#263245] pb-4 mb-5">
                <div className="flex items-center gap-2.5">
                  <div className="p-2 rounded-xl bg-[#1A2336] border border-[#263245] text-[#3B82F6]">
                    <Radio className="w-4 h-4 text-[#3B82F6]" />
                  </div>
                  <div>
                    <h2 className="text-sm font-mono font-bold uppercase tracking-wider text-[#F8FAFC]">
                      MRI UPLOAD DOCK STATION
                    </h2>
                    <p className="text-xs text-[#94A3B8]">
                      Drag or select an MRI scan file for classification
                    </p>
                  </div>
                </div>
                <span className="text-[10px] font-mono text-[#22D3EE] bg-[#1A2336] border border-[#263245] px-2.5 py-1 rounded">
                  DOCK ACTIVE
                </span>
              </div>

              <UploadZone
                onFileSelected={handleFileSelected}
                onAnalyze={handleAnalyze}
                onClear={handleClear}
                selectedFile={selectedFile}
                imagePreview={imagePreview}
                isLoading={isLoading}
                error={error}
              />

              {prediction && (
                <div className="mt-5 pt-5 border-t border-[#263245]">
                  <PredictionResult result={prediction} />
                </div>
              )}
            </div>
          </div>
        </section>
      </main>

      {/* Medical Radiology Software Footer */}
      <footer className="w-full border-t border-[#263245] bg-[#0A0F1C] py-6 text-xs text-[#94A3B8] relative z-10 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2 font-mono">
            <Brain className="w-4 h-4 text-[#3B82F6]" />
            <span className="font-bold text-[#F8FAFC]">NEUROSCAN AI</span>
            <span className="text-[#263245]">•</span>
            <span>ResNet-50</span>
            <span className="text-[#263245]">•</span>
            <span className="text-[#22D3EE]">95.06% Verified Test Accuracy</span>
          </div>

          <div className="flex items-center gap-2 font-mono text-[11px]">
            <span className="bg-[#131A2A] border border-[#263245] px-2.5 py-1 rounded">Glioma</span>
            <span className="bg-[#131A2A] border border-[#263245] px-2.5 py-1 rounded">Meningioma</span>
            <span className="bg-[#131A2A] border border-[#263245] px-2.5 py-1 rounded">Pituitary</span>
            <span className="bg-[#131A2A] border border-[#263245] px-2.5 py-1 rounded">No Tumor</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
