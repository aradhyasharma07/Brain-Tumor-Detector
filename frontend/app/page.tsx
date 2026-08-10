"use client";

import React, { useState } from "react";
import Header from "@/components/Header";
import UploadZone from "@/components/UploadZone";
import PredictionResult from "@/components/PredictionResult";
import { predictMriImage } from "@/lib/api";
import { PredictionResponse } from "@/types/api";

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

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-white">
      <Header />

      <main className="flex-1 max-w-5xl w-full mx-auto px-4 sm:px-6 py-8 sm:py-12 flex flex-col items-center gap-8">
        {/* Title Section */}
        <div className="text-center max-w-2xl mx-auto">
          <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-100 tracking-tight mb-3">
            Brain Tumor MRI Classification
          </h2>
          <p className="text-sm text-slate-400 leading-relaxed">
            Upload an MRI brain scan image to detect and classify tumor types using our trained{" "}
            <span className="text-cyan-400 font-semibold">ResNet-50 Transfer Learning</span> model.
          </p>
        </div>

        {/* Upload & Preview Card */}
        <div className="w-full max-w-2xl">
          <UploadZone
            onFileSelected={handleFileSelected}
            onAnalyze={handleAnalyze}
            onClear={handleClear}
            selectedFile={selectedFile}
            imagePreview={imagePreview}
            isLoading={isLoading}
            error={error}
          />
        </div>

        {/* Prediction Results Display */}
        {prediction && (
          <div className="w-full max-w-2xl">
            <PredictionResult result={prediction} />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="w-full border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-500">
        <div className="max-w-5xl mx-auto px-4">
          <p>© 2026 NeuroScan AI | ResNet-50 Brain Tumor Classifier</p>
        </div>
      </footer>
    </div>
  );
}
