"use client";

import React, { useState, useRef } from "react";

interface UploadZoneProps {
  onFileSelected: (file: File) => void;
  onAnalyze: () => void;
  onClear: () => void;
  selectedFile: File | null;
  imagePreview: string | null;
  isLoading: boolean;
  error: string | null;
}

const ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "webp"];

export default function UploadZone({
  onFileSelected,
  onAnalyze,
  onClear,
  selectedFile,
  imagePreview,
  isLoading,
  error,
}: UploadZoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateAndProcessFile = (file: File) => {
    setLocalError(null);
    const ext = file.name.split(".").pop()?.toLowerCase() || "";
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setLocalError(
        `Invalid file type (.${ext}). Please select a valid MRI image (JPG, JPEG, PNG, BMP, or WEBP).`
      );
      return;
    }
    onFileSelected(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndProcessFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndProcessFile(e.target.files[0]);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className="w-full flex flex-col gap-4">
      {/* Upload Box / Image Preview Zone */}
      {!imagePreview ? (
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => fileInputRef.current?.click()}
          className={`relative cursor-pointer w-full rounded-2xl border-2 border-dashed transition-all duration-300 p-8 sm:p-12 flex flex-col items-center justify-center text-center ${
            isDragOver
              ? "border-cyan-400 bg-cyan-500/10 shadow-xl shadow-cyan-500/10 scale-[1.01]"
              : "border-slate-800 bg-slate-900/50 hover:border-slate-700 hover:bg-slate-900/80"
          }`}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".jpg,.jpeg,.png,.bmp,.webp"
            className="hidden"
          />

          <div className="h-16 w-16 rounded-2xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center mb-4 text-cyan-400">
            <svg
              className="w-8 h-8"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
          </div>

          <h3 className="text-base font-semibold text-slate-200 mb-1">
            Drag & Drop MRI Brain Scan Here
          </h3>
          <p className="text-xs text-slate-400 mb-4 max-w-xs">
            Supported formats: JPG, JPEG, PNG, BMP, WEBP
          </p>

          <button
            type="button"
            className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition"
          >
            Browse File
          </button>
        </div>
      ) : (
        <div className="w-full bg-slate-900/60 rounded-2xl border border-slate-800 p-4 sm:p-6 flex flex-col items-center">
          <div className="relative w-full max-w-sm aspect-square rounded-xl overflow-hidden bg-slate-950 border border-slate-800 mb-4 shadow-2xl flex items-center justify-center">
            {/* eslint-disable-next-next/no-img-element */}
            <img
              src={imagePreview}
              alt="Uploaded MRI Preview"
              className="w-full h-full object-contain"
            />
          </div>

          {selectedFile && (
            <div className="w-full flex items-center justify-between bg-slate-950/50 px-4 py-2.5 rounded-xl border border-slate-800/80 mb-4">
              <div className="truncate mr-2">
                <p className="text-xs font-medium text-slate-200 truncate">
                  {selectedFile.name}
                </p>
                <p className="text-[10px] text-slate-400">
                  {formatFileSize(selectedFile.size)}
                </p>
              </div>

              <button
                onClick={onClear}
                disabled={isLoading}
                className="text-xs text-rose-400 hover:text-rose-300 font-medium px-2 py-1 rounded-lg hover:bg-rose-500/10 transition disabled:opacity-50"
              >
                Change Image
              </button>
            </div>
          )}

          <button
            onClick={onAnalyze}
            disabled={isLoading}
            className="w-full py-3 px-6 rounded-xl font-semibold text-sm bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white shadow-lg shadow-cyan-500/20 transition-all transform active:scale-[0.99] flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <>
                <svg
                  className="animate-spin h-5 w-5 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                <span>Analyzing MRI Scan...</span>
              </>
            ) : (
              <>
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M13 10V3L4 14h7v7l9-11h-7z"
                  />
                </svg>
                <span>Analyze Scan with ResNet-50</span>
              </>
            )}
          </button>
        </div>
      )}

      {/* Error Banners */}
      {(localError || error) && (
        <div className="w-full p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-start gap-3 animate-fade-in">
          <svg
            className="w-5 h-5 text-rose-400 shrink-0 mt-0.5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <div className="flex-1">
            <p className="font-semibold mb-0.5">Error</p>
            <p>{localError || error}</p>
          </div>
        </div>
      )}
    </div>
  );
}
