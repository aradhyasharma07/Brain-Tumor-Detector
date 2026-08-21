"use client";

import React, { useState, useRef } from "react";
import {
  UploadCloud,
  FileImage,
  RotateCcw,
  Zap,
  AlertCircle,
  Loader2,
} from "lucide-react";

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
          className={`relative cursor-pointer w-full rounded-2xl border-2 border-dashed transition-all duration-300 p-8 sm:p-10 flex flex-col items-center justify-center text-center overflow-hidden group ${
            isDragOver
              ? "border-[#22D3EE] bg-[#22D3EE]/5 shadow-xl shadow-[#22D3EE]/10 scale-[1.01]"
              : "border-[#263245] bg-[#1A2336]/60 hover:border-[#3B82F6]/60 hover:bg-[#1A2336]/90"
          }`}
        >
          {/* Smooth Moving Scan Line Overlay */}
          <div
            className="absolute inset-x-0 h-0.5 bg-gradient-to-r from-transparent via-[#22D3EE] to-transparent pointer-events-none opacity-70 shadow-[0_0_8px_#22D3EE]"
            style={{
              animation: "scanLine 3.5s linear infinite",
            }}
          />

          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".jpg,.jpeg,.png,.bmp,.webp"
            className="hidden"
          />

          <div className="h-14 w-14 rounded-2xl bg-[#131A2A] border border-[#263245] flex items-center justify-center mb-4 text-[#22D3EE] group-hover:border-[#3B82F6]/50 shadow-inner transition-colors">
            <UploadCloud className="w-7 h-7 text-[#22D3EE]" />
          </div>

          <h3 className="text-sm sm:text-base font-mono font-bold text-[#F8FAFC] tracking-wide mb-1">
            Drag & Drop MRI Brain Scan Here
          </h3>
          <p className="text-xs text-[#94A3B8] mb-4 max-w-xs">
            Supported Formats: JPG, JPEG, PNG, BMP, WEBP
          </p>

          <button
            type="button"
            className="px-4 py-2 rounded-xl bg-[#131A2A] hover:bg-[#1A2336] text-[#F8FAFC] text-xs font-mono border border-[#263245] hover:border-[#3B82F6]/40 transition shadow-sm"
          >
            BROWSE MRI FILE
          </button>
        </div>
      ) : (
        <div className="w-full bg-[#1A2336]/80 rounded-2xl border border-[#263245] p-4 sm:p-5 flex flex-col items-center">
          <div className="relative w-full max-w-sm aspect-square rounded-xl overflow-hidden bg-[#0A0F1C] border border-[#263245] mb-4 shadow-2xl flex items-center justify-center group">
            {/* Smooth Moving Scan Line for Preview Box */}
            <div
              className="absolute inset-x-0 h-0.5 bg-gradient-to-r from-transparent via-[#22D3EE] to-transparent pointer-events-none opacity-60 z-10 shadow-[0_0_8px_#22D3EE]"
              style={{
                animation: "scanLine 4s linear infinite",
              }}
            />
            {/* eslint-disable-next-next/no-img-element */}
            <img
              src={imagePreview}
              alt="Uploaded MRI Preview"
              className="w-full h-full object-contain"
            />
          </div>

          {selectedFile && (
            <div className="w-full flex items-center justify-between bg-[#0A0F1C]/80 px-4 py-2.5 rounded-xl border border-[#263245] mb-4 font-mono">
              <div className="truncate mr-2 flex items-center gap-2">
                <FileImage className="w-4 h-4 text-[#22D3EE] shrink-0" />
                <div className="truncate">
                  <p className="text-xs font-bold text-[#F8FAFC] truncate">
                    {selectedFile.name}
                  </p>
                  <p className="text-[10px] text-[#94A3B8]">
                    {formatFileSize(selectedFile.size)}
                  </p>
                </div>
              </div>

              <button
                onClick={onClear}
                disabled={isLoading}
                className="text-xs text-rose-400 hover:text-rose-300 font-mono px-2.5 py-1 rounded-lg hover:bg-rose-500/10 transition disabled:opacity-50 flex items-center gap-1 shrink-0"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>CHANGE</span>
              </button>
            </div>
          )}

          <button
            onClick={onAnalyze}
            disabled={isLoading}
            className="w-full py-3.5 px-6 rounded-xl font-mono font-bold text-xs sm:text-sm bg-gradient-to-r from-[#3B82F6] to-[#2563EB] hover:from-[#2563EB] hover:to-[#1D4ED8] text-white shadow-lg shadow-[#3B82F6]/20 border border-[#3B82F6]/50 transition-all transform active:scale-[0.99] flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed tracking-widest uppercase"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 text-white animate-spin" />
                <span>ANALYZING MRI SCAN...</span>
              </>
            ) : (
              <>
                <Zap className="w-4 h-4 text-white" />
                <span>ANALYZE SCAN WITH RESNET-50</span>
              </>
            )}
          </button>
        </div>
      )}

      {/* Error Banners */}
      {(localError || error) && (
        <div className="w-full p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-start gap-3">
          <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="font-bold font-mono uppercase mb-0.5">SCAN ERROR</p>
            <p className="text-[11px] text-rose-200">{localError || error}</p>
          </div>
        </div>
      )}

      {/* Embedded CSS for smooth scanLine keyframes */}
      <style jsx>{`
        @keyframes scanLine {
          0% {
            top: 0%;
            opacity: 0;
          }
          10% {
            opacity: 0.8;
          }
          90% {
            opacity: 0.8;
          }
          100% {
            top: 100%;
            opacity: 0;
          }
        }
      `}</style>
    </div>
  );
}
