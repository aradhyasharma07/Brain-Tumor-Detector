import { PredictionResponse, HealthResponse } from "@/types/api";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: "GET",
      cache: "no-store",
      headers: {
        "Accept": "application/json",
      },
    });
    if (!response.ok) return false;
    const data: HealthResponse = await response.json();
    return data.status === "healthy";
  } catch (error) {
    console.error("Backend health check error:", error);
    return false;
  }
}

export async function predictMriImage(file: File): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let errorMessage = `Server error (${response.status})`;
    try {
      const errorData = await response.json();
      if (errorData.detail) {
        errorMessage = errorData.detail;
      }
    } catch {
      // Fallback to HTTP status
    }
    throw new Error(errorMessage);
  }

  return response.json();
}

export async function downloadPdfReport(file: File): Promise<Blob> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/v1/report`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let errorMessage = `Failed to generate PDF report (${response.status})`;
    try {
      const errorData = await response.json();
      if (errorData.detail) {
        errorMessage = errorData.detail;
      }
    } catch {
      // Fallback
    }
    throw new Error(errorMessage);
  }

  return response.blob();
}
