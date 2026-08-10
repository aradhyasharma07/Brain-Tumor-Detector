export type TumorClass = "glioma" | "meningioma" | "pituitary" | "notumor";

export interface PredictionResponse {
  status: string;
  predicted_class: TumorClass;
  confidence: number;
  class_index: number;
  probabilities: Record<TumorClass | string, number>;
}

export interface HealthResponse {
  status: string;
  service: string;
}
