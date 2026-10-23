export type TumorClass = "glioma" | "meningioma" | "pituitary" | "notumor";

export interface ClassScore {
  class_name: TumorClass;
  display_name: string;
  score: number;
  percentage: number;
  is_highest: boolean;
}

export interface PredictionResponse {
  status: string;
  predicted_class: TumorClass;
  confidence: number;
  class_index: number;
  probabilities: Record<TumorClass | string, number>;
  class_scores?: ClassScore[];
  gradcam_image?: string | null;
  prediction_id?: string;
  timestamp?: string;
}

export interface HealthResponse {
  status: string;
  service: string;
}
