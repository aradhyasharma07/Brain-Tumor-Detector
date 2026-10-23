from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ClassScoreItem(BaseModel):
    class_name: str = Field(..., description="Canonical class name identifier")
    display_name: str = Field(..., description="Human-readable class label")
    score: float = Field(..., description="Probability score between 0.0 and 1.0")
    percentage: float = Field(..., description="Percentage representation 0-100%")
    is_highest: bool = Field(..., description="True if this is the top predicted class")


class PredictionResponse(BaseModel):
    status: str = Field(default="success", description="API response status")
    predicted_class: str = Field(..., description="Target class with highest predicted probability")
    confidence: float = Field(..., description="Probability of the predicted class (0.0 to 1.0)")
    class_index: int = Field(..., description="Integer index of the predicted class")
    probabilities: Dict[str, float] = Field(..., description="Predicted probability distribution for all 4 classes")
    class_scores: Optional[List[ClassScoreItem]] = Field(default=None, description="Ordered canonical class scores list")
    gradcam_image: Optional[str] = Field(default=None, description="Grad-CAM overlay JPEG data URI")
    prediction_id: Optional[str] = Field(default=None, description="Unique prediction identifier")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "predicted_class": "glioma",
                "confidence": 0.9214,
                "class_index": 0,
                "probabilities": {
                    "glioma": 0.9214,
                    "meningioma": 0.0574,
                    "pituitary": 0.0117,
                    "notumor": 0.0095
                },
                "class_scores": [
                    {"class_name": "glioma", "display_name": "Glioma", "score": 0.9214, "percentage": 92.14, "is_highest": True},
                    {"class_name": "meningioma", "display_name": "Meningioma", "score": 0.0574, "percentage": 5.74, "is_highest": False},
                    {"class_name": "pituitary", "display_name": "Pituitary Tumor", "score": 0.0117, "percentage": 1.17, "is_highest": False},
                    {"class_name": "notumor", "display_name": "No Tumor Detected", "score": 0.0095, "percentage": 0.95, "is_highest": False}
                ],
                "gradcam_image": "data:image/jpeg;base64,...",
                "prediction_id": "PRED-8A7B2C"
            }
        }
