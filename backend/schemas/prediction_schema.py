from typing import Dict
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    status: str = Field(default="success", description="API response status")
    predicted_class: str = Field(..., description="Target class with highest predicted probability")
    confidence: float = Field(..., description="Probability of the predicted class (0.0 to 1.0)")
    class_index: int = Field(..., description="Integer index of the predicted class")
    probabilities: Dict[str, float] = Field(..., description="Predicted probability distribution for all 4 classes")

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
                }
            }
        }
