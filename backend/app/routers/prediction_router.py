import os
from fastapi import APIRouter, File, UploadFile, HTTPException, status, Depends
from backend.services.inference_service import InferenceService
from backend.schemas.prediction_schema import PredictionResponse

router = APIRouter(prefix="/api/v1", tags=["Inference"])

_inference_service: InferenceService = None


def get_inference_service() -> InferenceService:
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service


@router.post("/predict", response_model=PredictionResponse)
async def predict_mri(
    file: UploadFile = File(...),
    service: InferenceService = Depends(get_inference_service)
):
    """
    Accepts a single MRI image file upload, validates the image format,
    runs ResNet50 inference, and returns predicted class and probabilities.
    """
    # 1. Validate filename extension if present
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    if ext and ext not in valid_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension '{ext}'. Allowed extensions: {', '.join(sorted(valid_extensions))}"
        )

    # 2. Read image file bytes
    try:
        contents = await file.read()
        if not contents or len(contents) == 0:
            raise ValueError("Uploaded file is empty.")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}"
        )

    # 3. Run Inference Pipeline
    try:
        result = service.predict_bytes(contents)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file could not be parsed as a valid image: {str(e)}"
        )

    return PredictionResponse(
        status="success",
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        class_index=result["class_index"],
        probabilities=result["probabilities"]
    )
