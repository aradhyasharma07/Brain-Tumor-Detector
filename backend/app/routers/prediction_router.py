import os
import uuid
from typing import Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, status, Depends, Response, Query
from services.inference_service import InferenceService
from services.validation_service import MriValidationService
from services.pdf_service import PdfReportService
from schemas.prediction_schema import PredictionResponse

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
    include_gradcam: bool = Query(default=True, description="Whether to include Grad-CAM overlay heatmap"),
    service: InferenceService = Depends(get_inference_service)
):
    """
    Accepts a single MRI image file upload, validates that it is a genuine,
    supported brain MRI scan using multi-layer validation, runs ResNet50 inference,
    and returns predicted class, complete probability scores, and Grad-CAM heatmap.
    
    Arbitrary non-MRI images (e.g. photos, screenshots, noise, blank images, corrupt files)
    are strictly rejected with HTTP 400 Bad Request before reaching the classifier.
    """
    filename = file.filename or ""

    # 1. Read image file bytes safely
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}"
        )

    # 2. Multi-Stage MRI Validation (Security, Format, Domain & Morphological Checks)
    is_valid, validation_error, pil_image = MriValidationService.validate_upload(
        image_bytes=contents,
        filename=filename
    )
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=validation_error
        )

    # 3. Run Inference Pipeline with Deep-Feature OOD Screening and Grad-CAM
    try:
        result = service.predict_pil(pil_image, validate=True, include_gradcam=include_gradcam)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during model inference: {str(e)}"
        )

    prediction_id = f"PRED-{uuid.uuid4().hex[:8].upper()}"

    return PredictionResponse(
        status="success",
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        class_index=result["class_index"],
        probabilities=result["probabilities"],
        class_scores=result.get("class_scores"),
        gradcam_image=result.get("gradcam_image"),
        prediction_id=prediction_id
    )


@router.post("/gradcam")
async def generate_gradcam_overlay(
    file: UploadFile = File(...),
    service: InferenceService = Depends(get_inference_service)
):
    """
    Dedicated endpoint returning the Grad-CAM visualization for a validated MRI upload.
    """
    filename = file.filename or ""

    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}"
        )

    is_valid, validation_error, pil_image = MriValidationService.validate_upload(
        image_bytes=contents,
        filename=filename
    )
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=validation_error
        )

    try:
        result = service.predict_pil(pil_image, validate=True, include_gradcam=True)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating Grad-CAM: {str(e)}"
        )

    if not result.get("gradcam_image"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Grad-CAM visualization could not be generated for this scan."
        )

    return {
        "status": "success",
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "gradcam_image": result["gradcam_image"]
    }


@router.post("/report")
@router.post("/generate-report")
async def generate_pdf_report(
    file: UploadFile = File(...),
    service: InferenceService = Depends(get_inference_service)
):
    """
    Dedicated endpoint that accepts an authentic brain MRI scan upload,
    executes the authoritative server-side validation and inference pipeline,
    and returns a downloadable PDF medical research report.
    """
    filename = file.filename or ""

    # 1. Read bytes safely
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}"
        )

    # 2. Multi-Stage Validation (defense-in-depth)
    is_valid, validation_error, pil_image = MriValidationService.validate_upload(
        image_bytes=contents,
        filename=filename
    )
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=validation_error
        )

    # 3. Authoritative inference execution on server
    try:
        result = service.predict_pil(pil_image, validate=True, include_gradcam=True)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed during report generation: {str(e)}"
        )

    # 4. Generate PDF Report via ReportLab
    report_id = f"REP-{uuid.uuid4().hex[:8].upper()}"
    try:
        pdf_bytes = PdfReportService.generate_report(
            prediction_result=result,
            original_image=pil_image,
            overlay_image=result.get("gradcam_overlay_pil"),
            report_id=report_id
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to render PDF report: {str(e)}"
        )

    download_filename = f"neuroscan_report_{report_id}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{download_filename}"',
            "X-Report-Id": report_id,
        }
    )
