import os
import sys
import io
import asyncio
from fastapi import UploadFile, HTTPException

# Ensure backend package is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.main import app, health, home
from backend.app.routers.prediction_router import predict_mri, get_inference_service


async def run_api_verification():
    print("=" * 65)
    print("      Brain Tumor FastAPI Endpoint - Automated Verification")
    print("=" * 65)

    # 1. Verify Health Endpoint
    print("Test 1 - Verifying GET /health...")
    health_resp = health()
    print(f"  -> Health Response: {health_resp}")
    assert health_resp["status"] == "healthy"
    print("  -> GET /health PASSED!\n")

    # 2. Verify Prediction Endpoint with Valid Image Upload
    sample_image_path = r"C:\Users\aradhya\OneDrive\Desktop\Brain Tumor Dataset\Training\glioma\Tr-gl_1.jpg"
    print(f"Test 2 - Verifying POST /api/v1/predict with sample MRI image:")
    print(f"  -> Path: {sample_image_path}")

    with open(sample_image_path, "rb") as f:
        image_bytes = f.read()

    upload_file = UploadFile(
        filename="Tr-gl_1.jpg",
        file=io.BytesIO(image_bytes),
        headers={"content-type": "image/jpeg"}
    )

    service = get_inference_service()
    response = await predict_mri(file=upload_file, service=service)

    print("\n--- Response Payload from POST /api/v1/predict ---")
    print(f"  Status          : {response.status}")
    print(f"  Predicted Class : {response.predicted_class}")
    print(f"  Class Index     : {response.class_index}")
    print(f"  Confidence      : {response.confidence:.4f} ({response.confidence * 100:.2f}%)")
    print("  Class Probabilities:")
    for cls_name, prob in response.probabilities.items():
        print(f"    - {cls_name:12s}: {prob:.4f} ({prob * 100:.2f}%)")

    assert response.status == "success"
    assert response.predicted_class in ["glioma", "meningioma", "pituitary", "notumor"]
    assert 0.0 <= response.confidence <= 1.0
    assert len(response.probabilities) == 4
    assert abs(sum(response.probabilities.values()) - 1.0) < 1e-4
    print("  -> POST /api/v1/predict Valid Upload PASSED!\n")

    # 3. Verify Invalid File Extension Handling
    print("Test 3 - Verifying POST /api/v1/predict with invalid file extension...")
    invalid_file = UploadFile(
        filename="invalid.txt",
        file=io.BytesIO(b"Not an image"),
        headers={"content-type": "text/plain"}
    )

    try:
        await predict_mri(file=invalid_file, service=service)
        assert False, "Expected HTTPException for invalid extension, but none was raised."
    except HTTPException as exc:
        print(f"  -> Caught Expected HTTP Exception: Code {exc.status_code} | Detail: {exc.detail}")
        assert exc.status_code == 400
        print("  -> Invalid File Extension Validation PASSED!\n")

    # 4. Verify Corrupted Image Content Handling
    print("Test 4 - Verifying POST /api/v1/predict with corrupted image contents...")
    corrupted_file = UploadFile(
        filename="corrupted.png",
        file=io.BytesIO(b"Corrupted image bytes data"),
        headers={"content-type": "image/png"}
    )

    try:
        await predict_mri(file=corrupted_file, service=service)
        assert False, "Expected HTTPException for corrupted image, but none was raised."
    except HTTPException as exc:
        print(f"  -> Caught Expected HTTP Exception: Code {exc.status_code} | Detail: {exc.detail}")
        assert exc.status_code == 400
        print("  -> Corrupted Image Validation PASSED!\n")

    print("=" * 65)
    print("      FASTAPI API INFERENCE ENDPOINT VERIFICATION PASSED")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(run_api_verification())
