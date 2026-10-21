import os
import sys
import io
import asyncio
import argparse
from fastapi import UploadFile, HTTPException

# Ensure backend directory is discoverable
sys.path.insert(0, os.path.dirname(__file__))

from app.main import app, health, home
from app.routers.prediction_router import predict_mri, get_inference_service


def find_sample_image(explicit_path=None):
    if explicit_path:
        if os.path.isfile(explicit_path):
            return os.path.abspath(explicit_path)
        return None

    backend_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(backend_dir, "..", "..", "Brain Tumor Dataset"),
        r"D:\SEM 5 all projects\Brain Tumor Dataset",
        os.path.join(backend_dir, "..", "Brain Tumor Dataset"),
    ]

    for candidate_dir in candidates:
        if not os.path.isdir(candidate_dir):
            continue
        preferred = [
            os.path.join(candidate_dir, "Training", "glioma", "Tr-gl_1.jpg"),
            os.path.join(candidate_dir, "Testing", "glioma", "Te-gl_1.jpg"),
            os.path.join(candidate_dir, "Testing", "notumor", "Te-no_10.jpg"),
        ]
        for p in preferred:
            if os.path.isfile(p):
                return os.path.abspath(p)
        for root, _, files in os.walk(candidate_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    return os.path.abspath(os.path.join(root, f))
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Brain Tumor FastAPI Endpoint - Automated Verification"
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Path to an MRI test image for endpoint testing."
    )
    return parser.parse_args()


async def run_api_verification():
    args = parse_args()

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
    sample_image_path = find_sample_image(args.image_path)
    if not sample_image_path:
        print("\n" + "!" * 65)
        print("ERROR: Sample MRI image not found for API verification.")
        print("Please provide a valid image file using the --image-path argument:")
        print("  python verify_api.py --image-path <path_to_image>")
        print("!" * 65)
        sys.exit(1)

    print(f"Test 2 - Verifying POST /api/v1/predict with sample MRI image:")
    print(f"  -> Path: {sample_image_path}")

    filename = os.path.basename(sample_image_path)
    with open(sample_image_path, "rb") as f:
        image_bytes = f.read()

    upload_file = UploadFile(
        filename=filename,
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
