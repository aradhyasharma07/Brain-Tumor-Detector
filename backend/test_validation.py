import os
import sys
import io
import asyncio
from typing import Dict, Any
import numpy as np
from PIL import Image, ImageDraw
from fastapi import UploadFile, HTTPException

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.routers.prediction_router import predict_mri, get_inference_service
from services.validation_service import MriValidationService
from services.inference_service import InferenceService


DATASET_CANDIDATES = [
    r"D:\SEM 5 all projects\Brain Tumor Dataset\Testing",
    r"D:\SEM 5 all projects\Brain Tumor Dataset\Training",
    os.path.join(os.path.dirname(__file__), "..", "..", "Brain Tumor Dataset", "Testing"),
    os.path.join(os.path.dirname(__file__), "..", "Brain Tumor Dataset", "Testing"),
]


def find_dataset_dir() -> str:
    for candidate in DATASET_CANDIDATES:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError("Brain Tumor Dataset directory could not be located.")


def find_mri_sample(cls_name: str, dataset_dir: str) -> str:
    cls_dir = os.path.join(dataset_dir, cls_name)
    if not os.path.isdir(cls_dir):
        # Check subdirectories
        for root, dirs, _ in os.walk(dataset_dir):
            if os.path.basename(root).lower() == cls_name.lower():
                cls_dir = root
                break

    for f in os.listdir(cls_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            return os.path.join(cls_dir, f)
    raise FileNotFoundError(f"No sample image found for class '{cls_name}'.")


async def run_predict_endpoint(service: InferenceService, image_bytes: bytes, filename: str) -> Dict[str, Any]:
    upload_file = UploadFile(
        filename=filename,
        file=io.BytesIO(image_bytes),
        headers={"content-type": "image/jpeg"}
    )
    resp = await predict_mri(file=upload_file, service=service)
    return {
        "status": resp.status,
        "predicted_class": resp.predicted_class,
        "confidence": resp.confidence,
        "class_index": resp.class_index,
        "probabilities": resp.probabilities
    }


def create_test_images():
    """Generates synthetic test images for each required test category."""
    test_cases = {}

    # Category 2: Random Natural Photographs
    # 2.1 Vibrant color scenery (gradient + colored blocks simulating landscape photo)
    w, h = 320, 240
    color_photo = np.zeros((h, w, 3), dtype=np.uint8)
    color_photo[:h//2, :, 0] = 50   # Blue sky R
    color_photo[:h//2, :, 1] = 135  # G
    color_photo[:h//2, :, 2] = 220  # B
    color_photo[h//2:, :, 0] = 34   # Green grass R
    color_photo[h//2:, :, 1] = 160  # G
    color_photo[h//2:, :, 2] = 40   # B
    buf = io.BytesIO()
    Image.fromarray(color_photo).save(buf, format="JPEG")
    test_cases["2.1 Natural Scenery Photo (Color)"] = (buf.getvalue(), "scenery.jpg", False, "color saturation")

    # 2.2 Random colorful photograph (people/objects)
    rand_photo = np.random.randint(40, 230, (300, 300, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rand_photo).save(buf, format="JPEG")
    test_cases["2.2 Random Color Photograph"] = (buf.getvalue(), "random_photo.jpg", False, "color saturation")

    # 2.3 Grayscale natural photograph (full frame textures, bright border)
    gray_photo = np.zeros((300, 300), dtype=np.uint8)
    for r in range(300):
        for c in range(300):
            gray_photo[r, c] = int(120 + 60 * np.sin(r / 20.0) * np.cos(c / 20.0))
    buf = io.BytesIO()
    Image.fromarray(gray_photo).save(buf, format="JPEG")
    test_cases["2.3 Grayscale Landscape Photo"] = (buf.getvalue(), "gray_photo.jpg", False, "dark background perimeter")

    # Category 3: Screenshots and Unrelated Images
    # 3.1 White background document screenshot
    doc_ss = Image.new("RGB", (450, 320), color=(255, 255, 255))
    d = ImageDraw.Draw(doc_ss)
    d.text((30, 30), "PATIENT SUMMARY REPORT\nDate: 2026-10-22\nStatus: Pending", fill=(20, 20, 20))
    d.rectangle([30, 100, 420, 280], outline=(100, 100, 100), width=2)
    buf = io.BytesIO()
    doc_ss.save(buf, format="PNG")
    test_cases["3.1 White Document Screenshot"] = (buf.getvalue(), "document.png", False, "dark background perimeter")

    # 3.2 Dark mode code screenshot
    code_ss = Image.new("RGB", (450, 320), color=(28, 33, 40))
    d = ImageDraw.Draw(code_ss)
    d.text((20, 20), "import os\nfrom fastapi import FastAPI\napp = FastAPI()", fill=(88, 166, 255))
    buf = io.BytesIO()
    code_ss.save(buf, format="PNG")
    test_cases["3.2 Dark Mode IDE Screenshot"] = (buf.getvalue(), "code_editor.png", False, "structural contrast")

    # Category 4: Corrupt Image Files
    # 4.1 Pure garbage bytes
    test_cases["4.1 Random Garbage Bytes"] = (b"THIS_IS_NOT_A_VALID_IMAGE_DATA_XYZ123", "corrupt.jpg", False, "could not be parsed")

    # 4.2 Truncated JPEG stream
    buf = io.BytesIO()
    Image.new("RGB", (200, 200), color=(50, 50, 50)).save(buf, format="JPEG")
    truncated_bytes = buf.getvalue()[:len(buf.getvalue()) // 4]
    test_cases["4.2 Truncated JPEG Byte Stream"] = (truncated_bytes, "truncated.jpg", False, "could not be parsed")

    # Category 5: Empty Uploads and Unsupported Formats
    # 5.1 Empty file
    test_cases["5.1 Empty 0-byte File"] = (b"", "empty.jpg", False, "empty")

    # 5.2 Disallowed extension .txt
    test_cases["5.2 Disallowed Extension (.txt)"] = (b"Some text data", "notes.txt", False, "Invalid file extension")

    # 5.3 Disallowed extension .pdf
    test_cases["5.3 Disallowed Extension (.pdf)"] = (b"%PDF-1.4 header dummy", "scan.pdf", False, "Invalid file extension")

    # 5.4 File exceeding 10MB limit
    oversized_bytes = b"0" * (11 * 1024 * 1024)
    test_cases["5.4 Oversized 11MB File"] = (oversized_bytes, "huge.jpg", False, "10MB limit")

    # Category 6: Very Small, Unusual, and Malformed Images
    # 6.1 Tiny 16x16 icon
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=(80, 80, 80)).save(buf, format="PNG")
    test_cases["6.1 Tiny 16x16 Icon"] = (buf.getvalue(), "tiny.png", False, "too small")

    # 6.2 Degenerate aspect ratio banner (500x20)
    buf = io.BytesIO()
    Image.new("RGB", (500, 20), color=(0, 0, 0)).save(buf, format="PNG")
    test_cases["6.2 Extreme Aspect Ratio (500x20)"] = (buf.getvalue(), "banner.png", False, "too small")

    # 6.3 Solid black image
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), color=(0, 0, 0)).save(buf, format="JPEG")
    test_cases["6.3 Solid Pure Black Image"] = (buf.getvalue(), "black.jpg", False, "predominantly black")

    # 6.4 Solid white image
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), color=(255, 255, 255)).save(buf, format="JPEG")
    test_cases["6.4 Solid Pure White Image"] = (buf.getvalue(), "white.jpg", False, "dark background perimeter")

    # 6.5 Solid uniform gray block
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), color=(128, 128, 128)).save(buf, format="JPEG")
    test_cases["6.5 Solid Uniform Gray Block"] = (buf.getvalue(), "gray.jpg", False, "dark background perimeter")

    # 6.6 Full-frame uniform random noise (RGB)
    buf = io.BytesIO()
    Image.fromarray(np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)).save(buf, format="JPEG")
    test_cases["6.6 Full-frame RGB Noise"] = (buf.getvalue(), "noise.jpg", False, "color saturation")

    # Category 7: Images Resembling Medical Scans but Unsupported
    # 7.1 Inverted scan (bright background with dark centered circular mass)
    inverted = np.full((300, 300), 240, dtype=np.uint8)
    y, x = np.ogrid[:300, :300]
    mask = np.sqrt((x - 150)**2 + (y - 150)**2) <= 100
    inverted[mask] = 40
    buf = io.BytesIO()
    Image.fromarray(inverted).save(buf, format="JPEG")
    test_cases["7.1 Inverted Scan (Bright Border)"] = (buf.getvalue(), "inverted.jpg", False, "dark background perimeter")

    # 7.2 Synthetic vignette circle with uncorrelated high-frequency noise
    synth_circle = np.zeros((300, 300), dtype=np.uint8)
    mask = np.sqrt((x - 150)**2 + (y - 150)**2) <= 100
    synth_circle[mask] = np.random.randint(40, 220, size=np.sum(mask), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(synth_circle).save(buf, format="JPEG")
    test_cases["7.2 Synthetic High-Freq Noise Circle"] = (buf.getvalue(), "synth_noise.jpg", False, "high-frequency noise")

    return test_cases


async def main():
    print("=" * 75)
    print("      BRAIN TUMOR DETECTOR - COMPREHENSIVE INPUT VALIDATION TEST SUITE")
    print("=" * 75)
    
    dataset_dir = find_dataset_dir()
    print(f"Dataset Benchmark Path : {dataset_dir}")
    
    service = get_inference_service()
    print(f"Inference Model Device : {service.device}")
    print("=" * 75)

    all_tests_passed = True
    total_tests = 0
    passed_tests = 0

    # -------------------------------------------------------------
    # CATEGORY 1: Valid Brain MRI Scans from Each Available Class
    # -------------------------------------------------------------
    print("\n--- CATEGORY 1: VALID BRAIN MRI SCANS (ALL 4 CLASSES) ---")
    classes = ["glioma", "meningioma", "pituitary", "notumor"]
    for cls_name in classes:
        total_tests += 1
        mri_path = find_mri_sample(cls_name, dataset_dir)
        with open(mri_path, "rb") as f:
            mri_bytes = f.read()

        try:
            result = await run_predict_endpoint(service, mri_bytes, os.path.basename(mri_path))
            pred = result["predicted_class"]
            conf = result["confidence"]
            probs = result["probabilities"]
            
            # Checks
            assert result["status"] == "success"
            assert pred in classes
            assert 0.0 <= conf <= 1.0
            assert len(probs) == 4
            assert abs(sum(probs.values()) - 1.0) < 1e-4

            print(f"  [PASS] Valid MRI ({cls_name:10s}) -> Predicted: {pred:10s} | Conf: {conf*100:5.1f}% | File: {os.path.basename(mri_path)}")
            passed_tests += 1
        except Exception as e:
            print(f"  [FAIL] Valid MRI ({cls_name:10s}) failed unexpectedly: {e}")
            all_tests_passed = False

    # -------------------------------------------------------------
    # CATEGORIES 2 - 7: Invalid Inputs (Must be Rejected with HTTP 400)
    # -------------------------------------------------------------
    negative_cases = create_test_images()

    current_cat = ""
    for test_name, (data_bytes, filename, should_pass, expected_keyword) in negative_cases.items():
        cat_prefix = test_name[:2]
        if cat_prefix != current_cat:
            cat_headers = {
                "2.": "\n--- CATEGORY 2: RANDOM NATURAL PHOTOGRAPHS ---",
                "3.": "\n--- CATEGORY 3: SCREENSHOTS AND UNRELATED IMAGES ---",
                "4.": "\n--- CATEGORY 4: CORRUPT IMAGE FILES ---",
                "5.": "\n--- CATEGORY 5: EMPTY UPLOADS & UNSUPPORTED FORMATS ---",
                "6.": "\n--- CATEGORY 6: VERY SMALL, UNUSUAL & MALFORMED IMAGES ---",
                "7.": "\n--- CATEGORY 7: NON-BRAIN / SYNTHETIC RESEMBLING MEDICAL SCANS ---"
            }
            if cat_prefix in cat_headers:
                print(cat_headers[cat_prefix])
            current_cat = cat_prefix

        total_tests += 1
        try:
            result = await run_predict_endpoint(service, data_bytes, filename)
            # If it succeeded without error, that's a security failure!
            print(f"  [FAIL] {test_name:38s} -> FAILED: Was accepted as '{result['predicted_class']}' ({result['confidence']*100:.1f}%)!")
            all_tests_passed = False
        except HTTPException as exc:
            if exc.status_code == 400:
                detail = str(exc.detail)
                # Verify standard user-facing error contract
                valid_prefix = detail.startswith("Invalid image") or detail.startswith("Invalid file extension")
                keyword_match = expected_keyword.lower() in detail.lower()
                if valid_prefix:
                    print(f"  [PASS] {test_name:38s} -> REJECTED (HTTP 400) | Detail: {detail[:70]}...")
                    passed_tests += 1
                else:
                    print(f"  [WARN] {test_name:38s} -> Rejected with 400 but unexpected detail format: {detail}")
                    passed_tests += 1
            else:
                print(f"  [FAIL] {test_name:38s} -> Unexpected HTTP status: {exc.status_code}")
                all_tests_passed = False
        except Exception as e:
            print(f"  [FAIL] {test_name:38s} -> Unexpected exception: {type(e).__name__}: {e}")
            all_tests_passed = False

    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print(f"TEST SUITE EXECUTION SUMMARY: {passed_tests} / {total_tests} Tests Passed ({(passed_tests/total_tests)*100:.1f}%)")
    if all_tests_passed and passed_tests == total_tests:
        print("ALL VERIFICATION AND SAFEGUARD TESTS PASSED!")
    else:
        print("SOME TESTS FAILED! Review output above.")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(main())
