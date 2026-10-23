import os
import sys
import io
import asyncio
import unittest
from PIL import Image
import numpy as np

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import UploadFile, HTTPException
from services.inference_service import InferenceService
from services.validation_service import MriValidationService
from services.gradcam_service import GradCamService
from services.pdf_service import PdfReportService
from app.routers.prediction_router import predict_mri, generate_pdf_report


def get_sample_mri_path() -> str:
    candidates = [
        r"D:\SEM 5 all projects\Brain Tumor Dataset\Training\glioma\Tr-gl_1.jpg",
        r"D:\SEM 5 all projects\Brain Tumor Dataset\Testing\glioma\Te-gl_1.jpg",
        r"D:\SEM 5 all projects\Brain Tumor Dataset\Testing\notumor\Te-no_10.jpg",
        os.path.join(os.path.dirname(__file__), "..", "..", "Brain Tumor Dataset", "Training", "glioma", "Tr-gl_1.jpg")
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    raise FileNotFoundError("Could not find a sample MRI image in dataset paths.")


class TestFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint = os.path.join(os.path.dirname(__file__), "saved_models", "resnet50_brain_tumor.pth")
        cls.service = InferenceService(checkpoint_path=checkpoint)
        cls.sample_path = get_sample_mri_path()
        with open(cls.sample_path, "rb") as f:
            cls.sample_bytes = f.read()
        cls.sample_pil = Image.open(io.BytesIO(cls.sample_bytes)).convert("RGB")

    # --- Feature 1: Confidence Score Visualization Tests ---
    def test_feature1_class_scores_structure_and_ordering(self):
        result = self.service.predict_pil(self.sample_pil, validate=True, include_gradcam=False)
        self.assertIn("class_scores", result)
        self.assertIn("probabilities", result)

        class_scores = result["class_scores"]
        self.assertEqual(len(class_scores), 4)

        # Check canonical ordering: glioma, meningioma, pituitary, notumor
        expected_order = ["glioma", "meningioma", "pituitary", "notumor"]
        actual_order = [item["class_name"] for item in class_scores]
        self.assertEqual(actual_order, expected_order)

        # Ensure probabilities sum to 1.0 (after softmax)
        prob_sum = sum(result["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=4)

        # Ensure highest class is correctly flagged
        top_items = [item for item in class_scores if item["is_highest"]]
        self.assertEqual(len(top_items), 1)
        self.assertEqual(top_items[0]["class_name"], result["predicted_class"])
        self.assertEqual(top_items[0]["score"], result["confidence"])

    def test_feature1_real_model_outputs_not_fabricated(self):
        result = self.service.predict_pil(self.sample_pil, validate=True, include_gradcam=False)
        # Verify scores are real floats with non-trivial precision
        for item in result["class_scores"]:
            self.assertIsInstance(item["score"], float)
            self.assertGreaterEqual(item["score"], 0.0)
            self.assertLessEqual(item["score"], 1.0)
            self.assertEqual(item["percentage"], round(item["score"] * 100, 2))

    # --- Feature 2: Grad-CAM Heatmap Tests ---
    def test_feature2_gradcam_generation(self):
        result = self.service.predict_pil(self.sample_pil, validate=True, include_gradcam=True)
        self.assertIn("gradcam_image", result)
        self.assertIsNotNone(result["gradcam_image"])
        self.assertTrue(result["gradcam_image"].startswith("data:image/jpeg;base64,"))

        # Check overlay PIL dimensions
        overlay_pil = result["gradcam_overlay_pil"]
        self.assertIsNotNone(overlay_pil)
        self.assertEqual(overlay_pil.size, (224, 224))

    def test_feature2_gradcam_hook_cleanup_and_stability(self):
        # Run multiple Grad-CAM generations consecutively to verify hook removal
        for _ in range(3):
            result = self.service.predict_pil(self.sample_pil, validate=True, include_gradcam=True)
            self.assertIsNotNone(result["gradcam_image"])

    # --- Feature 3: PDF Report Generation Tests ---
    def test_feature3_pdf_generation_bytes_and_signature(self):
        result = self.service.predict_pil(self.sample_pil, validate=True, include_gradcam=True)
        pdf_bytes = PdfReportService.generate_report(
            prediction_result=result,
            original_image=self.sample_pil,
            overlay_image=result.get("gradcam_overlay_pil"),
            report_id="TEST-REP-001"
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        # PDF magic number signature and EOF
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))
        self.assertTrue(b"%%EOF" in pdf_bytes)
        self.assertIn(b"ReportLab", pdf_bytes)

    # --- Integration Tests: Router Endpoints & Validation Rejection ---
    def test_router_predict_endpoint_success(self):
        upload_file = UploadFile(
            filename="sample.jpg",
            file=io.BytesIO(self.sample_bytes),
            headers={"content-type": "image/jpeg"}
        )
        response = asyncio.run(predict_mri(file=upload_file, service=self.service))
        self.assertEqual(response.status, "success")
        self.assertIn(response.predicted_class, ["glioma", "meningioma", "pituitary", "notumor"])
        self.assertIsNotNone(response.class_scores)
        self.assertEqual(len(response.class_scores), 4)
        self.assertIsNotNone(response.gradcam_image)
        self.assertIsNotNone(response.prediction_id)

    def test_router_report_endpoint_success(self):
        upload_file = UploadFile(
            filename="sample.jpg",
            file=io.BytesIO(self.sample_bytes),
            headers={"content-type": "image/jpeg"}
        )
        response = asyncio.run(generate_pdf_report(file=upload_file, service=self.service))
        self.assertEqual(response.media_type, "application/pdf")
        self.assertTrue(response.body.startswith(b"%PDF-"))
        self.assertIn("Content-Disposition", response.headers)
        self.assertIn("neuroscan_report_", response.headers["Content-Disposition"])

    def test_validation_failure_prevents_prediction_and_report(self):
        # 1. Invalid file format (e.g. text/plain)
        bad_file = UploadFile(
            filename="notes.txt",
            file=io.BytesIO(b"This is a text file, not a brain MRI."),
            headers={"content-type": "text/plain"}
        )
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(predict_mri(file=bad_file, service=self.service))
        self.assertEqual(ctx.exception.status_code, 400)

        # 2. Invalid non-MRI image (e.g. colorful photo)
        color_photo = np.zeros((200, 200, 3), dtype=np.uint8)
        color_photo[:, :, 0] = 200  # Red
        buf = io.BytesIO()
        Image.fromarray(color_photo).save(buf, format="JPEG")
        photo_file = UploadFile(
            filename="photo.jpg",
            file=io.BytesIO(buf.getvalue()),
            headers={"content-type": "image/jpeg"}
        )
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(predict_mri(file=photo_file, service=self.service))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Invalid image", ctx.exception.detail)

        # 3. PDF report generation on rejected image
        photo_file.file.seek(0)
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(generate_pdf_report(file=photo_file, service=self.service))
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
