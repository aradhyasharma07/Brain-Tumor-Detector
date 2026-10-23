import os
import io
from typing import Dict, Any, Union, Optional, List
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

from models.resnet_model import build_resnet50_model
from utils.dataset_utils import CLASS_NAMES, IDX_TO_CLASS
from services.validation_service import MriValidationService
from services.gradcam_service import GradCamService

CLASS_DISPLAY_NAMES: Dict[str, str] = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary Tumor",
    "notumor": "No Tumor Detected",
}

CANONICAL_CLASS_ORDER: List[str] = ["glioma", "meningioma", "pituitary", "notumor"]


class InferenceService:
    """
    Service for loading trained ResNet50 model checkpoint and running single-image inference
    with multi-layered brain MRI validation, out-of-distribution (OOD) protection,
    and Grad-CAM explainability.
    """
    def __init__(self, checkpoint_path: str = "backend/saved_models/resnet50_brain_tumor.pth", device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            alt_path = self.checkpoint_path.replace("backend/", "")
            if os.path.exists(alt_path):
                self.checkpoint_path = alt_path
            else:
                raise FileNotFoundError(f"Model checkpoint not found at: {self.checkpoint_path}")

        # Load Checkpoint
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Extract Class Mapping
        self.idx_to_class = self.checkpoint.get("idx_to_class", IDX_TO_CLASS)
        self.idx_to_class = {int(k): v for k, v in self.idx_to_class.items()}
        self.class_names = self.checkpoint.get("class_names", CLASS_NAMES)
        self.num_classes = len(self.class_names)

        # Reconstruct Model Architecture & Load State Dict
        self.model = build_resnet50_model(num_classes=self.num_classes, freeze_backbone=True)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Initialize Grad-CAM service sharing the loaded ResNet-50 instance
        self.gradcam_service = GradCamService(model=self.model, device=self.device)

        # Image Transformation Pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict_pil(
        self,
        image: Image.Image,
        validate: bool = True,
        include_gradcam: bool = True
    ) -> Dict[str, Any]:
        """
        Runs inference on a PIL Image object with optional pre-classification validation,
        deep-feature Out-of-Distribution screening, and Grad-CAM heatmap generation.
        """
        if validate:
            is_valid, domain_error = MriValidationService.validate_domain_properties(image)
            if not is_valid:
                raise ValueError(domain_error)

        image_rgb = image.convert("RGB")
        tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)  # Shape: [1, 3, 224, 224]

        with torch.no_grad():
            # Step through layers to capture intermediate features for OOD validation
            x = self.model.conv1(tensor)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            feat2048 = torch.flatten(x, 1)
            feat512 = self.model.fc[1](self.model.fc[0](feat2048))
            outputs = self.model.fc[3](self.model.fc[2](feat512))

            feature_norm = float(torch.norm(feat512).item())
            max_logit = float(torch.max(outputs).item())

            if validate:
                ood_valid, ood_error = MriValidationService.validate_deep_features(feature_norm, max_logit)
                if not ood_valid:
                    raise ValueError(ood_error)

            probabilities_tensor = torch.softmax(outputs, dim=1)[0]

        prob_values = probabilities_tensor.cpu().tolist()
        predicted_idx = int(torch.argmax(probabilities_tensor).item())
        predicted_class = self.idx_to_class.get(predicted_idx, CLASS_NAMES[predicted_idx])
        confidence = float(prob_values[predicted_idx])

        all_probabilities = {
            self.idx_to_class[i]: float(prob_values[i])
            for i in range(self.num_classes)
        }

        # Canonical ordered class scores breakdown for visualization
        class_scores = [
            {
                "class_name": cls,
                "display_name": CLASS_DISPLAY_NAMES.get(cls, cls.capitalize()),
                "score": float(all_probabilities.get(cls, 0.0)),
                "percentage": round(float(all_probabilities.get(cls, 0.0)) * 100, 2),
                "is_highest": (cls == predicted_class)
            }
            for cls in CANONICAL_CLASS_ORDER
            if cls in all_probabilities
        ]

        # Generate Grad-CAM overlay if requested
        overlay_pil, gradcam_image = None, None
        if include_gradcam:
            overlay_pil, gradcam_image = self.gradcam_service.generate_heatmap_and_overlay(
                input_tensor=tensor,
                original_image=image_rgb,
                target_class_index=predicted_idx
            )

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_index": predicted_idx,
            "probabilities": all_probabilities,
            "class_scores": class_scores,
            "gradcam_image": gradcam_image,
            "gradcam_overlay_pil": overlay_pil,
            "feature_norm": feature_norm,
            "max_logit": max_logit
        }

    def predict_image(
        self,
        image_path: str,
        validate: bool = True,
        include_gradcam: bool = True
    ) -> Dict[str, Any]:
        """
        Runs inference on a single MRI image file path.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found at path: {image_path}")

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        return self.predict_bytes(
            image_bytes,
            filename=os.path.basename(image_path),
            validate=validate,
            include_gradcam=include_gradcam
        )

    def predict_bytes(
        self,
        image_bytes: bytes,
        filename: Optional[str] = None,
        validate: bool = True,
        include_gradcam: bool = True
    ) -> Dict[str, Any]:
        """
        Runs inference on raw image bytes with complete multi-stage validation.
        """
        if validate:
            is_valid, error_msg, pil_img = MriValidationService.validate_upload(image_bytes, filename=filename)
            if not is_valid:
                raise ValueError(error_msg)
            return self.predict_pil(pil_img, validate=True, include_gradcam=include_gradcam)
        else:
            image = Image.open(io.BytesIO(image_bytes))
            return self.predict_pil(image, validate=False, include_gradcam=include_gradcam)
