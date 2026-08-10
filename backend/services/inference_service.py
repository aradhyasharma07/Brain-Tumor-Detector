import os
from typing import Dict, Any, Union
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

from backend.models.resnet_model import build_resnet50_model
from backend.utils.dataset_utils import CLASS_NAMES, IDX_TO_CLASS


class InferenceService:
    """
    Service for loading trained ResNet50 model checkpoint and running single-image inference.
    """
    def __init__(self, checkpoint_path: str = "backend/saved_models/resnet50_brain_tumor.pth", device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {self.checkpoint_path}")

        # Load Checkpoint
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Extract Class Mapping
        self.idx_to_class = self.checkpoint.get("idx_to_class", IDX_TO_CLASS)
        # Ensure keys in idx_to_class are integer
        self.idx_to_class = {int(k): v for k, v in self.idx_to_class.items()}
        self.class_names = self.checkpoint.get("class_names", CLASS_NAMES)
        self.num_classes = len(self.class_names)

        # Reconstruct Model Architecture & Load State Dict
        self.model = build_resnet50_model(num_classes=self.num_classes, freeze_backbone=True)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Image Transformation Pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict_pil(self, image: Image.Image) -> Dict[str, Any]:
        """
        Runs inference on a PIL Image object.
        """
        image_rgb = image.convert("RGB")
        tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)  # Shape: [1, 3, 224, 224]

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities_tensor = torch.softmax(outputs, dim=1)[0]

        prob_values = probabilities_tensor.cpu().tolist()
        predicted_idx = int(torch.argmax(probabilities_tensor).item())
        predicted_class = self.idx_to_class.get(predicted_idx, CLASS_NAMES[predicted_idx])
        confidence = float(prob_values[predicted_idx])

        all_probabilities = {
            self.idx_to_class[i]: float(prob_values[i])
            for i in range(self.num_classes)
        }

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_index": predicted_idx,
            "probabilities": all_probabilities
        }

    def predict_image(self, image_path: str) -> Dict[str, Any]:
        """
        Runs inference on a single MRI image file path.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found at path: {image_path}")

        image = Image.open(image_path)
        return self.predict_pil(image)

    def predict_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Runs inference on raw image bytes (e.g. from FastAPI file upload).
        """
        import io
        image = Image.open(io.BytesIO(image_bytes))
        return self.predict_pil(image)

