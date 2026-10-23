import io
import base64
import threading
from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCamService:
    """
    Grad-CAM service for generating class activation heatmaps on ResNet-50.
    Targets the final convolutional block (layer4[-1]) to visualize regions
    influencing the model's diagnostic classification.
    
    Thread-safe implementation with proper hook lifecycle management.
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        # Use final Bottleneck block in layer4 of ResNet-50
        self.target_layer = self.model.layer4[-1]
        self._lock = threading.Lock()

    def generate_heatmap_and_overlay(
        self,
        input_tensor: torch.Tensor,
        original_image: Image.Image,
        target_class_index: Optional[int] = None,
        alpha: float = 0.45
    ) -> Tuple[Optional[Image.Image], Optional[str]]:
        """
        Computes Grad-CAM for the given input tensor and target class index,
        and generates an alpha-blended overlay image onto the original MRI.
        
        Args:
            input_tensor: Preprocessed image tensor [1, 3, 224, 224] on model device.
            original_image: Original PIL Image prior to tensor normalization.
            target_class_index: Integer class index to explain. If None, uses model argmax.
            alpha: Heatmap blend weight (0.0 to 1.0).
            
        Returns:
            Tuple[Optional[Image.Image], Optional[str]]:
                (overlay_pil_image, base64_data_uri_string)
        """
        with self._lock:
            activations = []
            gradients = []

            def forward_hook(module, inp, out):
                activations.append(out)

            def backward_hook(module, grad_in, grad_out):
                gradients.append(grad_out[0])

            handle_fwd = self.target_layer.register_forward_hook(forward_hook)
            handle_bwd = self.target_layer.register_full_backward_hook(backward_hook)

            try:
                # Ensure model is in eval mode and gradients are zeroed
                self.model.zero_grad()

                # Enable gradient calculation even if caller is outside no_grad
                with torch.enable_grad():
                    # Ensure input tensor requires grad or activations retain grad
                    tensor_input = input_tensor.clone().detach().requires_grad_(True)
                    outputs = self.model(tensor_input)

                    if target_class_index is None:
                        target_class_index = int(outputs.argmax(dim=1).item())

                    target_score = outputs[0, target_class_index]
                    target_score.backward()

                if not activations or not gradients:
                    return None, None

                act = activations[0].detach()  # Shape: [1, 2048, 7, 7]
                grad = gradients[0].detach()   # Shape: [1, 2048, 7, 7]

                # Global Average Pooling of gradients across spatial dimensions
                weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # [1, 2048, 1, 1]

                # Weighted linear combination of activation maps
                cam = torch.sum(weights * act, dim=1, keepdim=True)  # [1, 1, 7, 7]
                cam = F.relu(cam)  # Discard negative contributions
                cam_np = cam.squeeze().cpu().numpy()

                # Normalize CAM to [0.0, 1.0]
                cam_min = float(cam_np.min())
                cam_max = float(cam_np.max())
                if cam_max > cam_min:
                    cam_normalized = (cam_np - cam_min) / (cam_max - cam_min)
                else:
                    cam_normalized = np.zeros_like(cam_np)

                # Resize original image to standard 224x224 for overlay alignment
                orig_rgb = original_image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                orig_arr = np.array(orig_rgb)

                # Upsample CAM to 224x224
                cam_resized = cv2.resize(cam_normalized, (224, 224), interpolation=cv2.INTER_LINEAR)
                heatmap_uint8 = np.uint8(255 * cam_resized)

                # Colorize heatmap with JET colormap
                heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

                # Alpha blend heatmap over original MRI
                overlay_arr = np.uint8(alpha * heatmap_rgb + (1.0 - alpha) * orig_arr)
                overlay_pil = Image.fromarray(overlay_arr)

                # Encode to base64 JPEG data URI
                buffer = io.BytesIO()
                overlay_pil.save(buffer, format="JPEG", quality=90)
                b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
                data_uri = f"data:image/jpeg;base64,{b64_data}"

                return overlay_pil, data_uri

            except Exception as e:
                # Log or swallow safely without breaking the prediction
                print(f"[GradCamService Warning] Failed to compute Grad-CAM: {e}")
                return None, None
            finally:
                handle_fwd.remove()
                handle_bwd.remove()
                self.model.zero_grad()
