import io
import os
from typing import Tuple, Optional, Dict, Any
from PIL import Image, ImageOps
import numpy as np
import cv2


class MriValidationService:
    """
    Validation service for verifying whether an uploaded image is a valid,
    supported axial Brain MRI scan prior to running deep learning tumor classification.
    
    Implements a multi-layered defense-in-depth approach:
      Layer 1: Payload and Safe Image Decoding (format, size, corrupt data, dimensions)
      Layer 2: Visual & Photometric Domain Checks (saturation, dark border, center contrast)
      Layer 3: Morphological Cranial Structure Verification (contour, centroid, area)
      Layer 4: Deep Feature & Logit Out-of-Distribution (OOD) Screening
    """

    # Security & Size limits
    MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "BMP", "WEBP"}
    MIN_DIMENSION = 64
    MAX_DIMENSION = 4096
    MIN_ASPECT_RATIO = 0.33
    MAX_ASPECT_RATIO = 3.0

    # Photometric & Domain bounds (empirically calibrated on benchmark dataset)
    MAX_COLOR_SATURATION = 0.12       # Real brain MRIs: mean 0.0006, max 0.067; photos: >0.25
    MIN_DARK_BORDER_FRACTION = 0.60   # Real brain MRIs: mean 0.955, min 0.74; natural scenes: <0.40
    MAX_BORDER_MEAN_INTENSITY = 0.30  # Real brain MRIs: mean 0.041, max 0.245; bright scenes: >0.35
    MIN_CENTER_MEAN_INTENSITY = 0.05  # Rejects pure black / near-black images
    MAX_CENTER_MEAN_INTENSITY = 0.85  # Rejects pure white / inverted / overexposed images
    MIN_CENTER_CONTRAST_STD = 0.04   # Real brain MRIs: mean 0.25, min 0.083; solid blocks: 0.0
    MAX_LAPLACIAN_VARIANCE = 4000.0   # Real brain MRIs: <1500; synthetic uncorrelated noise: >10,000
    MIN_FOREGROUND_AREA_RATIO = 0.10  # Cranial structure coverage fraction
    CENTROID_MIN_RATIO = 0.25
    CENTROID_MAX_RATIO = 0.75

    # Deep feature OOD bounds
    MAX_FEATURE_NORM = 3500.0         # Real brain MRIs: max 2822; OOD noise/photos: >20,000
    MAX_LOGIT_VALUE = 500.0           # Real brain MRIs: max 318; OOD inputs: >2500

    @classmethod
    def validate_upload(
        cls,
        image_bytes: bytes,
        filename: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[Image.Image]]:
        """
        Validates raw uploaded bytes before feeding to the classifier.
        
        Returns:
            Tuple[bool, Optional[str], Optional[Image.Image]]:
                (is_valid, error_message, decoded_pil_image)
        """
        # --- Stage 1: Payload Checks ---
        if not image_bytes or len(image_bytes) == 0:
            return False, "Invalid image. Uploaded file is empty.", None

        if len(image_bytes) > cls.MAX_FILE_SIZE_BYTES:
            max_mb = cls.MAX_FILE_SIZE_BYTES // (1024 * 1024)
            return False, f"Invalid image. File size exceeds the {max_mb}MB limit.", None

        if filename:
            ext = os.path.splitext(filename)[1].lower()
            if ext and ext not in cls.ALLOWED_EXTENSIONS:
                allowed_str = ", ".join(sorted(cls.ALLOWED_EXTENSIONS))
                return False, f"Invalid file extension '{ext}'. Allowed extensions: {allowed_str}", None

        # --- Stage 2: Safe Image Decoding ---
        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Verify decoded format
            if image.format not in cls.ALLOWED_PIL_FORMATS:
                return (
                    False,
                    f"Invalid image. Unsupported format '{image.format}'. Supported formats: JPEG, PNG, BMP, WEBP.",
                    None
                )
            # Force full pixel load to detect truncated/corrupted files
            image.load()
        except Exception as e:
            return False, f"Invalid image. Uploaded file could not be parsed as a valid image: {str(e)}", None

        # Animated / multi-frame rejection
        if getattr(image, "is_animated", False) and getattr(image, "n_frames", 1) > 1:
            return False, "Invalid image. Animated or multi-frame images are not supported.", None

        # Dimensions & Aspect Ratio
        width, height = image.size
        if width < cls.MIN_DIMENSION or height < cls.MIN_DIMENSION:
            return (
                False,
                f"Invalid image. Image dimensions ({width}x{height}) are too small. Minimum resolution is {cls.MIN_DIMENSION}x{cls.MIN_DIMENSION}.",
                None
            )
        if width > cls.MAX_DIMENSION or height > cls.MAX_DIMENSION:
            return (
                False,
                f"Invalid image. Image dimensions ({width}x{height}) exceed maximum allowed {cls.MAX_DIMENSION}x{cls.MAX_DIMENSION}.",
                None
            )

        aspect_ratio = width / height
        if aspect_ratio < cls.MIN_ASPECT_RATIO or aspect_ratio > cls.MAX_ASPECT_RATIO:
            return (
                False,
                f"Invalid image. Unsupported aspect ratio ({aspect_ratio:.2f}). Please upload a standard axial brain MRI scan.",
                None
            )

        # --- Stage 3: Visual & Photometric Domain Checks ---
        valid, domain_error = cls.validate_domain_properties(image)
        if not valid:
            return False, domain_error, None

        return True, None, image

    @classmethod
    def validate_domain_properties(cls, image: Image.Image) -> Tuple[bool, Optional[str]]:
        """
        Evaluates visual domain characteristics of a brain MRI:
        - Chromaticity / Saturation (brain MRIs are grayscale acquisitions)
        - Peripheral Dark Perimeter (the dark scanner frame enclosing the cranial mass)
        - Central Brain Tissue Contrast (presence of organic grey/white matter variations)
        - Cranial Morphological Centering (elliptical skull contour in center of frame)
        """
        width, height = image.size
        rgb = image.convert("RGB")
        arr = np.array(rgb, dtype=np.float32) / 255.0

        # 1. Color Saturation Check
        mx = arr.max(axis=-1)
        mn = arr.min(axis=-1)
        sat = np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0.0)
        mean_sat = float(sat.mean())
        if mean_sat > cls.MAX_COLOR_SATURATION:
            return (
                False,
                f"Invalid image. Please upload a supported brain MRI scan. (Color saturation {mean_sat:.2f} is too high; brain MRIs are grayscale scans)"
            )

        # 2. Peripheral Dark Border Check (Scanner Frame)
        bw = max(1, int(width * 0.05))
        bh = max(1, int(height * 0.05))
        top = arr[:bh, :, :]
        bottom = arr[-bh:, :, :]
        left = arr[:, :bw, :]
        right = arr[:, -bw:, :]
        border_pixels = np.concatenate([
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3)
        ])
        border_mean = float(border_pixels.mean())
        dark_border_fraction = float(np.mean(border_pixels.mean(axis=1) < 0.28))

        if dark_border_fraction < cls.MIN_DARK_BORDER_FRACTION or border_mean > cls.MAX_BORDER_MEAN_INTENSITY:
            return (
                False,
                f"Invalid image. Please upload a supported brain MRI scan. (Missing dark background perimeter typical of axial brain MRI scans)"
            )

        # 3. Central Anatomical Tissue Contrast
        center = arr[bh * 2 : -bh * 2, bw * 2 : -bw * 2, :]
        center_mean = float(center.mean())
        center_std = float(center.std())

        if center_mean < cls.MIN_CENTER_MEAN_INTENSITY:
            return (
                False,
                "Invalid image. Please upload a supported brain MRI scan. (Image is completely or predominantly black)"
            )
        if center_mean > cls.MAX_CENTER_MEAN_INTENSITY:
            return (
                False,
                "Invalid image. Please upload a supported brain MRI scan. (Image is predominantly white or overexposed)"
            )
        if center_std < cls.MIN_CENTER_CONTRAST_STD:
            return (
                False,
                "Invalid image. Please upload a supported brain MRI scan. (Image lacks structural contrast or tissue variation)"
            )

        # 4. Cranial Morphological Centering & Structure Check
        gray = np.array(image.convert("L"), dtype=np.uint8)
        
        # High-frequency noise check (Laplacian variance on center)
        center_gray = gray[bh * 2 : -bh * 2, bw * 2 : -bw * 2]
        lap_var = float(cv2.Laplacian(center_gray, cv2.CV_64F).var())
        if lap_var > cls.MAX_LAPLACIAN_VARIANCE:
            return (
                False,
                "Invalid image. Please upload a supported brain MRI scan. (Excessive high-frequency noise or unnatural pattern detected)"
            )

        # Foreground contour segmentation
        _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (
                False,
                "Invalid image. Please upload a supported brain MRI scan. (No anatomical cranial structure detected)"
            )

        largest = max(contours, key=cv2.contourArea)
        area_ratio = float(cv2.contourArea(largest) / (width * height))
        if area_ratio < cls.MIN_FOREGROUND_AREA_RATIO:
            return (
                False,
                f"Invalid image. Please upload a supported brain MRI scan. (Foreground anatomical region is too small: {area_ratio:.2f})"
            )

        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = (M["m10"] / M["m00"]) / width
            cy = (M["m01"] / M["m00"]) / height
            if (
                cx < cls.CENTROID_MIN_RATIO
                or cx > cls.CENTROID_MAX_RATIO
                or cy < cls.CENTROID_MIN_RATIO
                or cy > cls.CENTROID_MAX_RATIO
            ):
                return (
                    False,
                    f"Invalid image. Please upload a supported brain MRI scan. (Anatomical structure is not centrally positioned: cx={cx:.2f}, cy={cy:.2f})"
                )

        return True, None

    @classmethod
    def validate_deep_features(
        cls,
        feature_norm: float,
        max_logit: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Deep-feature Out-of-Distribution (OOD) check on intermediate ResNet-50 activations.
        Catches synthetic patterns or adversarial images that bypass low-level photometric checks.
        """
        if feature_norm > cls.MAX_FEATURE_NORM:
            return (
                False,
                f"Invalid image. Please upload a supported brain MRI scan. (Deep feature activation norm {feature_norm:.1f} indicates an out-of-distribution input)"
            )
        if max_logit > cls.MAX_LOGIT_VALUE:
            return (
                False,
                f"Invalid image. Please upload a supported brain MRI scan. (Anomalous logit activation {max_logit:.1f} indicates an out-of-distribution input)"
            )
        return True, None
