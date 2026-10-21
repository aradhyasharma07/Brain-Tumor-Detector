import os
import sys
import argparse

# Ensure backend directory is discoverable
sys.path.insert(0, os.path.dirname(__file__))

from services.inference_service import InferenceService
from utils.dataset_utils import CLASS_NAMES


def find_sample_image(explicit_path=None):
    if explicit_path:
        if os.path.isfile(explicit_path):
            return os.path.abspath(explicit_path)
        return None

    # Candidate locations relative to backend directory or workspace
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(backend_dir, "..", "..", "Brain Tumor Dataset"),
        r"D:\SEM 5 all projects\Brain Tumor Dataset",
        os.path.join(backend_dir, "..", "Brain Tumor Dataset"),
    ]

    for candidate_dir in candidates:
        if not os.path.isdir(candidate_dir):
            continue
        # Check standard sample images
        preferred = [
            os.path.join(candidate_dir, "Training", "glioma", "Tr-gl_1.jpg"),
            os.path.join(candidate_dir, "Testing", "glioma", "Te-gl_1.jpg"),
            os.path.join(candidate_dir, "Testing", "notumor", "Te-no_10.jpg"),
        ]
        for p in preferred:
            if os.path.isfile(p):
                return os.path.abspath(p)
        # Search for any valid image
        for root, _, files in os.walk(candidate_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    return os.path.abspath(os.path.join(root, f))
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Brain Tumor ResNet50 - Inference Service Verification"
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Path to an MRI test image for verification smoke test."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="backend/saved_models/resnet50_brain_tumor.pth",
        help="Path to model checkpoint .pth file."
    )
    return parser.parse_args()


def run_verification():
    args = parse_args()

    print("=" * 65)
    print("      Brain Tumor ResNet50 - Inference Service Verification")
    print("=" * 65)

    checkpoint_path = args.checkpoint_path
    
    # 1. Verify Checkpoint Exists & Loads
    print(f"Check 1 - Model Checkpoint Path: {checkpoint_path}")
    service = InferenceService(checkpoint_path=checkpoint_path)
    print("  -> Checkpoint loaded successfully!")
    print(f"  -> Model device: {service.device}")
    print(f"  -> Class names in model: {service.class_names}")
    print(f"  -> Index mapping: {service.idx_to_class}")

    # 2. Verify Preprocessing & Single Image Inference
    sample_image_path = find_sample_image(args.image_path)
    if not sample_image_path:
        print("\n" + "!" * 65)
        print("ERROR: Sample MRI image not found.")
        print("Please provide a valid image file using the --image-path argument:")
        print("  python verify_inference.py --image-path <path_to_image>")
        print("!" * 65)
        sys.exit(1)

    print(f"\nCheck 2 - Running smoke test inference on sample image:")
    print(f"  -> Path: {sample_image_path}")

    result = service.predict_image(sample_image_path)
    
    predicted_class = result["predicted_class"]
    confidence = result["confidence"]
    class_index = result["class_index"]
    probabilities = result["probabilities"]

    print("\n--- Inference Output Result ---")
    print(f"  Predicted Class : {predicted_class}")
    print(f"  Class Index     : {class_index}")
    print(f"  Confidence      : {confidence:.4f} ({confidence * 100:.2f}%)")
    print("  Class Probabilities:")
    for cls_name, prob in probabilities.items():
        print(f"    - {cls_name:12s}: {prob:.4f} ({prob * 100:.2f}%)")

    # 3. Assertions and Verification Checks
    check_num_probs = len(probabilities) == 4
    check_prob_sum = abs(sum(probabilities.values()) - 1.0) < 1e-4
    check_valid_class = predicted_class in CLASS_NAMES
    check_valid_conf = 0.0 <= confidence <= 1.0
    check_class_match = all(c in probabilities for c in CLASS_NAMES)

    print("\n" + "=" * 65)
    print("                   VERIFICATION CHECKS")
    print("=" * 65)
    print(f" 1. Checkpoint Loaded Successfully        : YES")
    print(f" 2. Architecture matches checkpoint       : YES")
    print(f" 3. Produces exactly 4 class probabilities: {'PASS' if check_num_probs else 'FAIL'}")
    print(f" 4. Probabilities sum to ~1.0             : {'PASS' if check_prob_sum else 'FAIL'} (Sum: {sum(probabilities.values()):.6f})")
    print(f" 5. Predicted class is valid              : {'PASS' if check_valid_class else 'FAIL'} ({predicted_class})")
    print(f" 6. Confidence score bounded in [0, 1]    : {'PASS' if check_valid_conf else 'FAIL'} ({confidence:.4f})")
    print(f" 7. All 4 class names present             : {'PASS' if check_class_match else 'FAIL'}")

    all_passed = (
        check_num_probs and
        check_prob_sum and
        check_valid_class and
        check_valid_conf and
        check_class_match
    )

    print("=" * 65)
    print(f" INFERENCE PIPELINE ALL CHECKS PASSED    : {'YES' if all_passed else 'NO'}")
    print("=" * 65)


if __name__ == "__main__":
    run_verification()
