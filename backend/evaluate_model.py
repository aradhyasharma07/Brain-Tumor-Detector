import os
import sys
import csv
import argparse
from collections import Counter
from PIL import Image

# Ensure backend directory is discoverable
sys.path.insert(0, os.path.dirname(__file__))

from services.inference_service import InferenceService


# ==============================
# CLASS INFORMATION
# ==============================

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "pituitary",
    "notumor"
]

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


def find_testing_dir(explicit_dir=None):
    if explicit_dir:
        candidate_testing = (
            explicit_dir
            if os.path.basename(explicit_dir.rstrip("/\\")).lower() == "testing"
            else os.path.join(explicit_dir, "Testing")
        )
        if os.path.isdir(candidate_testing):
            return os.path.abspath(candidate_testing)
        if os.path.isdir(explicit_dir):
            return os.path.abspath(explicit_dir)
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
        candidate_testing = os.path.join(candidate_dir, "Testing")
        if os.path.isdir(candidate_testing):
            return os.path.abspath(candidate_testing)
        return os.path.abspath(candidate_dir)

    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Brain Tumor ResNet50 Single-Image Inference Evaluation"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to dataset root or Testing folder."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="backend/saved_models/resnet50_brain_tumor.pth",
        help="Path to trained model checkpoint."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to save output test_results.csv."
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Optional limit on number of images per class for fast testing."
    )
    return parser.parse_args()


# ==============================
# PURE PYTHON METRIC UTILITIES
# ==============================

def compute_metrics(actual_labels, predicted_labels, class_names):
    """
    Computes accuracy, confusion matrix, and classification report without requiring sklearn.
    """
    total = len(actual_labels)
    if total == 0:
        return 0.0, {}, [[0] * len(class_names) for _ in class_names]

    correct = sum(1 for a, p in zip(actual_labels, predicted_labels) if a == p)
    accuracy = correct / total

    num_classes = len(class_names)
    cls_to_idx = {name: i for i, name in enumerate(class_names)}

    # Confusion matrix: rows = actual, columns = predicted
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for a, p in zip(actual_labels, predicted_labels):
        if a in cls_to_idx and p in cls_to_idx:
            matrix[cls_to_idx[a]][cls_to_idx[p]] += 1

    per_class = {}
    for i, name in enumerate(class_names):
        tp = matrix[i][i]
        support = sum(matrix[i])
        fp = sum(matrix[j][i] for j in range(num_classes) if j != i)
        fn = sum(matrix[i][j] for j in range(num_classes) if j != i)

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support
        }

    return accuracy, per_class, matrix


def format_classification_report(per_class, class_names, accuracy):
    """
    Formats metrics identical to standard classification_report string.
    """
    total_support = sum(m["support"] for m in per_class.values())
    lines = []
    lines.append(f"{'':15}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}")
    lines.append("")

    for name in class_names:
        m = per_class[name]
        lines.append(
            f"{name:15}{m['precision']:10.4f}{m['recall']:10.4f}{m['f1-score']:10.4f}{m['support']:10d}"
        )
    lines.append("")

    # Overall accuracy line
    lines.append(f"{'accuracy':15}{'':10}{'':10}{accuracy:10.4f}{total_support:10d}")

    # Macro and weighted averages
    if len(class_names) > 0:
        macro_p = sum(m["precision"] for m in per_class.values()) / len(class_names)
        macro_r = sum(m["recall"] for m in per_class.values()) / len(class_names)
        macro_f1 = sum(m["f1-score"] for m in per_class.values()) / len(class_names)
        lines.append(f"{'macro avg':15}{macro_p:10.4f}{macro_r:10.4f}{macro_f1:10.4f}{total_support:10d}")

    if total_support > 0:
        weight_p = sum(m["precision"] * m["support"] for m in per_class.values()) / total_support
        weight_r = sum(m["recall"] * m["support"] for m in per_class.values()) / total_support
        weight_f1 = sum(m["f1-score"] * m["support"] for m in per_class.values()) / total_support
        lines.append(f"{'weighted avg':15}{weight_p:10.4f}{weight_r:10.4f}{weight_f1:10.4f}{total_support:10d}")

    return "\n".join(lines)


def main():
    args = parse_args()

    print("=" * 60)
    print("BRAIN TUMOR MODEL EVALUATION")
    print("=" * 60)

    # 1. Resolve Testing Directory
    testing_path = find_testing_dir(args.data_dir)
    if not testing_path or not os.path.exists(testing_path):
        print("\n" + "!" * 60)
        print("ERROR: Testing dataset folder not found.")
        print("Please provide path to your dataset containing Testing/ via --data-dir:")
        print("  python evaluate_model.py --data-dir <path_to_dataset>")
        print("!" * 60)
        sys.exit(1)

    output_csv = args.output_csv or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_results.csv"
    )

    print("\nLoading ResNet50 model...")
    model = InferenceService(checkpoint_path=args.checkpoint_path)
    print("Model loaded successfully.")
    print(f"Testing dataset: {testing_path}")
    print(f"Output CSV     : {output_csv}")

    # 2. Run Predictions
    actual_labels = []
    predicted_labels = []
    results = []

    total_images = 0
    failed_images = 0

    print("\nStarting evaluation...\n")

    for class_name in CLASS_NAMES:
        class_folder = os.path.join(testing_path, class_name)

        if not os.path.exists(class_folder):
            print(f"WARNING: Missing folder: {class_folder}")
            continue

        image_files = [
            file_name
            for file_name in os.listdir(class_folder)
            if file_name.lower().endswith(VALID_EXTENSIONS)
        ]

        if args.limit_per_class:
            image_files = image_files[:args.limit_per_class]

        print(f"Testing {class_name:12s}: {len(image_files)} images")

        for file_name in image_files:
            image_path = os.path.join(class_folder, file_name)

            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    prediction = model.predict_pil(image)

                predicted_class = prediction["predicted_class"]
                confidence = prediction["confidence"]

                actual_labels.append(class_name)
                predicted_labels.append(predicted_class)

                results.append({
                    "image": file_name,
                    "actual": class_name,
                    "predicted": predicted_class,
                    "confidence": confidence
                })

                total_images += 1

            except Exception as error:
                failed_images += 1
                print(f"ERROR: {file_name} -> {error}")

    # 3. Calculate Metrics
    accuracy, per_class, matrix = compute_metrics(actual_labels, predicted_labels, CLASS_NAMES)

    print("\n")
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print(f"\nTotal images tested : {total_images}")
    print(f"Failed images       : {failed_images}")
    print(f"\nAccuracy            : {accuracy * 100:.2f}%")

    # 4. Classification Report
    print("\n")
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(format_classification_report(per_class, CLASS_NAMES, accuracy))

    # 5. Confusion Matrix
    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    print("\nRows = Actual | Columns = Predicted\n")

    print(f"{'':15}" + "".join(f"{name[:12]:>14}" for name in CLASS_NAMES))
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name:15}" + "".join(f"{matrix[i][j]:>14}" for j in range(len(CLASS_NAMES))))

    # 6. Save CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image", "actual", "predicted", "confidence"]
        )
        writer.writeheader()
        writer.writerows(results)

    # 7. Class Counts
    print("\n")
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print("\nActual images:")
    actual_counts = Counter(actual_labels)
    for class_name in CLASS_NAMES:
        print(f"  {class_name:12s}: {actual_counts[class_name]}")

    print("\nPredicted images:")
    predicted_counts = Counter(predicted_labels)
    for class_name in CLASS_NAMES:
        print(f"  {class_name:12s}: {predicted_counts[class_name]}")

    print(f"\nDetailed results saved to: {output_csv}")
    print("\n" + "=" * 60)
    print("TESTING FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()
