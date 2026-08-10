import os
import argparse
import sys
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure backend package is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.models.resnet_model import build_resnet50_model
from backend.utils.dataset_utils import (
    BrainTumorDataset,
    get_transforms,
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    VALID_EXTENSIONS
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Brain Tumor ResNet50 Testing Dataset Evaluation"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to dataset root folder containing Testing/ directory."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="backend/saved_models/resnet50_brain_tumor.pth",
        help="Path to saved model checkpoint PyTorch .pth file."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation DataLoader."
    )
    return parser.parse_args()


def load_test_dataset(data_dir: str) -> Tuple[BrainTumorDataset, int]:
    testing_dir = data_dir
    if not os.path.basename(data_dir.rstrip("/\\")).lower() == "testing":
        potential_testing = os.path.join(data_dir, "Testing")
        if os.path.isdir(potential_testing):
            testing_dir = potential_testing

    if not os.path.exists(testing_dir):
        raise FileNotFoundError(f"Testing directory not found at path: {testing_dir}")

    test_samples: List[Tuple[str, int]] = []
    class_counts: Dict[str, int] = {cls_name: 0 for cls_name in CLASS_NAMES}

    for cls_name in CLASS_NAMES:
        cls_dir = os.path.join(testing_dir, cls_name)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"Class folder '{cls_name}' missing in {testing_dir}")

        cls_idx = CLASS_TO_IDX[cls_name]
        files = [
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if f.lower().endswith(VALID_EXTENSIONS)
        ]
        class_counts[cls_name] = len(files)
        test_samples.extend([(f, cls_idx) for f in files])

    _, val_transform = get_transforms()
    test_dataset = BrainTumorDataset(test_samples, transform=val_transform)

    print("\n--- Test Set Image Distribution ---")
    for cls_name, count in class_counts.items():
        print(f"  {cls_name:12s}: {count:4d} images (Class Index: {CLASS_TO_IDX[cls_name]})")
    print(f"  Total Test Images: {len(test_dataset)}")

    return test_dataset, len(test_samples)


def evaluate():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("      Brain Tumor ResNet50 Model - Testing Evaluation")
    print("=" * 65)
    print(f" Dataset Path    : {args.data_dir}")
    print(f" Checkpoint Path : {args.checkpoint_path}")
    print(f" Evaluation Device: {device}")
    print("=" * 65)

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {args.checkpoint_path}")

    # Load Test Dataset
    test_dataset, total_images = load_test_dataset(args.data_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Rebuild Model & Load Checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model = build_resnet50_model(num_classes=len(CLASS_NAMES), freeze_backbone=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    print("\nRunning evaluation on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    total_eval_samples = len(all_labels)
    test_loss = running_loss / total_eval_samples
    correct_predictions = sum(p == l for p, l in zip(all_preds, all_labels))
    incorrect_predictions = total_eval_samples - correct_predictions
    overall_accuracy = (correct_predictions / total_eval_samples) * 100.0

    # Calculate Confusion Matrix
    num_classes = len(CLASS_NAMES)
    conf_matrix = [[0] * num_classes for _ in range(num_classes)]
    for p, l in zip(all_preds, all_labels):
        conf_matrix[l][p] += 1

    # Per-Class Metrics Calculation (TP, FP, FN, Precision, Recall, F1)
    metrics_per_class = {}
    for i, cls_name in enumerate(CLASS_NAMES):
        tp = conf_matrix[i][i]
        fn = sum(conf_matrix[i][j] for j in range(num_classes) if j != i)
        fp = sum(conf_matrix[j][i] for j in range(num_classes) if j != i)
        total_cls = tp + fn
        
        cls_acc = (tp / total_cls * 100.0) if total_cls > 0 else 0.0
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        metrics_per_class[cls_name] = {
            "total": total_cls,
            "correct": tp,
            "incorrect": fn,
            "accuracy": cls_acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    # Print Summary Report
    print("\n" + "=" * 65)
    print("                    EVALUATION SUMMARY")
    print("=" * 65)
    print(f" Total Test Images Evaluated : {total_eval_samples}")
    print(f" Correct Predictions        : {correct_predictions}")
    print(f" Incorrect Predictions      : {incorrect_predictions}")
    print(f" Overall Test Loss          : {test_loss:.4f}")
    print(f" Overall Test Accuracy      : {overall_accuracy:.2f}%")
    print("=" * 65)

    print("\n--- Per-Class Performance Metrics ---")
    header = f"{'Class':12s} | {'Total':5s} | {'Correct':7s} | {'Accuracy':9s} | {'Precision':9s} | {'Recall':8s} | {'F1-Score':8s}"
    print(header)
    print("-" * len(header))
    for cls_name in CLASS_NAMES:
        m = metrics_per_class[cls_name]
        print(
            f"{cls_name:12s} | {m['total']:5d} | {m['correct']:7d} | "
            f"{m['accuracy']:8.2f}% | {m['precision']:9.4f} | "
            f"{m['recall']:8.4f} | {m['f1_score']:8.4f}"
        )

    print("\n--- Confusion Matrix (Rows: Actual, Columns: Predicted) ---")
    col_header = "Actual \\ Pred | " + " | ".join(f"{c:>10s}" for c in CLASS_NAMES)
    print(col_header)
    print("-" * len(col_header))
    for i, cls_name in enumerate(CLASS_NAMES):
        row_str = f"{cls_name:12s} | " + " | ".join(f"{conf_matrix[i][j]:10d}" for j in range(num_classes))
        print(row_str)

    print("=" * 65)
    print(f"Verification: Exactly 1,600 images evaluated? {'YES' if total_eval_samples == 1600 else 'NO'}")
    print("=" * 65)


if __name__ == "__main__":
    evaluate()
