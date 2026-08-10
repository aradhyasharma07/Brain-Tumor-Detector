import os
import csv
from collections import Counter

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from services.inference_service import InferenceService


# ==============================
# PATHS
# ==============================

DATASET_PATH = r"C:\Users\aradhya\OneDrive\Desktop\Brain Tumor Dataset"
TESTING_PATH = os.path.join(DATASET_PATH, "Testing")

OUTPUT_CSV = os.path.join(
    os.path.dirname(__file__),
    "test_results.csv"
)


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


# ==============================
# LOAD MODEL
# ==============================

print("=" * 60)
print("BRAIN TUMOR MODEL EVALUATION")
print("=" * 60)

print("\nLoading ResNet50 model...")

model = InferenceService()

print("Model loaded successfully.")
print(f"Testing dataset: {TESTING_PATH}")


# ==============================
# CHECK DATASET
# ==============================

if not os.path.exists(TESTING_PATH):
    print("\nERROR: Testing folder not found.")
    print(TESTING_PATH)
    raise SystemExit(1)


# ==============================
# RUN PREDICTIONS
# ==============================

actual_labels = []
predicted_labels = []

results = []

total_images = 0
failed_images = 0


print("\nStarting evaluation...\n")


for class_name in CLASS_NAMES:

    class_folder = os.path.join(
        TESTING_PATH,
        class_name
    )

    if not os.path.exists(class_folder):
        print(f"WARNING: Missing folder: {class_folder}")
        continue

    image_files = [
        file_name
        for file_name in os.listdir(class_folder)
        if file_name.lower().endswith(VALID_EXTENSIONS)
    ]

    print(
        f"Testing {class_name}: "
        f"{len(image_files)} images"
    )

    for file_name in image_files:

        image_path = os.path.join(
            class_folder,
            file_name
        )

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

            print(
                f"ERROR: {file_name} -> {error}"
            )


# ==============================
# CALCULATE METRICS
# ==============================

accuracy = accuracy_score(
    actual_labels,
    predicted_labels
)


print("\n")
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print(f"\nTotal images tested : {total_images}")
print(f"Failed images       : {failed_images}")

print(
    f"\nAccuracy            : "
    f"{accuracy * 100:.2f}%"
)


# ==============================
# CLASSIFICATION REPORT
# ==============================

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        actual_labels,
        predicted_labels,
        labels=CLASS_NAMES,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)


# ==============================
# CONFUSION MATRIX
# ==============================

matrix = confusion_matrix(
    actual_labels,
    predicted_labels,
    labels=CLASS_NAMES
)


print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(
    f"{'':15}"
    + "".join(
        f"{name[:12]:>14}"
        for name in CLASS_NAMES
    )
)

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name:15}"
        + "".join(
            f"{matrix[i][j]:>14}"
            for j in range(len(CLASS_NAMES))
        )
    )


# ==============================
# SAVE CSV
# ==============================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image",
            "actual",
            "predicted",
            "confidence"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


# ==============================
# CLASS COUNTS
# ==============================

print("\n")
print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print("\nActual images:")

actual_counts = Counter(actual_labels)

for class_name in CLASS_NAMES:
    print(
        f"{class_name:12}: "
        f"{actual_counts[class_name]}"
    )


print("\nPredicted images:")

predicted_counts = Counter(predicted_labels)

for class_name in CLASS_NAMES:
    print(
        f"{class_name:12}: "
        f"{predicted_counts[class_name]}"
    )


print("\n")
print(f"Detailed results saved to:")
print(OUTPUT_CSV)

print("\n" + "=" * 60)
print("TESTING FINISHED")
print("=" * 60)