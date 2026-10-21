import argparse
import sys
import os

# Add backend directory to sys.path if needed when running direct script
sys.path.insert(0, os.path.dirname(__file__))

from services.training_service import TrainingService


def parse_args():
    parser = argparse.ArgumentParser(
        description="Brain Tumor MRI ResNet50 Transfer Learning Training Pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to dataset root folder containing Training/ and Testing/ directories."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="backend/saved_models",
        help="Directory to save trained model checkpoints (default: backend/saved_models)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 5)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and validation DataLoaders (default: 32)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Initial learning rate for Adam optimizer (default: 0.001)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset split and PyTorch initialization (default: 42)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("      Brain Tumor MRI - ResNet50 Transfer Learning Training")
    print("=" * 60)
    print(f" Dataset Path : {args.data_dir}")
    print(f" Save Path    : {args.save_dir}")
    print(f" Epochs       : {args.epochs}")
    print(f" Batch Size   : {args.batch_size}")
    print(f" Learning Rate: {args.lr}")
    print(f" Seed         : {args.seed}")
    print("=" * 60)

    service = TrainingService(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed
    )

    results = service.run_training()
    print("\nTraining summary:")
    print(f" Best Val Accuracy : {results['best_val_acc'] * 100:.2f}%")
    print(f" Model Saved To    : {results['saved_model_path']}")


if __name__ == "__main__":
    main()
