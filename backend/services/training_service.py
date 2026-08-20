import os
import time
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from backend.models.resnet_model import build_resnet50_model, get_parameter_counts
from backend.utils.dataset_utils import (
    create_dataloaders,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    CLASS_NAMES
)


class TrainingService:
    """
    Service to manage model training, metric tracking, and checkpoint saving.
    """
    def __init__(
        self,
        data_dir: str,
        save_dir: str = "backend/saved_models",
        num_epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 0.0001,
        seed: int = 42
    ):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed

        # Device selection (GPU if available, CPU otherwise)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_training(self) -> Dict[str, Any]:
        """
        Executes the transfer-learning training loop for the ResNet50 model.

        Returns:
            Dict[str, Any]: Dictionary containing training history and best metrics.
        """
        # Set seeds for reproducibility
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        os.makedirs(self.save_dir, exist_ok=True)

        print(f"Using device: {self.device}")
        print("Loading dataset and creating 80/20 train/val dataloaders...")
        
        train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            val_split=0.2,
            seed=self.seed
        )

        print(f"Train samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

        # Build ResNet50 model with frozen backbone
        model = build_resnet50_model(num_classes=len(CLASS_NAMES), freeze_backbone=True)
        model.to(self.device)

        total_params, trainable_params = get_parameter_counts(model)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")

        criterion = nn.CrossEntropyLoss()
        
        # Optimize ONLY parameters with requires_grad=True (includes layer4 and classification head)
        trainable_parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = Adam(trainable_parameters, lr=self.learning_rate)

        # Learning Rate Scheduler: reduces LR by 0.5 when val_acc plateaus for 3 epochs
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3
        )

        best_val_acc = 0.0
        best_val_loss = float("inf")
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        print("\nStarting Training...")
        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            # --- Training Phase ---
            model.train()
            running_train_loss = 0.0
            running_train_corrects = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_train_loss += loss.item() * inputs.size(0)
                running_train_corrects += torch.sum(preds == labels.data).item()

            epoch_train_loss = running_train_loss / len(train_dataset)
            epoch_train_acc = running_train_corrects / len(train_dataset)

            # --- Validation Phase ---
            model.eval()
            running_val_loss = 0.0
            running_val_corrects = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    running_val_loss += loss.item() * inputs.size(0)
                    running_val_corrects += torch.sum(preds == labels.data).item()

            epoch_val_loss = running_val_loss / len(val_dataset)
            epoch_val_acc = running_val_corrects / len(val_dataset)

            history["train_loss"].append(epoch_train_loss)
            history["train_acc"].append(epoch_train_acc)
            history["val_loss"].append(epoch_val_loss)
            history["val_acc"].append(epoch_val_acc)

            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch}/{self.num_epochs}] | "
                f"LR: {current_lr:.6f} | "
                f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.4f}"
            )

            # Update learning rate based on validation accuracy
            scheduler.step(epoch_val_acc)

            # Save best checkpoint based on validation accuracy (or lower val loss as tiebreaker)
            if epoch_val_acc > best_val_acc or (epoch_val_acc == best_val_acc and epoch_val_loss < best_val_loss):
                best_val_acc = epoch_val_acc
                best_val_loss = epoch_val_loss
                
                checkpoint_path = os.path.join(self.save_dir, "resnet50_brain_tumor.pth")
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": epoch_val_acc,
                    "val_loss": epoch_val_loss,
                    "train_acc": epoch_train_acc,
                    "train_loss": epoch_train_loss,
                    "class_to_idx": CLASS_TO_IDX,
                    "idx_to_class": IDX_TO_CLASS,
                    "class_names": CLASS_NAMES,
                    "num_classes": len(CLASS_NAMES),
                    "architecture": "resnet50"
                }
                torch.save(checkpoint, checkpoint_path)
                print(f" -> Best model saved to {checkpoint_path} (Val Acc: {best_val_acc:.4f})")

        total_time = time.time() - start_time
        print(f"\nTraining Complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")

        return {
            "history": history,
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "saved_model_path": os.path.join(self.save_dir, "resnet50_brain_tumor.pth")
        }
