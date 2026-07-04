from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DATASET_FILE = Path(
    "data/processed/medium_dataset.npz"
)

MODEL_FILE = Path(
    "models/cnn_classifier.pt"
)

METADATA_FILE = Path(
    "models/cnn_classifier_metadata.joblib"
)

CONFUSION_MATRIX_FILE = Path(
    "outputs/cnn_confusion_matrix.png"
)

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

RANDOM_STATE = 42
BATCH_SIZE = 16
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
PATIENCE = 12


class SpectrumCNN(
    nn.Module
):
    """A small 1D CNN for classifying astronomical spectra."""

    def __init__(
        self,
        number_of_classes: int,
    ) -> None:
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=9,
                padding=4,
            ),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=2,
            ),

            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=7,
                padding=3,
            ),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=2,
            ),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=2,
            ),

            nn.AdaptiveAvgPool1d(
                output_size=1,
            ),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(
                p=0.30,
            ),
            nn.Linear(
                in_features=64,
                out_features=number_of_classes,
            ),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        features = self.feature_extractor(
            inputs
        )

        logits = self.classifier(
            features
        )

        return logits


def set_random_seeds() -> None:
    """Make results more reproducible."""

    np.random.seed(
        RANDOM_STATE
    )

    torch.manual_seed(
        RANDOM_STATE
    )


def make_tensor_dataset(
    features: np.ndarray,
    targets: np.ndarray,
) -> TensorDataset:
    """Convert NumPy arrays into a PyTorch dataset."""

    feature_tensor = torch.tensor(
        features,
        dtype=torch.float32,
    )

    # Conv1d expects shape:
    # batch, channels, length
    feature_tensor = feature_tensor.unsqueeze(
        1
    )

    target_tensor = torch.tensor(
        targets,
        dtype=torch.long,
    )

    return TensorDataset(
        feature_tensor,
        target_tensor,
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train the model for one epoch."""

    model.train()

    total_loss = 0.0

    for features, targets in dataloader:
        features = features.to(
            device
        )

        targets = targets.to(
            device
        )

        optimizer.zero_grad()

        logits = model(
            features
        )

        loss = loss_function(
            logits,
            targets,
        )

        loss.backward()

        optimizer.step()

        total_loss += float(
            loss.item()
        ) * len(targets)

    average_loss = total_loss / len(
        dataloader.dataset
    )

    return average_loss


def evaluate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    """Calculate validation or test loss."""

    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(
                device
            )

            targets = targets.to(
                device
            )

            logits = model(
                features
            )

            loss = loss_function(
                logits,
                targets,
            )

            total_loss += float(
                loss.item()
            ) * len(targets)

    average_loss = total_loss / len(
        dataloader.dataset
    )

    return average_loss


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted classes and confidence scores."""

    model.eval()

    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for features, _ in dataloader:
            features = features.to(
                device
            )

            logits = model(
                features
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predictions = torch.argmax(
                probabilities,
                dim=1,
            )

            all_predictions.append(
                predictions.cpu().numpy()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

    return (
        np.concatenate(all_predictions),
        np.concatenate(all_probabilities),
    )


def main() -> None:
    """Train and evaluate a 1D CNN spectrum classifier."""

    set_random_seeds()

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}"
        )

    dataset = np.load(
        DATASET_FILE
    )

    features = dataset["X"]
    targets = dataset["y"]
    class_names = dataset["class_names"]

    print("Dataset")
    print("-------")
    print(f"Feature shape: {features.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Classes: {class_names}")

    (
        train_validation_features,
        test_features,
        train_validation_targets,
        test_targets,
    ) = train_test_split(
        features,
        targets,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=targets,
    )

    (
        train_features,
        validation_features,
        train_targets,
        validation_targets,
    ) = train_test_split(
        train_validation_features,
        train_validation_targets,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_validation_targets,
    )

    print("\nDataset split")
    print("-------------")
    print(f"Training spectra: {len(train_targets)}")
    print(f"Validation spectra: {len(validation_targets)}")
    print(f"Testing spectra: {len(test_targets)}")

    train_dataset = make_tensor_dataset(
        train_features,
        train_targets,
    )

    validation_dataset = make_tensor_dataset(
        validation_features,
        validation_targets,
    )

    test_dataset = make_tensor_dataset(
        test_features,
        test_targets,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nUsing device: {device}")

    model = SpectrumCNN(
        number_of_classes=len(class_names)
    ).to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_loss = float(
        "inf"
    )

    best_model_state = None
    epochs_without_improvement = 0

    print("\nTraining CNN...")
    print("---------------")

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
        )

        validation_loss = evaluate_loss(
            model=model,
            dataloader=validation_loader,
            loss_function=loss_function,
            device=device,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_loss:.4f} | "
            f"validation loss={validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            best_model_state = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(
                "Early stopping: validation loss "
                "stopped improving."
            )
            break

    if best_model_state is not None:
        model.load_state_dict(
            best_model_state
        )

    test_loss = evaluate_loss(
        model=model,
        dataloader=test_loader,
        loss_function=loss_function,
        device=device,
    )

    predictions, probabilities = predict(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    accuracy = accuracy_score(
        test_targets,
        predictions,
    )

    print("\nFinal untouched-test evaluation")
    print("-------------------------------")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {accuracy:.3f}")

    print(
        classification_report(
            test_targets,
            predictions,
            target_names=class_names,
            zero_division=0,
        )
    )

    incorrect_positions = np.flatnonzero(
        predictions != test_targets
    )

    print("Incorrect predictions")
    print("---------------------")

    if len(incorrect_positions) == 0:
        print("No incorrect predictions.")
    else:
        for position in incorrect_positions:
            actual_index = int(
                test_targets[position]
            )

            predicted_index = int(
                predictions[position]
            )

            actual_class = str(
                class_names[actual_index]
            )

            predicted_class = str(
                class_names[predicted_index]
            )

            confidence = float(
                probabilities[position].max()
            )

            print(
                f"Actual: {actual_class} | "
                f"Predicted: {predicted_class} | "
                f"Confidence: {confidence:.3f}"
            )

    matrix = confusion_matrix(
        test_targets,
        predictions,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    display.plot(
        values_format="d",
    )

    plt.title(
        "CNN Confusion Matrix"
    )

    plt.tight_layout()

    CONFUSION_MATRIX_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=150,
    )

    plt.close()

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "test_accuracy": accuracy,
            "test_loss": test_loss,
            "random_state": RANDOM_STATE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
        },
        MODEL_FILE,
    )

    joblib.dump(
        {
            "class_names": class_names,
            "test_accuracy": accuracy,
            "test_loss": test_loss,
            "random_state": RANDOM_STATE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "model_architecture": "SpectrumCNN",
        },
        METADATA_FILE,
    )

    print(f"\nSaved PyTorch model to: {MODEL_FILE}")
    print(f"Saved metadata to: {METADATA_FILE}")
    print(
        "Saved confusion matrix to: "
        f"{CONFUSION_MATRIX_FILE}"
    )


if __name__ == "__main__":
    main()