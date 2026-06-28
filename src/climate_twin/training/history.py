from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


class TrainingHistory:

    def __init__(self):

        self.history = {
            "epoch": [],
            "train_loss": [],
            "valid_loss": [],
            "learning_rate": []
        }

    def update(
        self,
        epoch,
        train_loss,
        valid_loss,
        learning_rate
    ):

        self.history["epoch"].append(epoch)

        self.history["train_loss"].append(train_loss)

        self.history["valid_loss"].append(valid_loss)

        self.history["learning_rate"].append(learning_rate)

    def save_csv(
        self,
        filepath="results/history.csv"
    ):

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df = pd.DataFrame(self.history)

        df.to_csv(
            filepath,
            index=False
        )

        print(f"History saved to {filepath}")

    def plot(
        self,
        filepath="results/training_curve.png"
    ):

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.figure(figsize=(8,5))

        plt.plot(
            self.history["epoch"],
            self.history["train_loss"],
            label="Train"
        )

        plt.plot(
            self.history["epoch"],
            self.history["valid_loss"],
            label="Validation"
        )

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.title("Training Curve")

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.savefig(filepath)

        plt.close()

        print(f"Training curve saved to {filepath}")