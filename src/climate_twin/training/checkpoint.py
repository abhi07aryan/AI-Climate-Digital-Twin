from pathlib import Path
import torch


class ModelCheckpoint:
    """
    Saves the best model during training and
    supports loading checkpoints to resume training.
    """

    def __init__(
        self,
        filepath="models/checkpoint.pth"
    ):

        self.filepath = Path(filepath)

        self.filepath.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.best_loss = float("inf")

    def save(
        self,
        model,
        optimizer,
        scheduler,
        epoch,
        val_loss
    ):

        if val_loss < self.best_loss:

            self.best_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict":
                        scheduler.state_dict()
                        if scheduler is not None
                        else None,
                    "best_loss": self.best_loss,
                },
                self.filepath,
            )

            print(
                f"Checkpoint saved "
                f"(Validation Loss = {val_loss:.6f})"
            )

    def load(
        self,
        model,
        optimizer=None,
        scheduler=None,
        device="cpu",
    ):

        if not self.filepath.exists():

            print("No checkpoint found.")

            return 0, float("inf")

        checkpoint = torch.load(
            self.filepath,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if optimizer is not None:

            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        if (
            scheduler is not None
            and checkpoint["scheduler_state_dict"] is not None
        ):

            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        self.best_loss = checkpoint["best_loss"]

        print(
            f"Checkpoint loaded "
            f"(Epoch {checkpoint['epoch']})"
        )

        return (
            checkpoint["epoch"] + 1,
            checkpoint["best_loss"],
        )