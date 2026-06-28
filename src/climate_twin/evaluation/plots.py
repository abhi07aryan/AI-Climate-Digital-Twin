import numpy as np
import matplotlib.pyplot as plt
import os

class ClimatePlots:

    @staticmethod
    def prediction(
        truth,
        prediction,
        title="Prediction",
        save_path = None
    ):

        vmin = min(
            np.nanmin(truth),
            np.nanmin(prediction)
        )

        vmax = max(
            np.nanmax(truth),
            np.nanmax(prediction)
        )

        fig, ax = plt.subplots(
            1,
            2,
            figsize=(12,5)
        )

        im1 = ax[0].imshow(
            truth,
            cmap="Blues",
            vmin=vmin,
            vmax=vmax
        )

        ax[0].set_title("Ground Truth")
        ax[0].axis("off")

        im2 = ax[1].imshow(
            prediction,
            cmap="Blues",
            vmin=vmin,
            vmax=vmax
        )

        ax[1].set_title(title)
        ax[1].axis("off")

        fig.colorbar(
            im2,
            ax=ax,
            shrink=0.75
        )

        plt.tight_layout()

        if save_path:
            from pathlib import Path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")

        plt.close()
        

    @staticmethod
    def error(
        truth,
        prediction,
        save_path = None
    ):

        error = prediction - truth

        plt.figure(figsize=(6,6))

        plt.imshow(
            error,
            cmap="RdBu_r"
        )

        plt.colorbar()

        plt.title("Prediction Error")

        plt.axis("off")

        plt.tight_layout()

        if save_path:
            from pathlib import Path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")

        plt.close()

    @staticmethod
    def histogram(
        truth,
        prediction,
        save_path=None
        ):

        error = prediction - truth

        plt.figure(figsize=(7,4))

        plt.hist(
            error.flatten(),
            bins=100
        )

        plt.title("Prediction Error Distribution")

        plt.xlabel("Prediction - Truth")

        plt.ylabel("Pixels")

        plt.tight_layout()

        if save_path:

            from pathlib import Path

            Path(save_path).parent.mkdir(
                parents=True,
                exist_ok=True
            )

            plt.savefig(
                save_path,
                dpi=200,
                bbox_inches="tight"
            )

        plt.close()

    @staticmethod
    def training(
        train_loss,
        valid_loss
    ):

        epochs = range(
            1,
            len(train_loss)+1
        )

        plt.figure(figsize=(7,5))

        plt.plot(
            epochs,
            train_loss,
            label="Train"
        )

        plt.plot(
            epochs,
            valid_loss,
            label="Validation"
        )

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.title("Training Curve")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.show()

    @staticmethod
    def time_series(
        truth,
        prediction
    ):

        plt.figure(figsize=(10,4))

        plt.plot(
            truth,
            label="Ground Truth"
        )

        plt.plot(
            prediction,
            label="Prediction"
        )

        plt.legend()

        plt.xlabel("Time")

        plt.ylabel("Rainfall")

        plt.tight_layout()

        plt.show()