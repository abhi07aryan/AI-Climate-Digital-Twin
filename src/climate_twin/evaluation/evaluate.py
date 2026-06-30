from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

from climate_twin.preprocessing.split import TimeSeriesSplit
from climate_twin.preprocessing.normalize import ClimateNormalizer
from climate_twin.ml.dataset import ClimateTorchDataset

from climate_twin.models.convlstm import ConvLSTM

from climate_twin.evaluation.metrics import ClimateMetrics
from climate_twin.evaluation.plots import ClimatePlots


# =====================================================
# Configuration
# =====================================================

DATASET = "data/processed/climate_up.nc"

MODEL = "models/convlstm_best.pth"

WINDOW_SIZE = 30

FEATURES = [
        "rainfall",
        "tmax",
        "tmin"]
    #     "temp_mean",
    #     "temp_range",
    #     "rain_7day",
    #     "rain_30day",
    #     "rain_lag1",
    #     "rain_lag3",
    #     "rain_lag7",
    #     "month",
    #     "season",
    #     "dayofyear",
    #     "rain_anomaly"
    # ]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MC_SAMPLES = 30

def mc_predict(model, x, samples=MC_SAMPLES):

    model.train()      # Enable dropout

    predictions = []

    with torch.no_grad():

        for _ in range(samples):

            pred = model(x)

            predictions.append(
                pred.cpu().numpy()
            )

    predictions = np.stack(predictions)

    mean = predictions.mean(axis=0)

    std = predictions.std(axis=0)

    confidence = np.exp(
        -std.mean()
    ) * 100

    return mean, std, confidence

def main():

    print(f"\nUsing device : {DEVICE}\n")

    # =====================================================
    # Dataset
    # =====================================================

    print("Loading dataset...")

    ds = xr.open_dataset(DATASET)

    splitter = TimeSeriesSplit()

    train, valid, test = splitter.split(ds)

    normalizer = ClimateNormalizer()

    normalizer.fit(train, FEATURES)

    test = normalizer.transform(test)

    dataset = ClimateTorchDataset(
        test,
        input_features=FEATURES,
        target="rainfall",
        window_size=WINDOW_SIZE
    )

    print(f"Test samples : {len(dataset)}")

    # =====================================================
    # Model
    # =====================================================

    print("\nLoading model...")

    model = ConvLSTM(
        input_channels=len(FEATURES),
        hidden_channels=8,
        output_channels=1
    )

    checkpoint = torch.load(
        MODEL,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    print(f"Checkpoint loaded (Epoch {checkpoint['epoch']})")

    # =====================================================
    # Evaluation
    # =====================================================

    first_truth = None
    first_prediction = None
    first_uncertainty = None

    all_rmse = []
    all_mae = []
    all_corr = []
    all_confidence = []
    all_uncertainty = []

    first_truth = None
    first_prediction = None

    print("\nRunning evaluation...\n")

    with torch.no_grad():

        for X, y in dataset:

            X = X.unsqueeze(0).to(DEVICE)

            prediction, uncertainty, confidence = mc_predict(
                model,
                X,
                samples=MC_SAMPLES
            )

            prediction = prediction.squeeze()

            uncertainty = uncertainty.squeeze()

            truth = y.squeeze().numpy()

            if first_truth is None:

                first_truth = truth

                first_prediction = prediction

                first_uncertainty = uncertainty

            # -----------------------------------------
            # Compute metrics
            # -----------------------------------------

            difference = prediction - truth

            rmse = np.sqrt(
                np.mean(difference ** 2)
            )

            mae = np.mean(
                np.abs(difference)
            )

            correlation = np.corrcoef(
                truth.flatten(),
                prediction.flatten()
            )[0, 1]

            # -----------------------------------------
            # Store metrics
            # -----------------------------------------

            all_rmse.append(rmse)

            all_mae.append(mae)

            all_corr.append(correlation)

            all_confidence.append(confidence)

            all_uncertainty.append(
                uncertainty.mean()
            )


    print(f"Average RMSE        : {np.mean(all_rmse):.4f}")
    print(f"Average MAE         : {np.mean(all_mae):.4f}")
    print(f"Average Correlation : {np.mean(all_corr):.4f}")
    print(f"Average Confidence  : {np.mean(all_confidence):.2f}%")
    print(f"Average Uncertainty : {np.mean(all_uncertainty):.4f}")

    print("=" * 45)

    # -------------------------------------------------------
    # Save Results
    # -------------------------------------------------------
    from pathlib import Path
    RESULTS = Path("results")
    
    RESULTS.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------------
    # Ground Truth
    # -------------------------------------------------------

    plt.figure(figsize=(6,6))

    plt.imshow(
        first_truth,
        cmap="Blues"
    )

    plt.colorbar()

    plt.axis("off")

    plt.title("Ground Truth")

    plt.savefig(
        RESULTS / "ground_truth.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # -------------------------------------------------------
    # Prediction
    # -------------------------------------------------------

    plt.figure(figsize=(6,6))

    plt.imshow(
        first_prediction,
        cmap="Blues"
    )

    plt.colorbar()

    plt.axis("off")

    plt.title("Prediction")

    plt.savefig(
        RESULTS / "prediction.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # -------------------------------------------------------
    # Difference
    # -------------------------------------------------------

    plt.figure(figsize=(6,6))

    plt.imshow(
        first_prediction - first_truth,
        cmap="RdBu_r"
    )

    plt.colorbar()

    plt.axis("off")

    plt.title("Prediction Error")

    plt.savefig(
        RESULTS / "difference.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # -------------------------------------------------------
    # Uncertainty
    # -------------------------------------------------------

    plt.figure(figsize=(6,6))

    plt.imshow(
        first_uncertainty,
        cmap="inferno"
    )

    plt.colorbar()

    plt.axis("off")

    plt.title("Prediction Uncertainty")

    plt.savefig(
        RESULTS / "uncertainty.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    metrics = {
        "Average RMSE": np.mean(all_rmse),
        "Average MAE": np.mean(all_mae),
        "Average Correlation": np.mean(all_corr),
        "Average Confidence (%)": np.mean(all_confidence),
        "Average Uncertainty": np.mean(all_uncertainty)
    }

    import pandas as pd

    pd.DataFrame(
        metrics.items(),
        columns=["Metric", "Value"]
    ).to_csv(
        RESULTS / "metrics.csv",
        index=False
    )


if __name__ == "__main__":

    main()