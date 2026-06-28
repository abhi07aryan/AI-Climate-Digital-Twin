from pathlib import Path

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

WINDOW_SIZE = 7

FEATURES = [
    "rainfall",
    "tmax",
    "tmin"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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
        input_channels=3,
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

    rmse_scores = []
    mae_scores = []
    r2_scores = []
    corr_scores = []

    first_truth = None
    first_prediction = None

    print("\nRunning evaluation...\n")

    with torch.no_grad():

        for X, y in dataset:

            X = X.unsqueeze(0).to(DEVICE)

            pred = model(X)

            prediction = pred.squeeze().cpu().numpy()

            truth = y.squeeze().numpy()

            if first_truth is None:

                first_truth = truth

                first_prediction = prediction

            rmse_scores.append(
                ClimateMetrics.rmse(
                    truth,
                    prediction
                )
            )

            mae_scores.append(
                ClimateMetrics.mae(
                    truth,
                    prediction
                )
            )

            r2_scores.append(
                ClimateMetrics.r2(
                    truth,
                    prediction
                )
            )

            corr_scores.append(
                ClimateMetrics.pearson(
                    truth,
                    prediction
                )
            )

    # =====================================================
    # Average Metrics
    # =====================================================

    avg_rmse = np.mean(rmse_scores)
    avg_mae = np.mean(mae_scores)
    avg_r2 = np.mean(r2_scores)
    avg_corr = np.mean(corr_scores)

    print("=" * 45)
    print("Evaluation Results")
    print("=" * 45)

    print(f"Average RMSE : {avg_rmse:.4f}")
    print(f"Average MAE  : {avg_mae:.4f}")
    print(f"Average R^2   : {avg_r2:.4f}")
    print(f"Correlation  : {avg_corr:.4f}")

    print("=" * 45)

    # =====================================================
    # Save Results
    # =====================================================

    output_dir = Path("results/evaluation")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics = pd.DataFrame({
        "RMSE": [avg_rmse],
        "MAE": [avg_mae],
        "R2": [avg_r2],
        "Correlation": [avg_corr]
    })

    metrics.to_csv(
        output_dir / "metrics.csv",
        index=False
    )

    ClimatePlots.prediction(
        first_truth,
        first_prediction,
        save_path=output_dir / "prediction.png"
    )

    ClimatePlots.error(
        first_truth,
        first_prediction,
        save_path=output_dir / "error.png"
    )

    ClimatePlots.histogram(
        first_truth,
        first_prediction,
        save_path=output_dir / "histogram.png"
    )

    print("\nResults saved successfully!")

    print(output_dir.resolve())


if __name__ == "__main__":

    main()