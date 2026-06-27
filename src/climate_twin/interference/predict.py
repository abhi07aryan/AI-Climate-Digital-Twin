import xarray as xr
import torch
import matplotlib.pyplot as plt
import numpy as np

from climate_twin.preprocessing.split import TimeSeriesSplit
from climate_twin.preprocessing.normalize import ClimateNormalizer
from climate_twin.ml.dataset import ClimateTorchDataset
from climate_twin.models.convlstm import ConvLSTM


def main():

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    DATASET = "data/processed/climate_up.nc"
    MODEL = "models/convlstm_up_best.pth"

    WINDOW_SIZE = 30

    FEATURES = [
        "rainfall",
        "tmax",
        "tmin",
        "temp_mean",
        "temp_range",
        "rain_7day",
        "rain_30day",
        "rain_lag1",
        "rain_lag3",
        "rain_lag7",
        "month",
        "season",
        "dayofyear",
        "rain_anomaly"
    ]

    NORMALIZE = [
        "rainfall",
        "tmax",
        "tmin",
        "temp_mean",
        "temp_range",
        "rain_7day",
        "rain_30day",
        "rain_lag1",
        "rain_lag3",
        "rain_lag7",
        "rain_anomaly"
    ]

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # --------------------------------------------------------
    # Load Dataset
    # --------------------------------------------------------

    print("Loading dataset...")

    climate = xr.open_dataset(DATASET)

    splitter = TimeSeriesSplit()

    train, valid, test = splitter.split(climate)

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    normalizer = ClimateNormalizer()

    normalizer.fit(train, NORMALIZE)

    test = normalizer.transform(test)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = ClimateTorchDataset(
        test,
        input_features=FEATURES,
        target="rainfall",
        window_size=WINDOW_SIZE
    )

    print(f"Test samples: {len(test_dataset)}")

    # --------------------------------------------------------
    # Load Model
    # --------------------------------------------------------

    model = ConvLSTM(
        input_channels=len(FEATURES),
        hidden_channels=32,
        output_channels=1
    )

    model.load_state_dict(
        torch.load(
            MODEL,
            map_location=device
        )
    )

    model.to(device)

    model.eval()

    # --------------------------------------------------------
    # Predict Sample
    # --------------------------------------------------------

    sample = 0

    X, y = test_dataset[sample]

    X = X.unsqueeze(0).to(device)

    with torch.no_grad():

        prediction = model(X)

    prediction = prediction.squeeze().cpu().numpy()

    truth = y.squeeze().cpu().numpy()

    error = prediction - truth

    plt.figure(figsize=(6,4))

    plt.hist(error.flatten(), bins=100)

    plt.title("Prediction Error Distribution")

    plt.xlabel("Prediction - Ground Truth")

    plt.ylabel("Pixels")

    plt.show()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\nGround Truth")

    print("Min :", np.min(truth))
    print("Max :", np.max(truth))
    print("Mean:", np.mean(truth))

    print("\nPrediction")

    print("Min :", np.min(prediction))
    print("Max :", np.max(prediction))
    print("Mean:", np.mean(prediction))

    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(np.abs(error))

    print(f"\nRMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    vmin = min(truth.min(), prediction.min())
    vmax = max(truth.max(), prediction.max())

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    im0 = ax[0].imshow(
        truth,
        cmap="Blues",
        vmin=vmin,
        vmax=vmax
    )

    ax[0].set_title("Ground Truth")
    ax[0].axis("off")

    im1 = ax[1].imshow(
        prediction,
        cmap="Blues",
        vmin=vmin,
        vmax=vmax
    )

    ax[1].set_title("Prediction")
    ax[1].axis("off")

    im2 = ax[2].imshow(
        error,
        cmap="RdBu_r"
    )

    ax[2].set_title("Prediction Error")
    ax[2].axis("off")

    fig.colorbar(im0, ax=ax[:2], shrink=0.8)
    fig.colorbar(im2, ax=ax[2], shrink=0.8)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
