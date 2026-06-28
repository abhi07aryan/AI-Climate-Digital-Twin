import xarray as xr
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from climate_twin.preprocessing.split import TimeSeriesSplit
from climate_twin.preprocessing.normalize import ClimateNormalizer

from climate_twin.ml.dataset import ClimateTorchDataset

from climate_twin.models.convlstm import ConvLSTM
from climate_twin.models.losses import MaskedMSELoss

from climate_twin.training.callbacks import EarlyStopping
from climate_twin.training.checkpoint import ModelCheckpoint
from climate_twin.training.history import TrainingHistory
from climate_twin.training.scheduler import build_scheduler
from climate_twin.training.trainer import Trainer


def main():

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    DATASET = "data/processed/climate_up.nc"

    WINDOW_SIZE = 7           # 7/30
    BATCH_SIZE = 4            # 8 if RAM allows
    HIDDEN_CHANNELS = 8       # 8/32
    EPOCHS = 15               # 2/30
    LEARNING_RATE = 1e-3

    # FEATURES = [
    #     "rainfall",
    #     "tmax",
    #     "tmin",
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

    FEATURES = [
        "rainfall",
        "tmax",
        "tmin"
    ]

    # NORMALIZE = [
    #     "rainfall",
    #     "tmax",
    #     "tmin",
    #     "temp_mean",
    #     "temp_range",
    #     "rain_7day",
    #     "rain_30day",
    #     "rain_lag1",
    #     "rain_lag3",
    #     "rain_lag7",
    #     "rain_anomaly"
    # ]

    NORMALIZE = [
        "rainfall",
        "tmax",
        "tmin",
    ]

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nUsing device : {device}\n")

    # --------------------------------------------------------
    # Load Dataset
    # --------------------------------------------------------

    print("Loading dataset...")

    climate = xr.open_dataset(DATASET)

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    splitter = TimeSeriesSplit()

    train, valid, test = splitter.split(climate)

    # --------------------------------------------------------
    # Development Mode
    # --------------------------------------------------------

    train = train.isel(time=slice(0,365))
    valid = valid.isel(time=slice(0,100))
    test  = test.isel(time=slice(0,100))

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    normalizer = ClimateNormalizer()

    normalizer.fit(
        train,
        NORMALIZE
    )

    train = normalizer.transform(train)
    valid = normalizer.transform(valid)
    test = normalizer.transform(test)

    # --------------------------------------------------------
    # PyTorch Datasets
    # --------------------------------------------------------

    train_dataset = ClimateTorchDataset(
        train,
        input_features=FEATURES,
        target="rainfall",
        window_size=WINDOW_SIZE
    )

    valid_dataset = ClimateTorchDataset(
        valid,
        input_features=FEATURES,
        target="rainfall",
        window_size=WINDOW_SIZE
    )

    test_dataset = ClimateTorchDataset(
        test,
        input_features=FEATURES,
        target="rainfall",
        window_size=WINDOW_SIZE
    )

    print(f"Train samples : {len(train_dataset)}")
    print(f"Valid samples : {len(valid_dataset)}")
    print(f"Test samples  : {len(test_dataset)}")

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = ConvLSTM(
        input_channels=len(FEATURES),
        hidden_channels=HIDDEN_CHANNELS,
        output_channels=1
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = MaskedMSELoss()

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(
        model,
        optimizer,
        criterion,
        device
    )

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------

    best_loss = float("inf")

    scheduler =  build_scheduler(optimizer)

    checkpoint = ModelCheckpoint("models/convlstm_best.pth")

    start_epoch, best_loss = checkpoint.load(
        model,
        optimizer,
        scheduler,
        device
    )
    early_stopping = EarlyStopping(patience=5, min_delta=1e-4)
    
    history = TrainingHistory()
    # Check dataset shapes
    X, y = train_dataset[0]

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Check model output shape
    model.eval()

    with torch.no_grad():
        pred = model(X.unsqueeze(0).to(device))

    print("Prediction shape:", pred.shape)
    for epoch in range(start_epoch, EPOCHS):

        train_loss = trainer.train_epoch(train_loader)
        valid_loss = trainer.validate(valid_loader)
        scheduler.step(valid_loss)
        print(f"Epoch {epoch+1:02d}/{EPOCHS}")
        print(f"Train Loss : {train_loss:.6f}")
        print(f"Valid Loss : {valid_loss:.6f}")

        checkpoint.save(
            model,
            optimizer,
            scheduler,
            epoch,
            valid_loss
        )

        if early_stopping(valid_loss):
            print("Early stopping.")
            break

        # Save best model

        if valid_loss < best_loss:

            best_loss = valid_loss

            torch.save(model.state_dict(), "models/convlstm_up_best.pth")

            print("Best model saved.\n")

        current_lr = optimizer.param_groups[0]["lr"]
        history.update(
            epoch + 1,
            train_loss,
            valid_loss,
            current_lr
        )
        print(f"Learning Rate : {current_lr:.6f}")
    history.save_csv()
    history.plot()


if __name__ == "__main__":
    main()