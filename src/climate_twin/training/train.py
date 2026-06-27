import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from climate_twin.preprocessing.split import TimeSeriesSplit
from climate_twin.preprocessing.normalize import ClimateNormalizer
from climate_twin.ml.dataset import ClimateTorchDataset
from climate_twin.models.convlstm import ConvLSTM
from climate_twin.training.trainer import Trainer

print("Loading climate dataset...")

climate = xr.open_dataset("data/processed/climate_up.nc")

print("Splitting dataset...")

splitter = TimeSeriesSplit()

train, valid, test = splitter.split(climate)

print("Normalizing...")

variables = [
    "rainfall",
    "tmax",
    "tmin",
    "temp_mean",
    "temp_range",
    "rain_7day",
    "rain_30day"
]

normalizer = ClimateNormalizer()
normalizer.fit(train, variables)
train = normalizer.transform(train)
valid = normalizer.transform(valid)
test = normalizer.transform(test)

features = [
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

train_dataset = ClimateTorchDataset(
    train,
    input_features=features,
    target="rainfall")

valid_dataset = ClimateTorchDataset(
    valid,
    input_features=features,
    target="rainfall")

test_dataset = ClimateTorchDataset(
    test,
    input_features=features,
    target="rainfall")

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=4,
    shuffle=False)

test_loader = DataLoader(test_dataset, 
                         batch_size=4, 
                         shuffle=False)

device = torch.device("cuda" 
                      if torch.cuda.is_available()
                        else "cpu")
print(device)

model = ConvLSTM(input_channels=len(features),
                  hidden_channels=32,
                  output_channels=1)
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
criterion = nn.MSELoss()
trainer = Trainer(model, optimizer, criterion, device)

EPOCHS = 20

for epoch in range(EPOCHS):
    train_loss = trainer.train_epoch(train_loader)
    valid_loss = trainer.validate(valid_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss : {train_loss:.4f}")
    print(f"Valid Loss : {valid_loss:.4f}")

torch.save(model.state_dict(), "models/convlstm_up.pth")

print("Model saved.")