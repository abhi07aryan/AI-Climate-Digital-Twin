import torch
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device
    ):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, loader):

        self.model.train()

        running_loss = 0.0

        for X, y in tqdm(loader):

            X = X.to(self.device)

            y = y.to(self.device)

            self.optimizer.zero_grad()

            prediction = self.model(X)

            loss = self.criterion(
                prediction,
                y
            )

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(loader)
    
    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        running_loss = 0.0
        for X, y in loader:
            X = X.to(self.device)
            y = y.to(self.device)
            prediction = self.model(X)
            loss = self.criterion(prediction, y)
            running_loss += loss.item()
        return running_loss / len(loader)