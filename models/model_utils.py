# This file contains utility functions for model related operations.
# Import necessary libraries
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

# Import necessary functions
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import Callback


# Testing function 
def test_model(model_name, y_true, y_pred):
    """
    Test the model predictions and return multiple evaluation metrics.
    Parameters:
    model_name: The name of the model being evaluated.
    y_true: The true labels for the test dataset.
    y_pred: The predicted labels for the test dataset.

    Returns:
    A dictionary containing the evaluation metrics: RMSE, MAE, R2 Score, and Explained Variance Score.
    """
    if y_pred is None:
        raise ValueError("y_pred must be provided")
    if y_true is None:
        raise ValueError("y_true must be provided")

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    explained_variance = explained_variance_score(y_true, y_pred)

    return {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "R2 Score": r2,
        "Explained Variance Score": explained_variance
    }

# Define a simple feedforward neural network using PyTorch Lightning
class NeuralNetwork(pl.LightningModule):
    def __init__(self, input_sz, hidden_sz=128, lr=1e-3):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_sz, hidden_sz), # There is currently 3 hidden layers, but this can be easily changed by adding more blocks of Linear, ReLU, BatchNorm, and Dropout
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_sz),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(hidden_sz, hidden_sz),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_sz),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(hidden_sz, hidden_sz),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_sz),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(hidden_sz, 1)
        )

        self.lr = lr

    def forward(self, x):
        return self.net(x)

    def _shared_step(self, batch):
        x, y = batch
        preds = self(x)
        loss = F.mse_loss(preds, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
# Define a custom dataset class for PyTorch
class ESOLDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define a PyTorch Lightning DataModule for handling data loading
class NeuralNetworkDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size=256):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
    
# Capture training and validation loss history during training
class TrainingHistoryCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.epochs = []

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            self.train_losses.append(float(train_loss))
        else:
            self.train_losses.append(None)

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        self.epochs.append(trainer.current_epoch)
        if val_loss is not None:
            self.val_losses.append(float(val_loss))
        else:
            self.val_losses.append(None)
