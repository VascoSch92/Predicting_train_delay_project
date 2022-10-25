"""
Created on Monday Sep. 12-2022.

@author: Vasco Schiavo

How many times before heading to the station have we asked ourselves: but will my train be on time?
To answer this question, we developed a model using neural networks that consider the train station and
departure time of a train and predict its punctuality. This script contains the model to answer this question.
"""

# Packages and modules
import numpy as np
import pandas as pd

import os
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler


def preprocessing_data(data, list_columns):
    """
    The method preprocesses the data before the model training.
    :param data: DataFrame
    :param list_columns: list of selected columns in data
    :return data: np.array
    """

    # Scaler for preprocessing
    scaler = MaxAbsScaler()

    # Preprocessing selected columns in the DataFrame
    for column in list_columns:
        data[column] = scaler.fit_transform(data[column].values.reshape(-1, 1))

    return data


class train_delay_model(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.layer_1 = torch.nn.Linear(4, 10)
        self.layer_2 = torch.nn.Linear(10, 10)
        self.layer_3 = torch.nn.Linear(10, 10)
        self.layer_4 = torch.nn.Linear(10, 1)

    def forward(self, x):
        batch_size, input_dim = x.size()

        # (b, 4)
        x = x.view(batch_size, input_dim)

        # layer 1 (b, 4) -> (b, 10)
        x = self.layer_1(x)
        x = torch.relu(x)

        # layer 2 (b, 10) -> (b, 10)
        x = self.layer_2(x)
        x = torch.relu(x)

        # layer 3 (b, 10) -> (b, 10)
        x = self.layer_3(x)
        x = torch.relu(x)

        # layer 4 (b, 1) -> (b, 1)
        x = self.layer_4(x)

        return x

    def mean_absolute_error(self, logits, target):
        loss = torch.nn.L1Loss()
        return loss(logits, target)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y = y.unsqueeze(1)
        logits = self.forward(x)
        loss = self.mean_absolute_error(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y = y.unsqueeze(1)
        logits = self.forward(x)
        loss = self.mean_absolute_error(logits, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class CSV_Dataset(Dataset):

    def __init__(self, file_name):
        df = pd.read_csv(file_name)
        x = df.iloc[:, 1:5].values
        y = df.iloc[:, 5].values

        scaler = MaxAbsScaler()
        x = scaler.fit_transform(x)

        self.x = torch.tensor(x, device=device, dtype=torch.float32)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    # get indexes for train and test rows
    def get_splits(self, split=0.33):
        # determine sizes
        test_size = round(split * len(self.x))
        train_size = len(self.x) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

########################
#   MAIN
########################

device = torch.device("mps")

# Imports the data
data = CSV_Dataset(f"/Users/argo/PycharmProjects/Train_Delay_Predicion/cleaned_data/train_delay_data_17.10.2022.csv")


# Parameters
test_size = 0.2
batch_size = 64
seed = 42

# Splits the data into train set and validation set
train_set, val_set = data.get_splits()

# prepare data loaders
train_dl = DataLoader(train_set, shuffle=True)
val_dl = DataLoader(val_set, shuffle=False)

# Imports model and train
model = train_delay_model()

training = pl.Trainer(max_epochs=5, accelerator="cpu", devices=1)

training.fit(model, train_dl, val_dl)
