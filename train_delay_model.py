"""
Created on Monday Sep. 12-2022.

@author: Vasco Schiavo

How many times before heading to the station have we asked ourselves: but will my train be on time?
To answer this question, we developed a model using neural networks that consider the train station and
departure time of a train and predict its punctuality. This script contains the model to answer this question.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

wandb_logger = WandbLogger(project="my-test-project")

from sklearn.preprocessing import LabelEncoder

class TabularDataset(Dataset):
    
    def __init__(self, data, cat_cols=None, output_col=None):
        """
        Characterizes a Dataset for PyTorch
        
        Parameters
        ----------
        data: pandas dataframe
            It contains all the continuous, categorical and output columns to be used.
        cat_cols: list of strings
            Names of the categorical columns in the dataframe. These columns will be passed
            trought the embedding layers in the model. They must be label encoded beforhand. 
        output_col: string
            The name of the output varaible column in the data provided
        """
        # Number of samples
        self.n = data.shape[0]
        
        # Output part of the dataframe
        if output_col:
            self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
        else:
            self.y =  np.zeros((self.n, 1))
        
        # Selecting the categorical and continuous columns
        self.cat_cols = cat_cols if cat_cols else []
        self.cont_cols = [col for col in data.columns if col not in self.cat_cols + [output_col]]
        
        # Continuous part of the dataframe
        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.n, 1))
        
        # Categorical part of the dataframe
        if self.cat_cols:
            self.cat_X = data[cat_cols].astype(np.int64).values
        else:
            self.cat_X =  np.zeros((self.n, 1))
        
    
    def __len__(self):
        """
        Returns the total number of samples.
        """
        return self.n

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return {"target": self.y[idx], "cont_data": self.cont_X[idx], "cat_data": self.cat_X[idx]}
    
    # get indexes for train and test rows
    def get_splits(self, split=0.33):
        """
        Get indexes for train and validation rows
        """
        # Determines size
        test_size = round(split * self.n)
        train_size = self.n - test_size
        
        # calculate the split
        return random_split(self, [train_size, test_size])
class NNModel(pl.LightningModule):

    def __init__(self, emb_dims, no_of_cont, lin_layer_sizes, output_size, emb_dropout, lin_layer_dropouts):
        """
        Parameters
        ----------
        emd_dims: list of two elements tuples
            The list contains a two element tuple for each categorical feature. The first element tuple denotes 
            the number of unique values of the categorical feture. The second element tuple denotes the embedding 
            dimension to be used for the feature. 
        no_of_cont: int
            Number of continuous features in the data.
        lin_layer_sizes: list of int
            The list contains the size of each linear layer. The lenght of of the list is the total number of linear 
            layers in the NN.
        output_size: int 
            The size of the final output.
        emd_dropout: float
            The dropout to be used after the embedding layers.
        lin_layer_dropouts: list of floats
            The dropouts to be used after each linear layer.
        """

        super().__init__()

        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        
        # Number of embeddings
        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.no_of_cont = no_of_cont

        # Linear Layers
        first_lin_layer = nn.Linear(self.no_of_embs + self.no_of_cont,lin_layer_sizes[0])

        self.lin_layers = nn.ModuleList(
            [first_lin_layer] + [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1]
        ) for i in range(len(lin_layer_sizes) - 1)])
    
        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        # Output Layer
        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in lin_layer_sizes])

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList([nn.Dropout(size) for size in lin_layer_dropouts])
        
        self.relu = nn.ReLU()

    def forward(self, batch):
        
        # Embeds categorical data
        if self.no_of_embs != 0:
            x = [emb_layer(batch["cat_data"][:, i]) for i,emb_layer in enumerate(self.emb_layers)]
            x = torch.cat(x, 1)
            x = self.emb_dropout_layer(x)
        
        # Embeds continuous data
        if self.no_of_cont != 0:
            normalized_cont_data = self.first_bn_layer(batch["cont_data"])
        
        # Concatenation of categorical and continuous data after initialization
        if self.no_of_embs != 0:
            x = torch.cat([x, normalized_cont_data], 1) 
        else:
            x = normalized_cont_data
        
        # Hidden layers
        for lin_layer, dropout_layer, bn_layer in zip(self.lin_layers, self.droput_layers, self.bn_layers): 
            x = self.relu(lin_layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)
        
        # Outuput layer
        x = self.output_layer(x)

        return x

    def training_step(self, batch, idx):
        logits = self.forward(batch)
        loss = nn.L1Loss()(logits, batch["target"])
        self.log('train_loss', loss)
        
        return loss

    def validation_step(self, batch, idx):
        logits = self.forward(batch)
        loss = nn.L1Loss()(logits, batch["target"])
        self.log('val_loss', loss)
    
    def test_step(self, batch, idx):
        logits = self.forward(batch)
        test_loss = nn.L1Loss()(logits, batch['target'])
        print(logits, batch["target"], test_loss)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
    
class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")
        
# Reads the data
data = pd.read_csv("train_delay_data.csv", index_col=0)

# Select the categorical features and the target/output feature
categorical_features = ["Day of operation", "Linie", "Stop name"]
output_feature = "Delay"
no_output_feature = 1

# Label encoding categorical features
label_encoders = {}
for cat_col in categorical_features:
    label_encoders[cat_col] = LabelEncoder()
    data[cat_col] = label_encoders[cat_col].fit_transform(data[cat_col])

# Creates dataset
dataset = TabularDataset(data=data, cat_cols=categorical_features, output_col=output_feature)

# Splits the data into train and validation set
split = 0.2
train_set, val_set = dataset.get_splits(split=split)

# Creates dataloaders
batch_size = 64
train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
val_dl = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

# Computes the uniques valuse for every categorical features
cat_dims = [int(data[col].nunique())+1 for col in categorical_features]

# Creates emb_dims parameter. We can also do this by hand. ATTENTION: x > y mandatory!!!
emb_dims = [(x, int(x**(0.40))) for x in cat_dims]

# Control on the choice of the embedding dimensions
if sum([x <= y for x,y in emb_dims])>0: print("==> Warning! There is an x <= y")

# Number of continuous features
no_of_cont = len(data.columns) - len(categorical_features) - no_output_feature

# Defines the model
model = NNModel(
    emb_dims, 
    no_of_cont=no_of_cont, 
    lin_layer_sizes=[5, 5], 
    output_size=no_output_feature, 
    emb_dropout=0.04, 
    lin_layer_dropouts=[0.01,0.01]
    )

# Define an early stop
early_stop_callback = EarlyStopping(
    monitor="val_loss", 
    min_delta=3, 
    patience=3, 
    verbose=True, 
    mode="max"
)

# Training the model
training = pl.Trainer(
    max_epochs= 1, 
    callbacks=[MyPrintingCallback(), early_stop_callback],
    logger=wandb_logger
    )

training.fit(model, train_dl, val_dl)
