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

from sklearn.preprocessing import LabelEncoder

wandb_logger = WandbLogger(project="Train delay prediction")

# Dictionary containig all parameters used. The script will compute the parameters set to None
# However, we can also set these parameters manually if wished
params = {"data": {"cat_data": ["Day of operation", "Linie", "Stop name"], 
                    "cont_data": ["Departure time"], 
                    "output": "Delay"},   
          "preprocessing": {"MinMaxScaler": {"Departure time": {"min": 0, "max": 86400},
                                             "Delay": {"min": 0,"max": 1200}},
                            "LabelEncoder": {"Day of operation": None,
                                             "Linie": None, 
                                             "Stop name": None}},
          "model": {"emb_dims": None, 
                    "lin_layer_sizes": [5, 5, 5],
                    "emb_dropout": 0.04,
                    "lin_layer_dropouts": [0.01, 0.01, 0.01]},
          "epochs": 5  
}

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
 
def preprocessing_data(data, cat_data, cont_data, output_data):
    """
    Preprocessing the data.
    
    Parameters
    ----------
    data: DataFrame
        Data to preprocess
    cat_data: list of strings
        List containting the names of the categorical columns
    cont_data: list of strings
        List containing the names of the continuous columns
    
    Return
    ------
    data: DataFrame
        Preprocessed data.
    """
    
    # Preprocessing continuous data using MinMaxScaler
    for cont_col in cont_data: 
        
        # Retrieving the values from params dictionary
        minimum = params["preprocessing"]["MinMaxScaler"][cont_col]["min"]
        maximum = params["preprocessing"]["MinMaxScaler"][cont_col]["max"]
            
        data[cont_data] = data[cont_data].apply(lambda x: (x-minimum)/(maximum-minimum))

    # Preprocessing categorical data
    
    # If is the first time, it creates the dictionary labels and stores it in params
    if params["preprocessing"]["LabelEncoder"]["Day of operation"] == None:
        
        for cat_col in cat_data:
            le = LabelEncoder()
            ids = le.fit_transform(data[cat_col])
            mapping = dict(zip(le.classes_, range(len(le.classes_))))
            params["preprocessing"]["LabelEncoder"][cat_col] = mapping
            
    # Labelling the categorical data
    for cat_col in cat_data:
        data[cat_col] = data[cat_col].apply(lambda x: 
                            params["preprocessing"]["LabelEncoder"][cat_col][x])
        
        
    # Preprocessing output data using MinMaxScaler
    # Retrieving the values from params dictionary
    min_output = params["preprocessing"]["MinMaxScaler"][output_data]["min"]
    max_output = params["preprocessing"]["MinMaxScaler"][output_data]["max"]
            
    data[output_data] = data[output_data].apply(lambda x: (x-min_output)/(max_output-min_output))
    
    return data

def preprocessing_data_inverse(data, cont_data, maximum, minimum):
    """
    Inverse the continuous data preprocessed with MinMaxScaler. 
    
    Parameters
    ----------
    data: DataFrame
        All data
    cont_data: list of strings
        List containing the names of the continuous columns to inverse
    maximum: int
    minimum: int
    
    Return
    ------
    data: DataFrame
        Dataframe with the continuous columns inverse preprocessed.
    """
    for col in cont_data: 
        data[col] = data[col].apply(lambda x: x*(maximum-minimum)+minimum)
    
    return data

# Reads the data
data = pd.read_csv("train_data.csv", index_col=0)

# Selects the categorical and continuous features and the target/output feature
categorical_features = params["data"]["cat_data"]
cont_features = params["data"]["cont_data"]
output_feature = params["data"]["output"]
no_output_feature = 1
no_of_cont_features = len(cont_features)

# Preprocesses the data
data = preprocessing_data(data, categorical_features, cont_features, output_feature)

# Creates dataset
dataset = TabularDataset(data=data, cat_cols=categorical_features, output_col=output_feature)

# Splits the data into train and validation set
split = 0.2
train_set, val_set = dataset.get_splits(split=split)

# Creates dataloaders
batch_size = 64
train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True)
val_dl = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True)

# If the embedding dimensions are not given, it computes it automatically
if params["model"]["emb_dims"]==None:
    # Computes the uniques valuse for every categorical features
    cat_dims = [int(data[col].nunique())+1 for col in categorical_features]

    # Creates emb_dims parameter. We can also do this by hand. ATTENTION: x > y mandatory!!!
    params["model"]["emb_dims"] = [(x, int(x**(0.40))) for x in cat_dims]

# Defines the model
model = NNModel(
    emb_dims=params["model"]["emb_dims"], 
    no_of_cont=no_of_cont_features, 
    lin_layer_sizes=params["model"]["lin_layer_sizes"], 
    output_size=no_output_feature, 
    emb_dropout=params["model"]["emb_dropout"], 
    lin_layer_dropouts=params["model"]["lin_layer_dropouts"]
    )

# Define an early stop
early_stop_callback = EarlyStopping(
    monitor="val_loss", 
    min_delta=3, 
    patience=5, 
    verbose=True, 
    mode="max"
)

# Training the model
training = pl.Trainer(
    max_epochs= params["epochs"], 
    callbacks=[MyPrintingCallback(), early_stop_callback],
    logger=wandb_logger,
    )

training.fit(model, train_dl, val_dl)

# We use the model to make prediction on the test data. Aftert that we had the prediction on the test data DataFrame. Then, we will analyze it
# to improve the model

# Reads the test data
test_data = pd.read_csv("test_data.csv", index_col=0)

# Preprocesses the test_data
test_data = preprocessing_data(test_data, categorical_features, cont_features, output_feature)

# Creates dataset and dataloader
test_dataset = TabularDataset(data=test_data, cat_cols=categorical_features, output_col=output_feature)
test_dl = DataLoader(test_dataset, shuffle=False, num_workers=2, persistent_workers=True)

# Makes predictions
predicted_delay = training.predict(model, test_dl)

# Converts tensor to list and add it to the test_data DataFrame
predicted_delay_list = [np.array(x)[0][0] for x in predicted_delay]
test_data['Predicted delay'] = predicted_delay_list

# Inverse preprocessing of the data
test_data = preprocessing_data_inverse(data=test_data, 
                                       cols=["Delay", "Predicted delay"],
                                       minimum=params["preprocessing"]["MinMaxScaler"]["Delay"]["min"],
                                       maximum=params["preprocessing"]["MinMaxScaler"]["Delay"]["max"])
