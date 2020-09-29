import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.nn import functional as F
import pandas as pd

class LitWineRegressor(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hidden_dim = hparams.hidden_dim # Needs to have some value before initialization
        self.n_hidden_layers = hparams.n_hidden_layers # Needs to have some value before initialization
        self.save_hyperparameters(hparams)

        self.l1 = torch.nn.Linear(12, self.hidden_dim)
        self.l_hidden = [torch.nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_hidden_layers-1)]
        self.l_last = torch.nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        for n in range(self.n_hidden_layers-1):
            x = torch.relu(self.l_hidden[n](x))
        x = torch.relu(self.l_last(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss)
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('valid_loss', loss)
        return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('test_loss', loss)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

class WineDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with wine data.
            transform (callable, optional): Optional transform to be applied
        """
        self.data = pd.read_csv(csv_file)
        self.features = self.data.iloc[:,:-1].astype(float).to_numpy()
        self.targets = self.data.iloc[:,-1].astype(float).to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = torch.Tensor(self.features[idx,:]), torch.Tensor([self.targets[idx]])
        
        if self.transform:
            sample = self.transform(sample)
        return sample
