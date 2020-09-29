
from argparse import ArgumentParser
from time import time

import wandb
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split

# Sweep parameters
hyperparameter_defaults = dict(
    batch_size = 32,
    hidden_dim = 128,
    learning_rate = 1e-3,
    max_epochs = 30
)

wandb.init(config=hyperparameter_defaults)
config = wandb.config


class LitWineRegressor(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hidden_dim = hparams.hidden_dim # Needs to have some value before initialization
        self.save_hyperparameters(hparams)
        self.l1 = torch.nn.Linear(12, self.hidden_dim)
        self.l2 = torch.nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
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


def cli_main(hparams):
    pl.seed_everything(1337)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=hparams.batch_size, type=int)
    parser.add_argument('--max-epochs', default=hparams.max_epochs, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--learning_rate', default=hparams.learning_rate, type=int)
    parser.add_argument('--hidden_dim', default=hparams.hidden_dim, type=int)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dataset = WineDataset("data/train.csv")
    wine_test = WineDataset("data/test.csv")
    train_len = round(len(dataset)*0.8)
    wine_train, wine_val = random_split(dataset, [train_len, len(dataset)-train_len])

    train_loader = DataLoader(wine_train, batch_size=hparams.batch_size)
    val_loader = DataLoader(wine_val, batch_size=hparams.batch_size)
    test_loader = DataLoader(wine_test, batch_size=hparams.batch_size)

    # ------------
    # model
    # ------------
    model = LitWineRegressor(args)

    # ------------
    # logger
    # ------------
    wandb_logger = WandbLogger(name=f'Run-{int(time())}',project='WineRegressor')

    # ------------
    # training
    # ------------
    print(args)
    trainer = pl.Trainer.from_argparse_args(args,logger=wandb_logger, row_log_interval=5)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    
    #trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    cli_main(config)
