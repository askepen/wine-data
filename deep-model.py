# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser

import pandas as pd

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, Dataset

from time import time


class LitWineRegressor(pl.LightningModule):
    def __init__(self, hidden_dim=128, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(12, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        return loss

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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


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


def cli_main():
    pl.seed_everything(1337)

    # ------------
    # logger
    # ------------
    wandb_logger = WandbLogger(name=f'Run-{time()}',project='WineRegressor')

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max-epochs', default=10, type=int)
    parser.add_argument('--logger', default=wandb_logger)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitWineRegressor.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dataset = WineDataset("data/train.csv")
    wine_test = WineDataset("data/test.csv")
    train_len = round(len(dataset)*0.8)
    wine_train, wine_val = random_split(dataset, [train_len, len(dataset)-train_len])

    train_loader = DataLoader(wine_train, batch_size=args.batch_size,num_workers=4)
    val_loader = DataLoader(wine_val, batch_size=args.batch_size)
    test_loader = DataLoader(wine_test, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    model = LitWineRegressor(args.hidden_dim, args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    cli_main()