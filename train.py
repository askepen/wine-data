import pytorch_lightning as pl
import wandb

from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split
from argparse import ArgumentParser
from time import time
from model import WineDataset, LitWineRegressor

if __name__ == '__main__':
    pl.seed_everything(1337)
    
    # -----------
    # hyperparameters 
    # -----------
    hyperparameter_defaults = dict(
        batch_size = 32,
        hidden_dim = 128,
        learning_rate = 1e-3,
        n_hidden_layers = 2,
        max_epochs = 30
    )

    wandb.init(config=hyperparameter_defaults)
    hparams = wandb.config
    
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=hparams.batch_size, type=int)
    parser.add_argument('--max-epochs', default=hparams.max_epochs, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--learning_rate', default=hparams.learning_rate, type=float)
    parser.add_argument('--hidden_dim', default=hparams.hidden_dim, type=int)
    parser.add_argument('--n_hidden_layers', default=hparams.n_hidden_layers, type=int)
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
    wandb_logger = WandbLogger(name=f'Run-{int(time())}', project='WineRegressor')

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args,logger=wandb_logger, row_log_interval=5)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    
    #trainer.test(test_dataloaders=test_loader)