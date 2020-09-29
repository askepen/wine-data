import wandb

if __name__ == '__main__':

    sweep_config = {
        'program': 'train.py',
        'method': 'bayes', #grid, random, bayesian
        'metric': {
            'name': 'valid_loss',
            'goal': 'minimize'   
        },
        'parameters': {
            'batch_size': {
                'values': [8, 16, 32, 64, 128, 256]
            },
            'hidden_dim': {
                'values': [32, 64, 128, 256, 512, 1024]
            },
            'n_hidden_layers': {
                'values': [1, 2, 3, 4, 5]
            },
            'learning_rate': {
                'values': [1e-2, 1e-3, 1e-4, 3e-4, 3e-5, 1e-5]
            },
            'max_epochs': {
                'values': [2, 5, 10, 20, 40, 80]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="WineRegressor")
    wandb.agent(sweep_id)


