# Wine data 🍷
Personal project exploring and predicting wine quality 

## Data exploration 
__Key insight:__ The wine quality feature in the dataset is unbalanced and highly biased towards values of 5 and 6.

![Quality Balance](https://github.com/askepen/wine-data/blob/master/img/quality_balance.png?raw=true)

__Key insight:__ No significant correlation is fond directly between any of the features and wine quality

![Correlation Plot](https://github.com/askepen/wine-data/blob/master/img/corr_plot.png?raw=true)

## Simple models 
Reducing the data down to fewer dimensions we have the following dataset:

![PCA reduction](https://github.com/askepen/wine-data/blob/master/img/PCA.png?raw=true)

Training a SVM and a kNN, we obtain mediocre results

|         | __RMSE Score__ |
|---------|----------------|
| __SVM__ | 0.56           |
| __kNN__ | 0.59           |
|         |                |


## Deep model
Training a simple neural net and using [wandb](http://wandb.ai)'s sweeper to tune the hypperparameters, we obtain slightly better results:

![Train/Validation loss](https://github.com/askepen/wine-data/blob/master/img/wandb_train_val.png?raw=true)


