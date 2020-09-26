import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/winequality-red.csv')

data_train, data_test = train_test_split(data, test_size=0.25)

print(f"Train set has {len(data_train)} rows")
print(f"Test set has {len(data_test)} rows")

data_train.to_csv("data/train.csv")
data_test.to_csv("data/test.csv")
