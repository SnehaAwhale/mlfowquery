from sklearn.model_selection import train_test_split

import pandas as pd
URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(URL, sep=";")

train, test = train_test_split(df)
test_x = test.drop(["quality"], axis=1)

print(test_x)
