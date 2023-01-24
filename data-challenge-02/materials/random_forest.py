# importing packages
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# setting the random seed to get same results everytime.
np.random.seed(2)

# reading and preparing the dataset.
print('1. reading the dataset.')
data = pd.read_csv('./data/cropharvest-crop-detection/train.csv', index_col=0)

# Splitting the dataset to train and validation datasets.
data = data.values
inds = np.arange(data.shape[0])
np.random.shuffle(inds)
train_set = data[inds]
train_X = train_set[:, :-1]
train_y = train_set[:, -1]

param_grid = {
    "n_estimators": [1000],
    "min_samples_leaf": [3],
    "bootstrap": [False]
}

# Defining and training the model.
print('2. training a random forest model with the following hyper parameters:')
print('\thyper-parameters:', param_grid)
clf = GridSearchCV(RandomForestClassifier(), param_grid).fit(train_X, train_y)

# Reading the test dataset.
test_data = pd.read_csv('./data/cropharvest-crop-detection/test_nolabels.csv', index_col=0)
test_data = test_data.values

# Making predictions.
clf_predicts = clf.predict(test_data)

# Saving results.
print('3. saving the predictions.')
df = pd.DataFrame(clf_predicts.astype(int), columns=['LABELS'])
df.insert(0, 'S.No', np.array([i for i in range(len(test_data))]))
df.to_csv('./predictions/results.csv', index=False)
