import numpy as np
import pandas as pd
import os
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def get_sets(base_path):
    # Reading the training dataset.
    data_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
    data = data_df.drop(['S.No'], axis=1)
    data = data.drop(['time'], axis=1)
    data = data.values

    # Spliting it into two separate datasets, validation and train.
    np.random.seed(2)
    inds = np.arange(data.shape[0])
    np.random.shuffle(inds)
    num_train = int(0.95 * data.shape[0])
    train_inds = inds[:num_train]
    val_inds = inds[num_train:]

    trainset = data[train_inds]
    valset = data[val_inds]

    # Normalzing the dataset using z-score.
    mu = trainset[:, :-1].mean(axis=0)
    sigma = trainset[:, :-1].std(axis=0)
    trainset[:, :-1] = (trainset[:, :-1] - mu) / sigma
    valset[:, :-1] = (valset[:, :-1] - mu) / sigma

    # Extracing features and labels of each dataset.
    train_X = trainset[:, :-1]
    train_y = trainset[:, -1]

    val_X = valset[:, :-1]
    val_y = valset[:, -1]

    # Reading the test dataset.
    test_data_df = pd.read_csv(os.path.join(base_path, 'test.csv'))
    test_data_df = test_data_df.drop(['S.No'], axis=1)
    test_data_df = test_data_df.drop(['time'], axis=1)
    test_data = test_data_df.values
    test_data = (test_data - mu) / sigma

    return train_X, train_y, val_X, val_y, test_data


train_X, train_y, val_X, val_y, test_data = get_sets('data')

# Doing Grid Searching on Random Forest hyper-parameters
max_depth = [10, 20, 30, 40]
min_samples_leaf = [2, 5, 6, 10, 20, 30, 40]

param_grid = {
    "n_estimators": [100],
    "max_depth": max_depth,
    "min_samples_leaf": min_samples_leaf,
    "bootstrap": [True]
}

clf = GridSearchCV(RandomForestClassifier(), param_grid).fit(train_X, train_y)
y_pred = clf.predict(val_X)

# Printing best hyper parameters and some metrics such as a Accuracy, Balanced Accuracy and Confusion Matrix
print('best parameters:', clf.best_params_)
print('val_accuracy:', clf.score(val_X, val_y))
print('val_balanced_accuracy:', balanced_accuracy_score(y_pred, val_y))
print('confusion_matrix:\n', confusion_matrix(val_y, y_pred))

# Testing phase and writing results in file
clf_predicts = clf.predict(test_data)
df = pd.DataFrame(clf_predicts.astype(int), columns=['LABELS'])
df.insert(0, 'S.No', np.array([i for i in range(len(test_data))]))
df.to_csv('predictions/random_forest.csv', index=False)
