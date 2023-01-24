# Importing packages
import os
import numpy as np
import pandas as pd

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix


def get_sets(base_path):
    # Reading the training dataset.
    data_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
    data = data_df.values

    # Spliting it into two separate datasets, validation and train.
    np.random.seed(2)
    inds = np.arange(data.shape[0])
    np.random.shuffle(inds)
    num_train = int(0.8 * data.shape[0])
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
    test_data = pd.read_csv(os.path.join(base_path, 'test.csv'))
    test_data = test_data.values
    test_data = (test_data - mu) / sigma

    return train_X, train_y, val_X, val_y, test_data


# MultiClassLogisticRegression implementation
class MultiClassLogisticRegression:

    def __init__(self, n_iter=1000, threshold=1e-3):
        self.n_iter = n_iter
        self.threshold = threshold

    def fit(self, X, y, batch_size=64, lr=0.001, rand_seed=42):
        np.random.seed(rand_seed)
        self.classes = np.unique(y)
        self.class_labels = {c: i for i, c in enumerate(self.classes)}

        X = self.add_bias(X)
        y = self.one_hot(y)

        self.loss = []
        self.weights = np.zeros(shape=(len(self.classes), X.shape[1]))
        self._fit_data(X, y, batch_size, lr)
        return self

    def _fit_data(self, X, y, batch_size, lr):
        i = 0
        while (not self.n_iter or i < self.n_iter):
            current_loss = self.cross_entropy(y, self._predict(X))
            self.loss.append(current_loss)

            idx = np.random.choice(X.shape[0], batch_size)
            X_batch, y_batch = X[idx], y[idx]

            error = y_batch - self._predict(X_batch)

            update = (lr * np.dot(error.T, X_batch))

            self.weights += update

            if np.abs(update).max() < self.threshold:
                break

            i += 1

    def add_bias(self, X):
        return np.insert(X, 0, 1, axis=1)

    def one_hot(self, y):
        return np.eye(len(self.classes))[np.vectorize(lambda c: self.class_labels[c])(y).reshape(-1)]

    def cross_entropy(self, y, probs):
        return -1 * np.mean(y * np.log(probs))

    def predict(self, X):
        return self._predict(self.add_bias(X))

    def _predict(self, X):
        pre_vals = np.dot(X, self.weights.T).reshape(-1, len(self.classes))
        return self.softmax(pre_vals)

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)

    def score(self, X, y):
        return np.mean(self.predict_classes(X) == y)

    def predict_classes(self, X):
        self.probs_ = self.predict(X)
        return np.vectorize(lambda c: self.classes[c])(np.argmax(self.probs_, axis=1))


# Training phase
train_X, train_y, val_X, val_y, test_data = get_sets('data')
lr = MultiClassLogisticRegression(threshold=1e-5, n_iter=10000)
lr.fit(train_X, train_y[:, np.newaxis], lr=0.0001)
y_pred = lr.predict_classes(val_X)

# Printing some metrics such as a Accuracy, Balanced Accuracy and Confusion Matrix
val_accuracy = lr.score(val_X, val_y)
val_balanced_accuracy = balanced_accuracy_score(y_pred, val_y)
print('val_accuracy:', val_accuracy)
print('val_balanced_accuracy:', val_balanced_accuracy)

conf_matrix = confusion_matrix(val_y, y_pred)
print('confusion_matrix:\n', conf_matrix)

# Testing phase and Writing results in file
test_predicts = lr.predict_classes(test_data)
df = pd.DataFrame(test_predicts.astype(int), columns=['LABELS'])
df.insert(0, 'S.No', np.array([i for i in range(len(test_data))]))
df.to_csv('predictions/logistic_regression.csv', index=False)
