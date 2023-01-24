import numpy as np
import math

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################


class Q1:

    def feature_means(self, iris):
        features = iris[:, :-1]
        means = np.mean(features, axis=0)
        return means

    def covariance_matrix(self, iris):
        features = iris[:, :-1]
        features = [features[:, 0], features[:, 1], features[:, 2], features[:, 3]]
        return np.cov(features)

    def feature_means_class_1(self, iris):
        iris_class_1 = np.array([x for x in iris if x[4] == 1])
        features_class_1 = iris_class_1[:, :-1]
        mean_class_1 = np.mean(features_class_1, axis=0)
        return mean_class_1

    def covariance_matrix_class_1(self, iris):
        iris_class_1 = np.array([x for x in iris if x[4] == 1])
        features_class_1 = iris_class_1[:, :-1]
        iris_class_1 = [features_class_1[:, 0], features_class_1[:, 1],
                        features_class_1[:, 2], features_class_1[:, 3]]
        return np.cov(iris_class_1)


class HardParzen:
    def __init__(self, h):
        self.h = h

    def train(self, train_inputs, train_labels):
        self.label_list = np.unique(train_labels)
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.n_classes = len(np.unique(train_labels))

    def compute_distances(self, x, train_inputs):
        distances = []
        for (i, x_train) in enumerate(train_inputs):
            distances.append(np.linalg.norm(x - x_train))

        return np.array(distances)

    def one_hot(self, y):
        one_hot = np.zeros(self.n_classes)
        one_hot[int(y) - 1] = 1
        return one_hot

    def get_neighbors_idx(self, x):
        distances = self.compute_distances(x, self.train_inputs)
        neighbors_idx = np.array([i for i in range(len(distances)) if distances[i] < self.h])
        return neighbors_idx

    def compute_predictions(self, test_data):
        num_test = test_data.shape[0]
        classes_pred = np.zeros(num_test)
        for (i, x_test) in enumerate(test_data):

            neighbors_idx = self.get_neighbors_idx(x_test)
            if len(neighbors_idx) == 0:
                classes_pred[i] = draw_rand_label(x_test, self.label_list)
            else:
                one_hot_scores = []
                for neighbor_idx in neighbors_idx:
                    one_hot_yi = self.one_hot(self.train_labels[neighbor_idx])
                    one_hot_scores.append(one_hot_yi)
                one_hot_scores = np.array(one_hot_scores)
                classes_pred[i] = np.argmax(np.sum(one_hot_scores, axis=0)) + 1
        return classes_pred


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma = sigma

    def train(self, train_inputs, train_labels):
        self.label_list = np.unique(train_labels)
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.num_features = train_inputs.shape[1] - 1
        self.n_classes = len(np.unique(train_labels))

    def RBF_kernel(self, xi, x):
        fraction = 1 / (((2. * math.pi) ** (self.num_features / 2.)) * (self.sigma ** self.num_features))
        euclidean_distance = np.linalg.norm(xi - x) ** 2.
        exp = math.exp((-1 / 2) * (euclidean_distance / (self.sigma ** 2)))
        score = fraction * exp
        return score

    def get_scores(self, x, X):
        scores = []
        for (i, xi) in enumerate(X):
            scores.append(self.RBF_kernel(xi, x))
        return np.array(scores)

    def one_hot(self, y):
        one_hot = np.zeros(self.n_classes)
        one_hot[int(y) - 1] = 1
        return one_hot

    def compute_predictions(self, test_data):
        # For each test datapoint
        num_test = test_data.shape[0]
        classes_pred = np.zeros(num_test)
        for (i, x_test) in enumerate(test_data):

            train_scores = self.get_scores(x_test, self.train_inputs)
            one_hot_scores = []
            for (j, y_train) in enumerate(self.train_labels):
                one_hot_yi = self.one_hot(y_train)
                one_hot_scores.append(train_scores[j] * one_hot_yi)
            one_hot_scores = np.array(one_hot_scores)
            classes_pred[i] = np.argmax(np.sum(one_hot_scores, axis=0)) + 1
        return classes_pred


def split_dataset(iris):
    train = []
    val = []
    test = []
    for (i, ex) in enumerate(iris):
        # Training dataset
        if i % 5 == 0 or i % 5 == 1 or i % 5 == 2:
            train.append(ex)

        # Validation dataset
        if i % 5 == 3:
            val.append(ex)

        # Test dataset
        if i % 5 == 4:
            test.append(ex)

    train = np.array(train)
    val = np.array(val)
    test = np.array(test)

    return train, val, test


class ErrorRate:

    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def calc_error(self, model_type, param):
        if model_type == 'soft':
            model = SoftRBFParzen(sigma=param)
        elif model_type == 'hard':
            model = HardParzen(h=param)
        else:
            raise Exception('Model type is undefined!')

        model.train(train_inputs=self.x_train, train_labels=self.y_train)
        y_pred = model.compute_predictions(self.x_val)

        num_misclassified = 0
        for i in range(len(self.y_val)):
            if self.y_val[i] != y_pred[i]:
                num_misclassified += 1
        return num_misclassified / len(self.y_val)

    def hard_parzen(self, h):
        return self.calc_error(model_type='hard', param=h)

    def soft_parzen(self, sigma):
        return self.calc_error(model_type='soft', param=sigma)


def get_test_errors(iris):
    train, val, test, = split_dataset(iris)

    x_train = train[:, :-1]
    y_train = train[:, -1]

    x_val = val[:, :-1]
    y_val = val[:, -1]

    x_test = test[:, :-1]
    y_test = test[:, -1]

    er = ErrorRate(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

    Hs = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
    smallest_hard_parzen_error = float("inf")
    optimal_hard_parzen_h = -1
    for h in Hs:
        current_hard_parzen_error = er.hard_parzen(h=h)
        if current_hard_parzen_error < smallest_hard_parzen_error:
            smallest_hard_parzen_error = current_hard_parzen_error
            optimal_hard_parzen_h = h

    sigmas = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
    smallest_soft_parzen_error = float("inf")
    optimal_soft_parzen_sigma = -1
    for sigma in sigmas:
        current_soft_parzen_error = er.soft_parzen(sigma=sigma)
        if current_soft_parzen_error < smallest_soft_parzen_error:
            smallest_soft_parzen_error = current_soft_parzen_error
            optimal_soft_parzen_sigma = sigma

    test_er = ErrorRate(x_train=x_train, y_train=y_train, x_val=x_test, y_val=y_test)
    hard_parzen_error = test_er.hard_parzen(h=optimal_hard_parzen_h)
    soft_parzen_error = test_er.soft_parzen(sigma=optimal_soft_parzen_sigma)
    return np.array([hard_parzen_error, soft_parzen_error])


def random_projections(X, A):
    return (1/math.sqrt(2)) * np.matmul(X, A)


class Q9:

    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def calc_error(self, model_type, params):

        val_errors = -1 * np.ones((500, 10))
        for i in range(500):
            print(i)
            for (j, param) in enumerate(params):
                if model_type == 'soft':
                    model = SoftRBFParzen(sigma=param)
                elif model_type == 'hard':
                    model = HardParzen(h=param)
                else:
                    raise Exception('Model type is undefined!')

                A = np.random.normal(0, 1, (4, 2))
                X = random_projections(self.x_train, A)
                projected_x_val = random_projections(self.x_val, A)
                model.train(train_inputs=X, train_labels=self.y_train)
                y_pred = model.compute_predictions(projected_x_val)

                num_misclassified = 0
                for k in range(len(self.y_val)):
                    if self.y_val[k] != y_pred[k]:
                        num_misclassified += 1
                val_errors[i][j] = num_misclassified / len(self.y_val)

        return val_errors

    def hard_parzen(self, hs):
        return self.calc_error(model_type='hard', params=hs)

    def soft_parzen(self, sigmas):
        return self.calc_error(model_type='soft', params=sigmas)
