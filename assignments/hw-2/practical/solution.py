import numpy as np
import matplotlib.pyplot as plt


class SVM:
    def __init__(self, eta, C, niter, batch_size, verbose):
        self.eta = eta
        self.C = C
        self.niter = niter
        self.batch_size = batch_size
        self.verbose = verbose

    def make_one_versus_all_labels(self, y, m):
        """
        y : numpy array of shape (n,)
        m : int (num_classes)
        returns : numpy array of shape (n, m)
        """
        results = -1 * np.ones((y.shape[0], m))
        for inx, value in enumerate(y):
            results[inx][value] = 1
        return results.astype(int)

    def compute_loss(self, x, y):
        """
        x : numpy array of shape (minibatch size, num_features)
        y : numpy array of shape (minibatch size, num_classes)
        returns : float
        """
        all_losses = np.maximum(0, 2 - y * np.dot(x, self.w)) ** 2
        return np.mean(np.sum(all_losses, axis=1)) + (self.C / 2) * np.sum(self.w ** 2)

    def compute_gradient(self, x, y):
        """
        x : numpy array of shape (minibatch size, num_features)
        y : numpy array of shape (minibatch size, num_classes)
        returns : numpy array of shape (num_features, num_classes)
        """
        output = y * np.maximum(0, 2 - y * np.dot(x, self.w))
        first_term = np.dot(x.T, output)
        first_term = (-2. * first_term) / x.shape[0]

        second_term = self.C * self.w

        grad = first_term + second_term
        return grad

    # Batcher function
    def minibatch(self, iterable1, iterable2, size=1):
        l = len(iterable1)
        n = size
        for ndx in range(0, l, n):
            index2 = min(ndx + n, l)
            yield iterable1[ndx: index2], iterable2[ndx: index2]

    def infer(self, x):
        """
        x : numpy array of shape (num_examples_to_infer, num_features)
        returns : numpy array of shape (num_examples_to_infer, num_classes)
        """
        output = np.argmax(np.dot(x, self.w), axis=1)
        return self.make_one_versus_all_labels(output, self.m)

    def compute_accuracy(self, y_inferred, y):
        """
        y_inferred : numpy array of shape (num_examples, num_classes)
        y : numpy array of shape (num_examples, num_classes)
        returns : float
        """
        num_correct_predicted = 0
        for row in range(y.shape[0]):
            if np.argmax(y[row]) == np.argmax(y_inferred[row]):
                num_correct_predicted += 1
        return num_correct_predicted / y.shape[0]

    def fit(self, x_train, y_train, x_test, y_test):
        """
        x_train : numpy array of shape (number of training examples, num_features)
        y_train : numpy array of shape (number of training examples, num_classes)
        x_test : numpy array of shape (number of training examples, nujm_features)
        y_test : numpy array of shape (number of training examples, num_classes)
        returns : float, float, float, float
        """
        self.num_features = x_train.shape[1]
        self.m = y_train.max() + 1
        y_train = self.make_one_versus_all_labels(y_train, self.m)
        y_test = self.make_one_versus_all_labels(y_test, self.m)
        self.w = np.zeros([self.num_features, self.m])

        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []

        for iteration in range(self.niter):
            # Train one pass through the training set
            for x, y in self.minibatch(x_train, y_train, size=self.batch_size):
                grad = self.compute_gradient(x, y)
                self.w -= self.eta * grad

            # Measure loss and accuracy on training set
            train_loss = self.compute_loss(x_train, y_train)
            y_inferred = self.infer(x_train)
            train_accuracy = self.compute_accuracy(y_inferred, y_train)

            # Measure loss and accuracy on test set
            test_loss = self.compute_loss(x_test, y_test)
            y_inferred = self.infer(x_test)
            test_accuracy = self.compute_accuracy(y_inferred, y_test)

            if self.verbose:
                print(f"Iteration {iteration} | Train loss {train_loss:.04f} | Train acc {train_accuracy:.04f} |"
                      f" Test loss {test_loss:.04f} | Test acc {test_accuracy:.04f}")

            # Record losses, accs
            train_losses.append(train_loss)
            train_accs.append(train_accuracy)
            test_losses.append(test_loss)
            test_accs.append(test_accuracy)

        return train_losses, train_accs, test_losses, test_accs


# DO NOT MODIFY THIS FUNCTION
def load_data():
    # Load the data files
    print("Loading data...")
    x_train = np.load("x_train_cifar10_reduced.npz")["x_train"]
    x_test = np.load("x_test_cifar10_reduced.npz")["x_test"]
    y_train = np.load("y_train_cifar10_reduced.npz")["y_train"]
    y_test = np.load("y_test_cifar10_reduced.npz")["y_test"]

    # normalize the data
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # add implicit bias in the feature
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)

    return x_train, y_train, x_test, y_test


def generate_plots():
    cs = [1, 10, 30]
    x_train, y_train, x_test, y_test = load_data()
    train_ls = []
    test_ls = []
    train_as = []
    test_as = []

    for c in cs:
        print("Fitting the model...")
        svm = SVM(eta=0.0001, C=c, niter=200, batch_size=5000, verbose=True)
        train_losses, train_accs, test_losses, test_accs = svm.fit(x_train, y_train, x_test, y_test)

        train_ls.append(train_losses)
        train_as.append(train_accs)

        test_ls.append(test_losses)
        test_as.append(test_accs)

    save_plot(cs=cs, values_list=train_ls, name='Train Loss')
    save_plot(cs=cs, values_list=train_as, name='Train Accuracy')

    save_plot(cs=cs, values_list=test_ls, name='Test Loss')
    save_plot(cs=cs, values_list=test_as, name='Test Accuracy')


def save_plot(cs, values_list, name):
    plt.figure()
    X = [i for i in range(len(values_list[0]))]
    for idx, c in enumerate(cs):
        plt.plot(X, values_list[idx], label='c: {}'.format(c))
        plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel(name)
    plt.savefig('plots/{}.png'.format(name), dpi=300)


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data()

    print("Fitting the model...")
    svm = SVM(eta=0.0001, C=2, niter=200, batch_size=5000, verbose=True)
    train_losses, train_accs, test_losses, test_accs = svm.fit(x_train, y_train, x_test, y_test)
    #
    # # to infer after training, do the following:
    # y_inferred = svm.infer(x_test)
    #
    # # to compute the gradient or loss before training, do the following:
    # y_train_ova = svm.make_one_versus_all_labels(y_train, 8) # one-versus-all labels
    # svm.w = np.zeros([3073, 8])
    # grad = svm.compute_gradient(x_train, y_train_ova)
    # loss = svm.compute_loss(x_train, y_train_ova)
    # print(loss)

    generate_plots()
