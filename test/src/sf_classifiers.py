"""Binary classifiers using low-level ScalarFlow.

Author: Nathan Sprague && ...
Version: 4/16/2020

"""

import numpy as np
import scalarflow as sf
import sf_util

import sklearn.metrics
import abc
import random


class AbstractSFClassifier(abc.ABC):
    """ Abstract base class for simple ScalarFlow-based classifiers. """

    def __init__(self):
        self.parameters = []  # List of all trainable variables
        self.graph = sf.Graph()

    def predict(self, x):
        """ Predict real-valued probability for the provided input(s).

        Args:
            x (ndarray): (n, d) numpy array where n is the number of samples
                         and d is the number of features.
        Returns:
            length n numpy array where each entry corresponds to a row in x.
        """
        feed_dicts = sf_util.array_to_feed_dicts(x, 'x_')
        results = []
        for feed_dict in feed_dicts:
            results.append(self.graph.run(self.predict_node,
                                          feed_dict=feed_dict))
        return np.array(results)

    def train(self, x, y, learning_rate=.001, epochs=10):
        """Train the classifier using SGD for the specified number of epochs.

        No mini-batches are used.  Weight updates are performed individually
        for each sample.

        Args:
            x (ndarray): (n, d) numpy array where n is the number of samples
                         and d is the number of features.
            y (ndarray): length n numpy array containing class labels.
            learning_rate (float): the learning rate.
            epochs (int): The number of times to iterate through the dataset
                          during training.

        """
        feed_dicts = sf_util.xy_to_feed_dicts(x, y)

        print("TRAINING...")
        print(feed_dicts, '\n\n\n')
        print(len(feed_dicts))  
        for epoch in range(epochs):
            random.shuffle(feed_dicts)

            losses = []
            for feed_dict in feed_dicts:
                loss = self.graph.run(self.loss_node, feed_dict=feed_dict,
                                      compute_derivatives=True)
                losses.append(loss)

                for param in self.parameters:
                    param.assign(param.value - learning_rate * param.derivative)

            total_loss = sum(losses)

            print("Epoch {} loss: {}".format(epoch, total_loss))

    def score(self, x, y):
        """ Return the accuracy of this model on the provided dataset and
        print a confusion matrix.

        Args:
            x (ndarray): (n, d) numpy array where n is the number of samples
                         and d is the number of features.
            y (ndarray): length n numpy array containing class labels.


        Returns:
            Accuracy as a float.
        """

        all_y_hat = np.array(np.array(self.predict(x)) > .5, dtype=int)

        accuracy = sklearn.metrics.accuracy_score(y, all_y_hat)

        print("Accuracy: {:.5f}".format(accuracy))
        print("Confusion Matrix")
        print(sklearn.metrics.confusion_matrix(y, all_y_hat))
        return accuracy

    def plot_2d_predictions(self, features, labels):
        """Plot the decision surface and class labels for a 2D dataset."""

        import matplotlib.pyplot as plt

        minx = np.min(features[:, 0])
        maxx = np.max(features[:, 0])
        miny = np.min(features[:, 1])
        maxy = np.max(features[:, 1])

        x = np.linspace(minx, maxx, 100)
        y = np.linspace(miny, maxy, 100)
        xx, yy = np.meshgrid(x, y)
        fake_features = np.array(np.dstack((xx, yy)).reshape(-1, 2),
                                 dtype=np.float32)
        z = np.array(self.predict(fake_features)).reshape(len(x), len(y))
        plt.imshow(np.flipud(np.reshape(z, (len(x), len(y)))),
                   vmin=-.2, vmax=1.2, extent=(minx, maxx, miny, maxy),
                   cmap=plt.cm.gray)
        CS = plt.contour(x, y, z)
        plt.clabel(CS, inline=1)

        markers = ['o', 's']
        for label in np.unique(labels):
            plt.scatter(features[labels == label, 0],
                        features[labels == label, 1],
                        marker=markers[int(label)])

        plt.show()


class LogisticRegression(AbstractSFClassifier):
    """ Simple Logistic Regression Classifier """

    def __init__(self, input_dim):
        """ Build the classifier and initialize weights randomly.

        Args:
            input_dim (int): dimensionality of the input
        """

        super().__init__()

        with self.graph:
            b = sf.Variable(.1, name='b')
            self.parameters.append(b)

            products = [b]
            for i in range(input_dim):
                cur_w = sf.Variable(.1, name='w_{}'.format(i))
                cur_x = sf.Placeholder(name='x_{}'.format(i))
                self.parameters.append(cur_w)
                products.append(sf.Multiply(cur_w, cur_x))

            self.predict_node = sf_util.logistic(sf_util.cum_sum(products))

            y_true = sf.Placeholder(name='y')
            self.loss_node = sf_util.cross_entropy(y_true, self.predict_node)


class MLP(AbstractSFClassifier):
    """ Simple multi-layer neural network classifier."""

    def __init__(self, input_dim, hidden_sizes, activation='sigmoid'):
        """

        Args:
            input_dim (int): dimensionality of the input
            hidden_sizes:   A list of ints where each entry corresponds to the
                            number of units in the corresponding hidden layer.
                            E.g. [32, 32, 32] would result in an MLP with
                            three hidden layers with 32 units each.
            activation (string): 'sigmoid' or 'relu'
        """
        super().__init__()

        with self.graph:
            # Create the input nodes
            cur_layer = []
            for i in range(input_dim):
                cur_layer.append(sf.Placeholder(name='x_{}'.format(i)))

            # Create hidden layers
            prev_layer = cur_layer
            for layer, hidden_size in enumerate(hidden_sizes):
                prev_size = len(prev_layer)
                cur_layer = []
                for j in range(hidden_size):
                    cur_b = sf.Variable(1., name='b_{}_{}'.format(layer + 1, j))
                    self.parameters.append(cur_b)
                    products = [cur_b]
                    for i in range(prev_size):

                        if activation == 'sigmoid':
                            init_w = sf_util.glorot_init(prev_size, hidden_size)
                        else:
                            # use sf_util.he_init
                            raise ValueError("Unrecognized activation " +
                                             activation)

                        cur_w_name = 'w_{}_{}_{}'.format(layer + 1, i, j)
                        cur_w = sf.Variable(init_w, name=cur_w_name)
                        self.parameters.append(cur_w)
                        products.append(sf.Multiply(cur_w, prev_layer[i]))
                    total = sf_util.cum_sum(products)
                    if activation == 'sigmoid':
                        cur_layer.append(sf_util.logistic(total))
                    else:
                        raise ValueError("Unrecognized activation " +
                                         activation)

                prev_layer = cur_layer

            # Create output
            prev_size = len(prev_layer)
            b = sf.Variable(1., name='b')
            self.parameters.append(b)
            products = [b]
            for i in range(prev_size):
                cur_w = sf.Variable(sf_util.glorot_init(prev_size, 1),
                                    name='w_{}'.format(i))
                self.parameters.append(cur_w)
                products.append(sf.Multiply(cur_w, prev_layer[i]))

            self.predict_node = sf_util.logistic(sf_util.cum_sum(products))

            y_true = sf.Placeholder(name='y')
            self.loss_node = sf_util.cross_entropy(y_true, self.predict_node)


if __name__ == "__main__":
    pass