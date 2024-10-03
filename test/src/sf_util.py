""" Utility Functions for ScalarFlow

This module contains a collection of convenience functions to make it easier
to build machine learning models using ScalarFlow.

author: Nathan Sprague
version: 4/16/2020

"""
import math
import random
import scalarflow as sf
import numpy as np


def negate(x):
    """ Negate a scalarflow node.

    Args:
        x (scalarflow node): input

    Returns:
        Negated node.
    """
    neg_one = sf.Constant(-1.)
    return sf.Multiply(neg_one, x)


def logistic(x):
    """ The logistic function
    
        Args:
            x (scalarflow node): input
        Return:
           A scalarflow node representing logistic(x)
        """
    one = sf.Constant(1.)
    denominator = sf.Add(one, sf.Exp(negate(x)))
    return sf.Divide(one, denominator)


def cross_entropy(y_true, y_pred):
    """ Cross entropy loss
        
        Args:
            y_true (scalarflow node): actual value(s)
            y_pred (scalarflow node): predicted value(s)
        Return:
           Scalarflow node representing cross-entropy loss.
        """
    one = sf.Constant(1.)
    left = sf.Multiply(y_true, sf.Log(y_pred))
    right = sf.Multiply(sf.Subtract(one, y_true),
                        sf.Log(sf.Subtract(one, y_pred)))
    return negate(sf.Add(left, right))


def cum_sum(nodes):
    """ Cumulative Sum

    Args:
        nodes: A list of scalarflow nodes

    Returns:
        A scalarflow node representing the sum of the provided list
    """
    result = nodes[0]
    for node in nodes[1::]:
        result = sf.Add(result, node)
    return result


def array_to_feed_dicts(array, prefix):
    """ Convert a numpy array to a list of feed dictionaries suitable for use
    with the scalarflow run method.

    Each row in the array becomes a single feed dictionary, while each
    column becomes an entry in that dictionary.  For example:

    >>> x = np.array([[1, 2, 3],
    ...               [10, 11, 12]])
    >>> array_to_feed_dicts(x, prefix='x_')
    [{'x_0': 1, 'x_1': 2, 'x_2': 3}, {'x_0': 10, 'x_1': 11, 'x_2': 12}]

    Args:
        array (ndarray): (n, d) numpy array where n is the number of samples
                        and d is the number of features.  Or length n numpy
                        array.
        prefix (string): string prefix to use for variable names.

    Returns:
        A list of dictionaries, where each dictionary contains one entry
        for each column in the provided array.
    """
    feed_dicts = []
    if len(array.shape) == 2:
        for row in range(array.shape[0]):
            feed_dicts.append(dict())
            for col in range(array.shape[1]):
                name = '{}{}'.format(prefix, col)
                feed_dicts[-1][name] = array[row, col]

    else:
        for row in range(array.shape[0]):
            feed_dicts.append(dict())
            feed_dicts[-1][prefix] = array[row]

    return feed_dicts


def xy_to_feed_dicts(x, y, prefix_x='x_', prefix_y='y'):
    """  Create a list of feed dicts from numpy arrays containing feature
    values and labels.

    >>> x = np.array([[1, 2],
    ...               [10, 11]])
    >>> y = np.array([0, 1])
    >>> xy_to_feed_dicts(x, y)
    [{'x_0': 1, 'x_1': 2, 'y': 0}, {'x_0': 10, 'x_1': 11, 'y': 1}]

    Args:
        x (ndarray): (n, d) numpy array where n is the number of samples and
                     d is the number of features.
        y (ndarray):  Length n numpy array.
        prefix_x (string): prefix to use for feature values
        prefix_y (string): prefix to use for labels

    Returns:
        A list of feed_dicts containing both input and output values.

    """
    feed_dicts = array_to_feed_dicts(x, prefix_x)
    feed_dicts_y = array_to_feed_dicts(y, prefix_y)
    for fd, fdy in zip(feed_dicts, feed_dicts_y):
        fd.update(fdy)
    return feed_dicts


def glorot_init(fanin, fanout):
    """ Standard Glorot weight initialization

    Appropriate for sigmoid/logistic non-linearities.

    Args:
        fanin (int):  Number of incoming weights
        fanout (int): Number of outgoing weights

    Returns:
        Random weight value.

    """
    maxw = math.sqrt(6 / (fanin + fanout))
    return random.uniform(-maxw, maxw)


def he_init(fanin):
    """ Standard He weight initialization

     Appropriate for relu non-linearities.

     Args:
         fanin (int):  Number of incoming weights

     Returns:
         Random weight value.

     """
    return np.random.randn() * math.sqrt(2 / fanin)