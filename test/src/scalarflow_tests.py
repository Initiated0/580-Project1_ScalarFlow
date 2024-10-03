"""
Unit tests for ScalarFlow

Author: Nathan Sprague
Version: 4/19/2020
"""

import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import scalarflow as sf


def in_class_example_graph():
    """ return the graph for sqrt(x^2 + y^2) """
    graph = sf.Graph()
    with graph as g:
        x = sf.Placeholder(name='x')
        y = sf.Placeholder(name='y')

        x_squared = sf.Pow(x, 2, name='x2')
        y_squared = sf.Pow(y, 2, name='y2')

        xy_sum = sf.Add(x_squared, y_squared, 'sum')
        func = sf.Pow(xy_sum, .5, 'out')
    return graph


def video_example_graph():
    """ return the graph for (wx - y) + w^2 """
    graph = sf.Graph()
    with graph as g:
        w = sf.Placeholder(name='w')
        x = sf.Placeholder(name='x')
        y = sf.Placeholder(name='y')

        wx = sf.Multiply(w, x, name='wx')
        wx_minus_y = sf.Subtract(wx, y, name='wx_minus_y')
        wx_minus_y_squared = sf.Pow(wx_minus_y, 2, name='wx_minus_y_squared')
        w2 = sf.Pow(w, 2, name='w2')
        loss = sf.Add(wx_minus_y_squared, w2, name='loss')
    return graph


def cross_entropy_graph():
    """ Return a graph representing cross-entropy loss for a tiny logistic
    regression classier with one input and no bias."""
    graph = sf.Graph()
    with graph as g:
        x = sf.Placeholder(name='x')
        w = sf.Placeholder(name='w')
        y_true = sf.Placeholder(name='y')

        prod = sf.Multiply(x, w)

        # Logistic function
        one = sf.Constant(1.)
        neg_one = sf.Constant(-1.)
        denominator = sf.Add(one, sf.Exp(sf.Multiply(neg_one, prod)))
        y_pred = sf.Divide(one, denominator, name='y_pred')

        # Cross entropy loss
        left = sf.Multiply(y_true, sf.Log(y_pred))
        right = sf.Multiply(sf.Subtract(one, y_true),
                            sf.Log(sf.Subtract(one, y_pred)))
        loss = sf.Multiply(neg_one, sf.Add(left, right), name='loss')

    return graph


class ForwardTestCase(unittest.TestCase):

    def test_in_class_example_forward_final(self):
        graph = in_class_example_graph()
        sqrt = graph.nodes_by_name['out']
        result = graph.run(sqrt, feed_dict={'x': 3, 'y': 4})
        assert_almost_equal(result, 5)

    def test_in_class_example_forward_all_values(self):
        graph = in_class_example_graph()
        sqrt = graph.nodes_by_name['out']
        result = graph.run(sqrt, feed_dict={'x': 3, 'y': 4})
        assert_almost_equal(graph.nodes_by_name['x'].value, 3)
        assert_almost_equal(graph.nodes_by_name['y'].value, 4)
        assert_almost_equal(graph.nodes_by_name['x2'].value, 9)
        assert_almost_equal(graph.nodes_by_name['y2'].value, 16)
        assert_almost_equal(graph.nodes_by_name['sum'].value, 25)
        assert_almost_equal(graph.nodes_by_name['out'].value, 5)

    def test_video_example_forward_final(self):
        graph = video_example_graph()
        loss = graph.nodes_by_name['loss']
        result = graph.run(loss, feed_dict={'w': 1, 'x': 2, 'y': 3})
        assert_almost_equal(result, 2)

    def test_video_example_forward_all_values(self):
        graph = video_example_graph()
        loss = graph.nodes_by_name['loss']
        result = graph.run(loss, feed_dict={'w': 1, 'x': 2, 'y': 3})
        assert_almost_equal(graph.nodes_by_name['w'].value, 1)
        assert_almost_equal(graph.nodes_by_name['x'].value, 2)
        assert_almost_equal(graph.nodes_by_name['y'].value, 3)
        assert_almost_equal(graph.nodes_by_name['wx'].value, 2)
        assert_almost_equal(graph.nodes_by_name['wx_minus_y'].value, -1)
        assert_almost_equal(graph.nodes_by_name['wx_minus_y_squared'].value, 1)
        assert_almost_equal(graph.nodes_by_name['w2'].value, 1)
        assert_almost_equal(graph.nodes_by_name['loss'].value, 2)

    def test_cross_entropy_forward(self):
        graph = cross_entropy_graph()
        y_pred = graph.nodes_by_name['y_pred']
        loss = graph.nodes_by_name['loss']

        graph.run(loss, feed_dict={'x': 1, 'w': 0, 'y': 1})
        assert_almost_equal(y_pred.value, .5)
        assert_almost_equal(loss.value, 0.6931471805599453)

        graph.run(loss, feed_dict={'x': 1, 'w': 0, 'y': 0})
        assert_almost_equal(y_pred.value, .5)
        assert_almost_equal(loss.value, 0.6931471805599453)

        graph.run(loss, feed_dict={'x': 1, 'w': 1, 'y': 1})
        assert_almost_equal(y_pred.value, 0.7310585786300049)
        assert_almost_equal(loss.value, 0.3132616875182228)

        graph.run(loss, feed_dict={'x': 1, 'w': -1, 'y': 1})
        assert_almost_equal(y_pred.value, 0.2689414213699951)
        assert_almost_equal(loss.value, 1.3132616875182228)

    # ------------------------------------------------------
    # FORWARD TESTS FOR INDIVIDUAL OPERATORS
    # ------------------------------------------------------

    # Scalar types --------------------------------------

    def test_variable_forward(self):
        with sf.Graph() as g:
            x = sf.Variable(7)
            y = sf.Variable(-7)
            result = g.run(x)
            assert_almost_equal(x.value, 7)
            assert_almost_equal(result, 7)

            result = g.run(y)
            assert_almost_equal(y.value, -7)
            assert_almost_equal(result, -7)

            y.assign(12)
            result = g.run(y)
            assert_almost_equal(y.value, 12)
            assert_almost_equal(result, 12)

    def test_constant_forward(self):
        with sf.Graph() as g:
            x = sf.Constant(7)
            y = sf.Constant(-7)
            result = g.run(x)
            assert_almost_equal(x.value, 7)
            assert_almost_equal(result, 7)

            result = g.run(y)
            assert_almost_equal(y.value, -7)
            assert_almost_equal(result, -7)

    def test_placeholder_forward(self):
        with sf.Graph() as g:
            x = sf.Placeholder('x')
            y = sf.Placeholder('y')
            result = g.run(x, feed_dict={'x': 3, 'y': 4})
            assert_almost_equal(x.value, 3)
            assert_almost_equal(result, 3)

            result = g.run(y, feed_dict={'x': 3, 'y': 4})
            assert_almost_equal(y.value, 4)
            assert_almost_equal(result, 4)

        # UNARY OPERATORS--------------------------------------

    def test_pow_forward(self):
        with sf.Graph() as g:
            x = sf.Variable(4)
            x2 = sf.Pow(x, 2)
            x3 = sf.Pow(x, 3)
            sqrt = sf.Pow(x, .5)

            result = g.run(x2)
            assert_almost_equal(x2.value, 16)
            assert_almost_equal(result, 16)

            result = g.run(x3)
            assert_almost_equal(x3.value, 64)
            assert_almost_equal(result, 64)

            result = g.run(sqrt)
            assert_almost_equal(sqrt.value, 2)
            assert_almost_equal(result, 2)

    def test_exp_forward(self):
        with sf.Graph() as g:
            x = sf.Variable(0)
            exp = sf.Exp(x)

            result = g.run(exp)
            assert_almost_equal(exp.value, 1)
            assert_almost_equal(result, 1)

            x.assign(1)
            result = g.run(exp)
            assert_almost_equal(exp.value, np.e)
            assert_almost_equal(result, np.e)

            x.assign(2.5)
            result = g.run(exp)
            assert_almost_equal(exp.value, np.e ** 2.5)
            assert_almost_equal(result, np.e ** 2.5)

    def test_log_forward(self):
        with sf.Graph() as g:
            x = sf.Variable(1)
            log = sf.Log(x)

            result = g.run(log)
            assert_almost_equal(log.value, 0)
            assert_almost_equal(result, 0)

            x.assign(np.e)
            result = g.run(log)
            assert_almost_equal(log.value, 1)
            assert_almost_equal(result, 1)

            x.assign(2.5)
            result = g.run(log)
            assert_almost_equal(log.value, np.log(2.5))
            assert_almost_equal(result, np.log(2.5))

    def test_abs_forward(self):
        with sf.Graph() as g:
            x = sf.Variable(0)
            abs_node = sf.Abs(x)

            result = g.run(abs_node)
            assert_almost_equal(abs_node.value, 0)
            assert_almost_equal(result, 0)

            x.assign(1.5)
            result = g.run(abs_node)
            assert_almost_equal(abs_node.value, 1.5)
            assert_almost_equal(result, 1.5)

            x.assign(-2.5)
            result = g.run(abs_node)
            assert_almost_equal(abs_node.value, 2.5)
            assert_almost_equal(result, 2.5)

    # BINARY OPERATORS--------------------------------------

    def test_add_forward(self):
        with sf.Graph() as g:
            x = sf.Variable(1)
            y = sf.Variable(2)
            total = sf.Add(x, y)

            for x_val in np.linspace(-5, 4, 10):
                for y_val in np.linspace(-5, 4, 10):
                    x.assign(x_val)
                    y.assign(y_val)

                result = g.run(total)
                assert_almost_equal(total.value, x_val + y_val)
                assert_almost_equal(result, x_val + y_val)

    def test_sub_forward(self):
        with sf.Graph() as g:
            x = sf.Variable(1)
            y = sf.Variable(2)
            total = sf.Subtract(x, y)

            for x_val in np.linspace(-5, 4, 10):
                for y_val in np.linspace(-5, 4, 10):
                    x.assign(x_val)
                    y.assign(y_val)

                result = g.run(total)
                assert_almost_equal(total.value, x_val - y_val)
                assert_almost_equal(result, x_val - y_val)

    def test_multiply_forward(self):
        with sf.Graph() as g:
            x = sf.Variable(1)
            y = sf.Variable(2)
            total = sf.Multiply(x, y)

            for x_val in np.linspace(-5, 4, 10):
                for y_val in np.linspace(-5, 4, 10):
                    x.assign(x_val)
                    y.assign(y_val)

                result = g.run(total)
                assert_almost_equal(total.value, x_val * y_val)
                assert_almost_equal(result, x_val * y_val)

    def test_divide_forward(self):
        with sf.Graph() as g:
            x = sf.Variable(1)
            y = sf.Variable(2)
            total = sf.Divide(x, y)

            for x_val in np.linspace(-5, 4, 10):
                for y_val in np.linspace(-5, 4, 10):
                    x.assign(x_val)
                    y.assign(y_val)

                result = g.run(total)
                assert_almost_equal(total.value, x_val / y_val)
                assert_almost_equal(result, x_val / y_val)


class BackwardTestCase(unittest.TestCase):

    def test_in_class_example_backward(self):
        graph = in_class_example_graph()
        sqrt = graph.nodes_by_name['out']
        result = graph.run(sqrt, feed_dict={'x': 3, 'y': 4},
                           compute_derivatives=True)

        assert_almost_equal(graph.nodes_by_name['x'].derivative, .6)
        assert_almost_equal(graph.nodes_by_name['y'].derivative, .8)

    def test_video_example_backward_all_values(self):
        graph = video_example_graph()
        loss = graph.nodes_by_name['loss']
        result = graph.run(loss, feed_dict={'w': 1, 'x': 2, 'y': 3},
                           compute_derivatives=True)
        assert_almost_equal(graph.nodes_by_name['w'].derivative, -2)
        assert_almost_equal(graph.nodes_by_name['x'].derivative, -2)
        assert_almost_equal(graph.nodes_by_name['y'].derivative, 2)
        assert_almost_equal(graph.nodes_by_name['wx'].derivative, -2)
        assert_almost_equal(graph.nodes_by_name['wx_minus_y'].derivative, -2)
        assert_almost_equal(
            graph.nodes_by_name['wx_minus_y_squared'].derivative, 1)
        assert_almost_equal(graph.nodes_by_name['w2'].derivative, 1)

        result = graph.run(loss, feed_dict={'w': 2, 'x': 3, 'y': 4},
                           compute_derivatives=True)
        assert_almost_equal(graph.nodes_by_name['w'].derivative, 16)

        result = graph.run(loss, feed_dict={'w': -2, 'x': 3, 'y': 4},
                           compute_derivatives=True)
        assert_almost_equal(graph.nodes_by_name['w'].derivative, -64)

    def test_cross_entropy_backward(self):
        graph = cross_entropy_graph()
        loss = graph.nodes_by_name['loss']
        w = graph.nodes_by_name['w']

        graph.run(loss, feed_dict={'x': 1, 'w': 0, 'y': 1},
                  compute_derivatives=True)
        assert_almost_equal(w.derivative, -.5)

        graph.run(loss, feed_dict={'x': 1, 'w': 0, 'y': 0},
                  compute_derivatives=True)
        assert_almost_equal(w.derivative, .5)

        graph.run(loss, feed_dict={'x': 1, 'w': 1, 'y': 1},
                  compute_derivatives=True)
        assert_almost_equal(w.derivative, -0.26894142136999516)

        graph.run(loss, feed_dict={'x': 1, 'w': -1, 'y': 1},
                  compute_derivatives=True)
        assert_almost_equal(w.derivative, -0.7310585786300049)

    # ------------------------------------------------------
    # backward TESTS FOR INDIVIDUAL OPERATORS
    # ------------------------------------------------------

    # UNARY OPERATORS--------------------------------------

    def test_pow_backward(self):
        with sf.Graph() as g:
            x = sf.Variable(4)
            x2 = sf.Pow(x, 2)
            x3 = sf.Pow(x, 3)
            sqrt = sf.Pow(x, .5)

            result = g.run(x2, compute_derivatives=True)
            assert_almost_equal(x.derivative, 8)

            result = g.run(x3, compute_derivatives=True)
            assert_almost_equal(x.derivative, 48)

            result = g.run(sqrt, compute_derivatives=True)
            assert_almost_equal(x.derivative, .25)

    def test_exp_backward(self):
        with sf.Graph() as g:
            x = sf.Variable(0)
            exp = sf.Exp(x)

            result = g.run(exp, compute_derivatives=True)
            assert_almost_equal(x.derivative, np.exp(0))

            x.assign(1)
            result = g.run(exp, compute_derivatives=True)
            assert_almost_equal(x.derivative, np.exp(1))

            x.assign(2.5)
            result = g.run(exp, compute_derivatives=True)
            assert_almost_equal(x.derivative, np.exp(2.5))

    def test_log_backward(self):
        with sf.Graph() as g:
            x = sf.Variable(1)
            log = sf.Log(x)

            result = g.run(log, compute_derivatives=True)
            assert_almost_equal(x.derivative, 1)

            x.assign(np.e)
            result = g.run(log, compute_derivatives=True)
            assert_almost_equal(x.derivative, 1 / np.e)

            x.assign(2.5)
            result = g.run(log, compute_derivatives=True)
            assert_almost_equal(x.derivative, .4)

    def test_abs_backward(self):
        with sf.Graph() as g:
            x = sf.Variable(1.5)
            abs_node = sf.Abs(x)

            g.run(abs_node, compute_derivatives=True)
            assert_almost_equal(x.derivative, 1)

            x.assign(-2.5)
            g.run(abs_node, compute_derivatives=True)
            assert_almost_equal(x.derivative, -1)

    # BINARY OPERATORS--------------------------------------

    def test_add_backward(self):
        with sf.Graph() as g:
            x = sf.Variable(1)
            y = sf.Variable(2)
            total = sf.Add(x, y)

            for x_val in np.linspace(-5, 4, 10):
                for y_val in np.linspace(-5, 4, 10):
                    x.assign(x_val)
                    y.assign(y_val)

                result = g.run(total, compute_derivatives=True)
                assert_almost_equal(x.derivative, 1.0)
                assert_almost_equal(y.derivative, 1.0)

    def test_sub_backward(self):
        with sf.Graph() as g:
            x = sf.Variable(1)
            y = sf.Variable(2)
            total = sf.Subtract(x, y)

            for x_val in np.linspace(-5, 4, 10):
                for y_val in np.linspace(-5, 4, 10):
                    x.assign(x_val)
                    y.assign(y_val)

                result = g.run(total, compute_derivatives=True)
                assert_almost_equal(x.derivative, 1.0)
                assert_almost_equal(y.derivative, -1.0)

    def test_multiply_backward(self):
        with sf.Graph() as g:
            x = sf.Variable(1)
            y = sf.Variable(2)
            total = sf.Multiply(x, y)

            for x_val in np.linspace(-5, 4, 10):
                for y_val in np.linspace(-5, 4, 10):
                    x.assign(x_val)
                    y.assign(y_val)

                result = g.run(total, compute_derivatives=True)
                assert_almost_equal(x.derivative, y_val)
                assert_almost_equal(y.derivative, x_val)

    def test_divide_backward(self):
        with sf.Graph() as g:
            x = sf.Variable(1)
            y = sf.Variable(2)
            total = sf.Divide(x, y)

            for x_val in np.linspace(-5, 4, 10):
                for y_val in np.linspace(-5, 4, 10):
                    x.assign(x_val)
                    y.assign(y_val)

                result = g.run(total, compute_derivatives=True)
                assert_almost_equal(x.derivative, 1 / y_val)
                assert_almost_equal(y.derivative, -x_val / y_val ** 2)


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(ForwardTestCase)
    runner = unittest.TextTestRunner()
    print("FORWARD TESTS:")
    runner.run(suite)

    suite = loader.loadTestsFromTestCase(BackwardTestCase)
    runner = unittest.TextTestRunner()
    print("\n\nBACKWARD TESTS:")
    runner.run(suite)