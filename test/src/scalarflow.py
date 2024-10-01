"""
######################################################
ScalarFlow: A Python Automatic Differentiation Library
######################################################

ScalarFlow is a  automatic differentiation library that borrows heavily from
the TensorFlow 1.0 API.  The emphasis is on simplicity, not performance.


============================================
Building and Running Computation Graphs
============================================

The ScalarFlow library makes it possible to build and execute computation
graphs involving scalar quantities.  For example::

    import scalarflow as sf

    with sf.Graph() as g:
        a = sf.Constant(7.0)
        b = sf.Constant(3.0)
        sum = sf.Add(a, b)

        result = g.run(sum)
        print(result) # Prints 10.0

Notice that in the example above, the nodes are added to the graph
``g``, even though it is not  provided as an explicit argument when
constructing nodes in the computation graph.  The scalarflow library
maintains a default computation graph that can be set using the ``with``
keyword as above.

It is also possible to use the default computation computation graph outside
of any context.  For example::

    a = sf.Constant(7.0)
    b = sf.Constant(3.0)
    sum = sf.Add(a, b)

    result = sf.get_current_graph.run(sum)
    print(result) # Prints 10.0

============================================
Node names
============================================

All nodes in the computation graph must have unique names.  If the ``name``
argument is not provided a default name will be selected based on the node
type::

    x = sf.Constant(3.0, name='input')
    squared = sf.Pow(x, 2.0)

    print(x.name)       # Prints "input"
    print(squared.name) # Prints "Pow_0"

============================================
Variables and Placeholders
============================================

In addiction to Constants, scalarflow includes two scalar-type Nodes:
Variables and Placeholders.

============================================
Variables
============================================

Variables behave like constants, but their values may be set directly using
the ``assign`` method::

    with sf.Graph() as g:
        x = sf.Variable(4.0)
        sqrt = sf.Pow(x, .5)
        print(g.run(sqrt)) # Prints 2.0

        x.assign(25.0)
        print(g.run(sqrt)) # Prints 5.0

Variables are useful as trainable parameters in machine learning applications.

============================================
Placeholders
============================================

Placeholders must be assigned a value when ``run`` is called on the graph::

    with sf.Graph() as g:
        x = sf.Constant(4.0)
        y = sf.Placeholder(name='y')
        sum = sf.Add(x, y)

        print(g.run(sum, feed_dict={'y': 5.0})) # Prints 9.0
        print(g.run(sum, feed_dict={'y': 10.0})) # Prints 14.0

Here, ``feed_dict`` is a dictionary that maps from placeholder node names to the
value that should be used in the requested computation.  Placeholder nodes
are useful for representing inputs and outputs in machine learning training
algorithms.

============================================
Node values
============================================

The ``run`` method of the graph will only execute the subset of nodes that are
ancestors of the requested output.  As a side effect, all of the values of
those nodes are cached and are available through the ``value`` attribute::

    with sf.Graph() as g:
        a = sf.Constant(7.0, name='a')
        b = sf.Constant(3.0, name='b')
        sum = sf.Add(a, b)
        sum_sqrd = sf.Pow(sum, 2.0)
        sum_to_fourth = sf.Pow(sum_sqrd, 2.0)

        g.run(sum_sqrd)

        print(sum.value) # prints 10.0, sum was computed!
        print(sum_to_fourth.value) # Something terrible happens, never computed.

============================================
Node derivatives
============================================

If the ``compute_derivatives`` argument is True, then ``run`` perform both a
forward and backward pass.  After the backward pass completes, partial
derivatives will be available through the ``derivative`` attribute of each
node that is involved in the computation::

    with sf.Graph() as g:
        x = sf.Constant(4.0)
        y = sf.Pow(x, 2.0)  # y = x^2

        print(g.run(y)) # prints 16.0
        print(x.derivative) # prints dy/dx = 2x = 8.0

"""
import networkx.algorithms.dag as dag
import networkx as nx
import math


class Graph:
    """
    Computation Graph

    A computation graph is a directed acyclic graph that represents a numerical
    computation performed on scalar-valued inputs. This class supports
    forward computations as well as reverse-mode automatic differentiation.

    The Graph class also acts as a Python context manager, supporting the
    ``with`` keyword.  For example::

        with sf.Graph() as g:
            a = sf.Constant(1.0)
            b = sf.Constant(2.0)
            c = sf.Add(a, b)
            result = g.run(c)

    Attributes:

        nodes_by_name: a dictionary mapping from unique names to the
                       corresponding node

    """

    def __init__(self):
        """ Empty Computation Graph.

        """
        self._graph = nx.DiGraph()

        # Cached mapping from nodes to a topologically sorted list of
        # their ancestors
        self._ancestor_lists = dict()

        # mapping from names to nodes
        self.nodes_by_name = dict()

        # used to save the previous default graph when this graph
        # is in context.
        self._old_graph = None

    def __enter__(self):
        """  Enter a context for this graph.  This graph will become the
        default graph until we exit this context. The previous
        default graph will be automatically be restored on __exit__.

        Returns: The graph

        """
        global _GRAPH
        self._old_graph = _GRAPH
        _GRAPH = self
        return _GRAPH

    def __exit__(self, exc_type, exc_val, exc_tb):
        """  Exit the context for this graph.

        The previous default graph is restored.
        """
        global _GRAPH
        _GRAPH = self._old_graph

    def _add_node(self, node):
        """ Add a new node to the graph. (All nodes must have unique names.)

        Args:
            node: The node to add
        """
        if node.name in self.nodes_by_name:
            raise ValueError("Duplicate node name: {}".format(node.name))

        self._graph.add_node(node)

        self.nodes_by_name[node.name] = node

        # clear ancestor cache on graph modification
        self._ancestor_lists = dict()

    def _add_edge(self, node1, node2):
        """  Add a directed edge between node1 and node2

        Args:
            node1: Start node
            node2: End node
        """
        self._graph.add_edge(node1, node2)

        # clear ancestor cache on graph modification
        self._ancestor_lists = dict()

    def gen_dot(self, filename, show_value=True, show_derivative=True):
        """  Write a dot file representing the structure and contents of
        the graph.

        The .dot file is a standard text-based format for representing graph
        structures.  The graphviz library can convert dot files to images in
        many formats.   There are also web-based tools for visualizing dot
        files.  E.g. http://www.webgraphviz.com/

        Args:
            filename:  Name of the file to create
            show_value:  Show evaluated node values in the graph
            show_derivative:  Show evaluated node derivatives in the graph

        """
        with open(filename, 'w') as f:
            f.write('digraph scalarflow {\nrankdir="LR"\n')

            for node in self._graph.nodes:
                f.write('{} [label="{} '.format(node.name, repr(node)))
                if show_value and node.value is not None:
                    f.write('\\nvalue:  {:g}'.format(node.value))
                if show_derivative and node.derivative is not None:
                    f.write('\\ngrad: {:g}'.format(node.derivative))
                f.write('"]\n'.format(node.name, repr(node)))

            for edge in self._graph.edges:
                f.write('   {} -> {}\n'.format(edge[0].name, edge[1].name))
            f.write('}\n')

    def _ancestor_list(self, node):
        """ Return a topologically sorted list containing a node's ancestors.

        """

        # Finding and and sorting the list of ancestors is likely to be
        # a relatively expensive operation, and the same list is likely
        # to be used many times when a graph is used for training.  Therefore,
        # we cache the lists and recalculate them only if the graph has
        # changed.
        if node not in self._ancestor_lists:
            nodes = dag.ancestors(self._graph, node)
            subgraph = self._graph.subgraph(nodes)
            self._ancestor_lists[node] = list(dag.topological_sort(subgraph))

        return self._ancestor_lists[node]

    def run(self, node, feed_dict=None, compute_derivatives=False):
        """  Run the computation graph and return the value of the
        indicated node.

        After this method is called the ``value`` attribute of all the node
        and all of its ancestors will be set. The ``value`` attributes of
        non-ancestors are not defined.

        If ``compute_derivatives`` is true, this method will also perform a
        backward pass to determine the numerical derivatives for the indicated
        node with respect to every ancestor.  The derivatives will be
        accessible from the ``derivative`` attribute of the nodes.  The
        derivatives of non-ancestors are not defined.

        Args:
            node:       Determine the value of this node
            feed_dict:  A dictionary mapping from Placeholder node names to
                        values.  E.g. {'x': 1.0, 'y': 2.0}.
            compute_derivatives:  True if we should perform a backward pass
                                  to compute partial derivatives.

        Returns:  The numeric value of of the indicated node.

        """

    # UNFINISHED!!
    pass


# Construct a default computation graph.
_GRAPH = Graph()


def get_current_graph():
    """ Return the currently active computation graph.

    Inside of a graph context this will return the graph associated with that
    that context.  Outside of any context this will return the default graph.

    """
    return _GRAPH


# ABSTRACT NODE CLASSES---------------------

class Node:
    """ Abstract base class for all nodes in a computation graph.

    Attributes:
        value:  The most recently calculated value for this node. This is
                undefined if the node has never been involved in a computation.
        derivative: The most recently calculated partial derivative for this
                    node.  This is undefined if the node has not been involved
                    in a backward pass.
        name (string):  The name of this node.  All nodes must have a unique
                        name.
    """

    def __init__(self, name):
        """ The abstract node constructor handles naming and inserting the
        node into the graph.

        Args:
            name (string):  Name for this node.  If this is an empty string
                            then a unique name will be generated based on the
                            node's class.
        """
        # Use the node's actual class to generate an appropriate name.
        if name == "":
            name = "{}_{}".format(self.__class__.__name__,
                                  self.__class__._COUNT)
            self.__class__._COUNT += 1

        self.name = name
        _GRAPH._add_node(self)

    @property
    def value(self):
        """ Value should be read-only (except for variable nodes). """
        # UNFINISHED!!
        return None

    @property
    def derivative(self):
        """ derivative should be read-only. """
        # UNFINISHED!!
        return None

    def __repr__(self):
        """ Default string representation is the Node's name. """
        return self.name


class BinaryOp(Node):
    """ Abstract base class for all nodes representing binary operators"""

    def __init__(self, operand1, operand2, name):
        """ BinaryOp constructor handles updating the graph structure and
        storing the operands.

        Args:
            operand1: Node representing the first operand
            operand2: Node representing the second operand
            name (string): node name
        """
        super().__init__(name)
        self.operand1 = operand1
        self.operand2 = operand2
        _GRAPH._add_edge(operand1, self)
        _GRAPH._add_edge(operand2, self)


class UnaryOp(Node):
    """ Abstract base class for all nodes representing unary operators"""

    def __init__(self, operand, name):
        """ UnaryOp constructor handles updating the graph structure and storing
        the operand.

        Args:
            operand: Node representing the operand for this node
            name (string): Name for this node.
        """
        super().__init__(name)
        self.operand = operand
        _GRAPH._add_edge(operand, self)


# INPUT NODES ------------------------

class Variable(Node):
    """ Variable.  A node that can be assigned a value. """
    _COUNT = 0

    def __init__(self, value, name=""):
        """  Variable

        Args:
            value: Initial value of this variable
            name: Variable name
        """
        super().__init__(name)

    def assign(self, value):
        """ Assign a new value to this variable

        """
        # UNFINISHED!
        pass


class Constant(Node):
    """ Constants behave like Variables that cannot be assigned values
    after they are created. """
    _COUNT = 0

    def __init__(self, value, name=""):
        super().__init__(name)
        self._value = value

    def __repr__(self):
        return self.name + ": " + str(self._value)


class Placeholder(Node):
    """  Placeholders behave like Variables that can only be assigned values
    by including an appropriate value in the feed_dict passed to ``run``.
    """
    _COUNT = 0

    def __init__(self, name=""):
        super().__init__(name)


# BINARY OPERATORS ---------------------

class Add(BinaryOp):
    """ Addition.  Node representing operand1 + operand2."""
    _COUNT = 0

    def __init__(self, operand1, operand2, name=""):
        super().__init__(operand1, operand2, name)


class Subtract(BinaryOp):
    """ Subtraction.  Node representing operand1 - operand2. """
    _COUNT = 0

    def __init__(self, operand1, operand2, name=""):
        super().__init__(operand1, operand2, name)


class Multiply(BinaryOp):
    """ Multiplication.  Node representing operand1 * operand2."""
    _COUNT = 0

    def __init__(self, operand1, operand2, name=""):
        super().__init__(operand1, operand2, name)


class Divide(BinaryOp):
    """ Division.  Node representing operand1 / operand2.  """
    _COUNT = 0

    def __init__(self, operand1, operand2, name=""):
        super().__init__(operand1, operand2, name)


# UNARY OPERATORS --------------------

class Pow(UnaryOp):
    """ Power  E.g. operand^2 or operand^3"""
    _COUNT = 0

    def __init__(self, operand, power, name=""):
        """  Construct a Pow node

        Args:
            operand: The operand
            power: The power to raise the operand to
            name:  Name for this node
        """

        super().__init__(operand, name)


class Exp(UnaryOp):
    """ Exponential node:  e^operand
    """
    _COUNT = 0

    def __init__(self, operand, name=""):
        super().__init__(operand, name)


class Log(UnaryOp):
    """ Log base e. """
    _COUNT = 0

    def __init__(self, operand, name=""):
        super().__init__(operand, name)


class Abs(UnaryOp):
    """ Absolute Value.  |operand| """
    _COUNT = 0

    def __init__(self, operand, name=""):
        super().__init__(operand, name)


def main():
    pass


if __name__ == "__main__":
    main()