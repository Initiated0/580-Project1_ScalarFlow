import scalarflow as sf

graph = sf.Graph()

with graph:
    x = sf.Variable(2.0, name='x')
    y = sf.Variable(4.0, name='y')

    x_squared = sf.Pow(x, 2)
    y_squared = sf.Pow(y, 2)

    xy_sum = sf.Add(x_squared, y_squared)

    func = sf.Pow(xy_sum, .5) # (Square root)

graph.gen_dot("sample.dot")


