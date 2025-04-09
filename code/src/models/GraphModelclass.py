class GraphModel:
    """
    GraphModel class containing the parameters of the graph model

    Attributes:
        name (str): Name of the graph model
        type (str): Type of graph
        param (dict): Parameters
        typegraph (str): Type of graph ('undirected', 'simple', 'bipartite')
    """

    def __init__(self, type, *args):
        """
        Constructor for GraphModel class.

        Args:
        ------
            type (str): Type of the graph model
            *args: Variable number of arguments based on the type of the graph model
        """
        self.name = None
        self.type = type
        self.param = {}
        self.typegraph = 'undirected'

        self._create_graph_model(*args)

    def _create_graph_model(self, *args):
        """
        Create a graph model based on the provided arguments.

        Args:
        ------
            *args: Variable number of arguments based on the type of the graph model
        """
        if self.type == 'ER':
            self.param['n'] = args[0]
            self.param['p'] = args[1]
            self.name = 'Erdos-Renyi'

        elif self.type == 'GG':
            self.name = 'Generalized gamma process'
            self.param['alpha'] = args[0]
            self.param['sigma'] = args[1]
            self.param['tau'] = args[2]

            if len(args) == 4:
                self.typegraph = args[3]
                if self.typegraph == 'bipartite':
                    if isinstance(args[0], (int, float)):
                        self.param['alpha'] = [args[0], args[0]]
                    if isinstance(args[1], (int, float)):
                        self.param['sigma'] = [args[1], args[1]]
                    if isinstance(args[2], (int, float)):
                        self.param['tau'] = [args[2], args[2]]

        elif self.type == 'Rapid':
            self.name = 'Rapidly Varying'
            self.param['alpha'] = args[0]
            self.param['tau'] = args[1]
            self.param['beta'] = args[2]
            self.param['c'] = args[3]
            self.param['eta'] = args[4]

        elif self.type == 'BA':
            self.param['n'] = args[0]
            self.name = 'Barabasi-Albert'

        elif self.type == 'Lloyd':
            self.name = 'Lloyd'
            self.param['n'] = args[0]
            self.param['sig'] = args[1]
            self.param['c'] = args[2]
            self.param['d'] = args[3]

        else:
            raise ValueError('Unknown type {}'.format(self.type))

    def graphrnd(self, *args):
        """
        Sample a graph.

        Args:
        ------   
            *args: Variable number of arguments
        """

        pass
