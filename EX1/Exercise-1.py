from collections import defaultdict

import numpy as np


class Variable:
    def __init__(self, name, no_states, table, parents=[], no_parent_states=[]):
        """
        name (string): Name of the variable
        no_states (int): Number of states this variable can take
        table (list or Array of reals): Conditional probability table (see below)
        parents (list of strings): Name for each parent variable.
        no_parent_states (list of ints): Number of states that each parent variable can take.

        The table is a 2d array of size #events * #number_of_conditions.
        #number_of_conditions is the number of possible conditions (prod(no_parent_states))
        If the distribution is unconditional #number_of_conditions is 1.
        Each column represents a conditional distribution and sum to 1.

        Here is an example of a variable with 3 states and two parents cond0 and cond1,
        both with 2 possible states.
        +----------+----------+----------+----------+----------+----------+----------+
        |  cond0   | cond0(0) | cond0(1) | cond0(0) | cond0(1) | cond0(0) | cond0(1) |
        +----------+----------+----------+----------+----------+----------+----------+
        |  cond1   | cond1(0) | cond1(0) | cond1(0) | cond1(1) | cond1(1) | cond1(1) |
        +----------+----------+----------+----------+----------+----------+----------+
        | event(0) |  0.2000  |  0.2000  |  0.7000  |  0.0000  |  0.2000  |  0.4000  |
        +----------+----------+----------+----------+----------+----------+----------+
        | event(1) |  0.3000  |  0.8000  |  0.2000  |  0.0000  |  0.2000  |  0.4000  |
        +----------+----------+----------+----------+----------+----------+----------+
        | event(2) |  0.5000  |  0.0000  |  0.1000  |  1.0000  |  0.6000  |  0.2000  |
        +----------+----------+----------+----------+----------+----------+----------+

        To create this table you would use the following parameters:

        Variable('event', 3, [[0.2, 0.2, 0.7, 0.0, 0.2, 0.4],
                              [0.3, 0.8, 0.2, 0.0, 0.2, 0.4],
                              [0.5, 0.0, 0.1, 1.0, 0.6, 0.2]],
                 parents=['cond0', 'cond1'],
                 no_parent_states=[2, 2])
        """
        self.name = name
        self.no_states = no_states
        self.table = np.array(table)
        self.parents = parents
        self.no_parent_states = no_parent_states

        if self.table.shape[0] != self.no_states:
            raise ValueError(f"Number of states and number of rows in table must be equal. "
                             f"Recieved {self.no_states} number of states, but table has "
                             f"{self.table.shape[0]} number of rows.")

        if self.table.shape[1] != np.prod(no_parent_states):
            raise ValueError("Number of table columns does not match number of parent states combinations.")

        if not np.allclose(self.table.sum(axis=0), 1):
            print(self.table.sum(axis=0))
            raise ValueError("All columns in table must sum to 1.")

        if len(parents) != len(no_parent_states):
            raise ValueError("Number of parents must match number of length of list no_parent_states.")

    def __str__(self):
        """
        Pretty string for the table distribution
        For printing to display properly, don't use variable names with more than 7 characters
        """
        width = int(np.prod(self.no_parent_states))
        grid = np.meshgrid(*[range(i) for i in self.no_parent_states])
        s = ""
        for (i, e) in enumerate(self.parents):
            s += '+----------+' + '----------+' * width + '\n'
            gi = grid[i].reshape(-1)
            s += f'|{e:^10}|' + '|'.join([f'{e + "("+str(j)+")":^10}' for j in gi])
            s += '|\n'

        for i in range(self.no_states):
            s += '+----------+' + '----------+' * width + '\n'
            state_name = self.name + f'({i})'
            s += f'|{state_name:^10}|' + '|'.join([f'{p:^10.4f}' for p in self.table[i]])
            s += '|\n'

        s += '+----------+' + '----------+' * width + '\n'

        return s

    def probability(self, state, parentstates):
        """
        Returns probability of variable taking on a "state" given "parentstates"
        This method is a simple lookup in the conditional probability table, it does not calculate anything.

        Input:
            state: integer between 0 and no_states
            parentstates: dictionary of {'parent': state}
        Output:
            float with value between 0 and 1
        """
        if not isinstance(state, int):
            raise TypeError(f"Expected state to be of type int; got type {type(state)}.")
        if not isinstance(parentstates, dict):
            raise TypeError(f"Expected parentstates to be of type dict; got type {type(parentstates)}.")
        if state >= self.no_states:
            raise ValueError(f"Recieved state={state}; this variable's last state is {self.no_states - 1}.")
        if state < 0:
            raise ValueError(f"Recieved state={state}; state cannot be negative.")

        table_index = 0
        for variable in self.parents:
            if variable not in parentstates:
                raise ValueError(f"Variable {variable.name} does not have a defined value in parentstates.")

            var_index = self.parents.index(variable)
            table_index += parentstates[variable] * np.prod(self.no_parent_states[:var_index])

        return self.table[state, int(table_index)]


class BayesianNetwork:
    """
    Class representing a Bayesian network.
    Nodes can be accessed through self.variables['variable_name'].
    Each node is a Variable.

    Edges are stored in a dictionary. A node's children can be accessed by
    self.edges[variable]. Both the key and value in this dictionary is a Variable.
    """
    def __init__(self):
        self.edges = defaultdict(lambda: [])  # All nodes start out with 0 edges
        self.variables = {}                   # Dictionary of "name":TabularDistribution

    def add_variable(self, variable):
        """
        Adds a variable to the network.
        """
        if not isinstance(variable, Variable):
            raise TypeError(f"Expected {Variable}; got {type(variable)}.")
        self.variables[variable.name] = variable

    def add_edge(self, from_variable, to_variable):
        """
        Adds an edge from one variable to another in the network. Both variables must have
        been added to the network before calling this method.
        """
        if from_variable not in self.variables.values():
            raise ValueError("Parent variable is not added to list of variables.")
        if to_variable not in self.variables.values():
            raise ValueError("Child variable is not added to list of variables.")
        self.edges[from_variable].append(to_variable)

    def sorted_nodes(self):
        """
        TODO: Implement Kahn's algorithm (or some equivalent algorithm) for putting
              variables in lexicographical topological order.
        Returns: List of sorted variable names.
        """
        # Empty list that will contain the sorted elements
        sorted_list = []

        # Get set of all nodes with no incoming edge
        s = []
        for node in self.variables.keys():
            edge_exists = False
            for edges in self.edges.values():
                for edge in edges:
                    if node == edge.name:
                        edge_exists = True
            if not edge_exists:
                s.append(node)

        while s:
            # Remove a node n from s and add it to the list
            n = s.pop()
            sorted_list.append(n)

            # Get child nodes and remove them from the graph
            children = self.edges[self.variables.get(n)]
            children = np.array(children)

            # Add child nodes to the list 
            for child in children:
                self.edges[self.variables[n]].remove(child)

                # If child has no other incoming edges, add it to the sorted list
                if not self.has_incoming_edges(child):
                    s.append(child.name)
                
                    
        
        # Check if graph is DAG (edges should be empty if it is not)   
        if not self.edges:
            return -1
        
        else:
            return sorted_list

    # Helper function to check if variable any incomming edges
    def has_incoming_edges(self, variable):
        exists = False

        for value in self.edges.values():
            if variable in value:
                exists = True
        
        return exists





class InferenceByEnumeration:
    def __init__(self, bayesian_network):
        self.bayesian_network = bayesian_network
        self.topo_order = bayesian_network.sorted_nodes()

    def _enumeration_ask(self, X, evidence):
        # Initialize distribution over X
        q = []

        for i in range(self.bayesian_network.variables.get(X).no_states):
            evidence_ext = evidence.copy()
            evidence_ext[X] = i
            q.append(self._enumerate_all(self.bayesian_network.variables, evidence_ext, self.topo_order))
        q = np.array(q)
        # return normalized result 
        return q/q.sum(0)  
        

    def _enumerate_all(self, variables, evidence, order):
        # Check if vars is empty 
        if not variables:
            return 1.0
        
        # Get first node from topological sorted order
        first = order[0]
        rest_order = order[1:]

        # Get rest of the nodes from topological sorted order
        rest = dict(variables)
        del rest[first]
        
        # Make parent dict for probability calculation
        parentstates = {}
        for parent in variables[first].parents:
            if parent in evidence.keys():
                parentstates[parent] = evidence[parent]

        if first in evidence.keys():
            return variables[first].probability(evidence[first], parentstates)*self._enumerate_all(rest, evidence, rest_order)
        else:
            summation = 0
            for i in range(variables[first].no_states):
                evidence_ext = evidence.copy()
                evidence_ext[first] = i
                summation += variables[first].probability(evidence_ext[first], parentstates)*self._enumerate_all(rest, evidence_ext, rest_order)
            
            return summation



    def query(self, var, evidence={}):
        """
        Wrapper around "_enumeration_ask" that returns a
        Tabular variable instead of a vector
        """
        q = self._enumeration_ask(var, evidence).reshape(-1, 1)
        return Variable(var, self.bayesian_network.variables[var].no_states, q)


def problem3c():
    d1 = Variable('A', 2, [[0.8], [0.2]])
    d2 = Variable('B', 2, [[0.5, 0.2],
                           [0.5, 0.8]],
                  parents=['A'],
                  no_parent_states=[2])
    d3 = Variable('C', 2, [[0.1, 0.3],
                           [0.9, 0.7]],
                  parents=['B'],
                  no_parent_states=[2])
    d4 = Variable('D', 2, [[0.6, 0.8],
                           [0.4, 0.2]],
                  parents=['B'],
                  no_parent_states=[2])

    print(f"Probability distribution, P({d1.name})")
    print(d1)

    print(f"Probability distribution, P({d2.name} | {d1.name})")
    print(d2)

    print(f"Probability distribution, P({d3.name} | {d2.name})")
    print(d3)

    print(f"Probability distribution, P({d4.name} | {d2.name})")
    print(d3)

    bn = BayesianNetwork()

    bn.add_variable(d1)
    bn.add_variable(d2)
    bn.add_variable(d3)
    bn.add_variable(d4)
    bn.add_edge(d1, d2)
    bn.add_edge(d2, d3)
    bn.add_edge(d2, d4)

    inference = InferenceByEnumeration(bn)
    
    #posterior = inference.query('B', {})
    #posterior = inference.query('D', {})
    #posterior = inference.query('C', {'D': 1})
    posterior = inference.query('A', {'C': 1, 'D': 0})

    print(f"Probability distribution, P({d3.name} | {d4.name})")
    print(posterior)


def monty_hall():
    h = [[0, 0, 0, 0, 0.5, 1.0, 0, 1.0, 0.5],
        [0.5, 0, 1.0, 0, 0, 0, 1.0, 0, 0.5],
        [0.5, 1.0, 0, 1.0, 0.5, 0, 0, 0, 0]]

    h = np.array(h)
    
    d1 = Variable('P', 3, [[1/3],[1/3],[1/3]])
    d2 = Variable('CBG',3, [[1/3],[1/3],[1/3]]) 
    d3 = Variable('OBH', 3, [[0, 0, 0, 0, 0.5, 1.0, 0, 1.0, 0.5],
                            [0.5, 0, 1.0, 0, 0, 0, 1.0, 0, 0.5],
                            [0.5, 1.0, 0, 1.0, 0.5, 0, 0, 0, 0]],
                parents=['P','CBG'],
                no_parent_states=[3, 3])
   

    print(f"Probability distribution, P({d1.name})")
    print(d1)

    print(f"Probability distribution, P({d3.name} | {d1.name})")
    print(d3)

    print(f"Probability distribution, P({d3.name} | {d2.name})")
    print(d3)


    bn = BayesianNetwork()

    bn.add_variable(d1)
    bn.add_variable(d2)
    bn.add_variable(d3)
    bn.add_edge(d1, d3)
    bn.add_edge(d2, d3)

    inference = InferenceByEnumeration(bn)
    posterior = inference.query('P', {'CBG': 0, 'OBH': 2})

    print(f"Probability distribution, P({d1.name} | {d2.name}, {d3.name})")
    print(posterior)



if __name__ == '__main__':
    #problem3c()
    monty_hall()
