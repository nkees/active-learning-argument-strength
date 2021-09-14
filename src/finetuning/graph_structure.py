from collections import defaultdict
from copy import deepcopy


class Graph(object):

    """
    Graph data structure for representing transitive relations
    between arguments, modelled as graphs, as well as operations on them.
    """

    def __init__(self, *tuples):
        # a graph made with lists and dictionaries
        self.elements = defaultdict(list)
        for n in tuples:
            if type(n) is set:
                for tuple in n:
                    self.elements[tuple[0]].append(tuple[1])
            else:
                self.elements[n[0]].append(n[1])
        self.n = len(self.unique_elements_set())

    def __repr__(self):
        return f"elements={self.elements}"

    def copy(self):
        return deepcopy(self)

    def add_edge(self, higher_element, lower_element):
        """
        Adds a new argument pair to the graph.
        Args:
            higher_element: a stronger argument m in m>n
            lower_element: a weaker argument n in m>n
        """
        if higher_element not in self.unique_elements_set():
            self.n += 1
        if lower_element not in self.unique_elements_set():
            self.n += 1
        self.elements[higher_element].append(lower_element)

    def find_path(self, start, end, path=[]):
        """
        Finds a path between two given arguments (start and end)
        Args:
            start: argument of interest
            end: a second argument of interest

        Returns: a list of arguments lying in between

        """
        path = path + [start]
        if start == end:
            return path
        for node in self.elements[start]:
            if node not in path:
                newpath = self.find_path(node, end, path)
                if newpath:
                    return newpath
        return None

    def view_as_tuples(self):
        """
        Displays the graph as tuples of all possible argument pairs
        to be derived from the graph
        Returns: a tuple list

        """
        tuple_list = []
        for i in self.elements:
            for node in self.elements[i]:
                tuple_list.append((i, node))
        return tuple_list

    def connected(self, element1, element2):
        """
        Checks if two arguments are connected by a transitive relation
        Args:
            element1: argument of interest
            element2: another argument of interest

        Returns:

        """
        down_direction = self.find_path(element1, element2, path=[])
        up_direction = self.find_path(element2, element1, path=[])
        if down_direction or up_direction:
            return True
        else:
            return False

    def is_in_graph(self, element1, element2):
        """
        Checks if an argument pair is already contained in the graph
        (also through a transitive relation)
        Args:
            element1: argument of interest
            element2: another argument of interest
        """
        if self.connected(element1, element2):
            return True
        else:
            return False

    def push_argument_pair_to_graph(self, argument1, argument2):
        """
        Adds an argument pair to the graph; order matters!
        Args:
            element1: stronger argument m in m>n
            element2: weaker argument n in m>n

        Returns:

        """
        self.add_edge(argument1, argument2)

    def push_argument_pairs_from_df_to_graph(self, from_df):
        """
        Displays the argument pairs from the data frame as a graph
        Args:
            from_df: data frame with the argument pairs

        Returns:

        """
        for i in range(len(from_df)):
            row = from_df.loc[[i]]
            argument1, argument2 = retrieve_argument_ids(row)
            label = row["label"].iloc[0]
            if label == "a1":
                self.push_argument_pair_to_graph(argument1, argument2)
            elif label == "a2":
                self.push_argument_pair_to_graph(argument2, argument1)
            else:
                raise AssertionError(f"Such label ({label}) does not exist")

    def add_ordered_edges(self, from_graph, reference_df):
        """
        Adds some new argument pairs from an unordered graph to the existing graphs,
        taking the labels in the reference data frame as a strength relation
        indicator.
        Args:
            from_graph: unordered argument pairs source
            reference_df: reference data frame

        Returns:

        """
        argument_pairs = from_graph.view_as_tuples()
        for argument_pair in argument_pairs:
            argument1 = argument_pair[0]
            argument2 = argument_pair[1]
            id_ = str(argument1) + "_" + str(argument2)
            if id_ in reference_df["id"].tolist():
                row = reference_df[reference_df["id"] == id_]
                label = row["label"].iloc[0]
                if label == "a1":
                    self.push_argument_pair_to_graph(argument1, argument2)
                elif label == "a2":
                    self.push_argument_pair_to_graph(argument2, argument1)
                else:
                    raise AssertionError(f"Such label ({label}) does not exist")

    def unique_elements_set(self):
        """
        Returns a set of unique elements in the graph.

        """
        elements = list(item[0] for item in self.view_as_tuples())
        second_elements = list(item[1] for item in self.view_as_tuples())
        elements.extend(second_elements)
        return set(elements)

    def length(self):
        return len(self.view_as_tuples())

    def transitive_closure(self):
        """
        Creates a transitive closure of all the nodes in the graph.
        Returns: transitive closure as a Graph

        """
        graph_tuples = self.view_as_tuples()
        closure = set(graph_tuples)
        while True:
            new_relations = set((x, w) for x, y in closure for q, w in closure if q == y)
            closure_until_now = closure | new_relations
            if len(closure_until_now) == len(closure):
                break
            closure = closure_until_now
        empty = Graph()
        for n in closure:
            empty.add_edge(n[0], n[1])
        closure_graph = Graph(closure)
        return closure_graph


def retrieve_argument_ids(row):
    argument1 = row["id"].iloc[0].split("_")[0]
    argument2 = row["id"].iloc[0].split("_")[1]
    return argument1, argument2



