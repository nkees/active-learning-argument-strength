import unittest
import pandas as pd
from src.finetuning.graph_structure import Graph, retrieve_argument_ids


class TestGraphStructureFunctions(unittest.TestCase):

    graph = Graph((1, 2), (3, 4), (3, 5))
    df = pd.DataFrame({"id": ["arg0_arg1", "arg2_arg3"], "label": ["a1", "a2"]}, columns=["id", "label"], index=[0, 1])

    def test_copy_and_add_edge(self, graph=graph):
        graph_test = graph.copy()
        self.assertEqual(graph.elements, graph_test.elements)
        graph_test.add_edge(3, 6)
        self.assertEqual(graph_test.elements, {1: [2], 3: [4, 5, 6]})
        self.assertNotEqual(graph.elements, graph_test.elements)

    def test_find_path(self, graph=graph):
        graph_test = graph.copy()
        self.assertFalse(graph_test.find_path(3, 6))
        self.assertTrue(graph_test.find_path(3, 5))
        self.assertIsNone(graph_test.find_path(6, 3))

    def test_connected(self, graph=graph):
        graph_test = graph.copy()
        self.assertTrue(graph_test.connected(3, 5))
        self.assertTrue(graph_test.connected(5, 3))
        self.assertFalse(graph_test.connected(3, 6))

    def test_is_in_graph(self, graph=graph):
        graph_test = graph.copy()
        graph_test.add_edge(2, 3)
        self.assertTrue(graph_test.connected(1, 3))
        self.assertTrue(graph_test.connected(3, 1))
        self.assertFalse(graph_test.connected(1, 6))

    def test_push_argument_pair_to_graph(self, graph=graph):
        graph_test = graph.copy()
        graph_test.push_argument_pair_to_graph("arg1", "arg2")
        self.assertEqual(graph_test.elements, {1: [2], 3: [4, 5], "arg1": ["arg2"]})

    def test_retrieve_argument_ids(self, df=df):
        row = df.loc[[1]]
        self.assertEqual(retrieve_argument_ids(row), ('arg2', 'arg3'))

    def test_push_argument_pairs_from_df_to_graph(self, graph=graph, df=df):
        graph_test = graph.copy()
        graph_test.push_argument_pairs_from_df_to_graph(df)
        self.assertEqual(graph_test.elements, {1: [2], 3: [4, 5], 'arg0': ['arg1'], 'arg3': ['arg2']})

    def test_view_as_tuples(self, graph=graph):
        test_graph = graph.copy()
        tuple_list = test_graph.view_as_tuples()
        self.assertEqual(tuple_list, [(1, 2), (3, 4), (3, 5)])

    def test_add_ordered_edges(self, graph=graph, df=df):
        from_graph = Graph(("arg0", "arg1"), ("arg2", "arg3"))
        to_graph = graph.copy()
        to_graph.add_ordered_edges(from_graph, df)
        self.assertEqual(to_graph.elements, {1: [2], 3: [4, 5], "arg0": ["arg1"], "arg3": ["arg2"]})


if __name__ == "__main__":
    unittest.main()
