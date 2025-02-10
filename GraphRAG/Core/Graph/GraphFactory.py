"""
Graph Factory.
"""
import os
from Core.Graph.BaseGraph import BaseGraph
from Core.Graph.ERGraph import ERGraph
from Core.Graph.PassageGraph import PassageGraph
from Core.Graph.TreeGraph import TreeGraph
from Core.Graph.TreeGraphBalanced import TreeGraphBalanced
from Core.Graph.RKGraph import RKGraph
from Core.Graph.ChunkGraph import ChunkGraph

class GraphFactory():
    def __init__(self):
        self.creators = {
            "er_graph": self._create_er_graph,
            "rkg_graph": self._create_rkg_graph,
            "tree_graph": self._create_tree_graph,
            "tree_graph_balanced": self._create_tree_graph_balanced,
            "passage_graph": self._create_passage_graph,
            "chunk_graph": self._create_chunk_graph,
        }


    def get_graph(self, config, **kwargs) -> BaseGraph:
        """Key is PersistType."""
        is_second_graph = kwargs.get('is_second_graph', False)
        if not is_second_graph:
            # Default graph type
            return self.creators[config.graph.graph_type](config, **kwargs)
        else:
            # Second graph type
            return self.creators[config.graph.second_graph_type](config, **kwargs)

    @staticmethod
    def _create_er_graph(config, **kwargs):
        return ERGraph(
            config.graph, **kwargs
        )

    @staticmethod
    def _create_rkg_graph(config, **kwargs):
        return RKGraph(config.graph, **kwargs)

    @staticmethod
    def _create_tree_graph(config, **kwargs):
        return TreeGraph(config, **kwargs)

    @staticmethod
    def _create_tree_graph_balanced(config, **kwargs):
        return TreeGraphBalanced(config, **kwargs)

    @staticmethod
    def _create_passage_graph(config, **kwargs):
        return PassageGraph(config.graph, **kwargs)
    
    @staticmethod
    def _create_chunk_graph(config, **kwargs):
        return ChunkGraph(config.graph, **kwargs)


get_graph = GraphFactory().get_graph
