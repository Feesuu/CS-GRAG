from Core.Common.Memory import Memory
from Core.Storage.NetworkXStorage import NetworkXStorage
from Core.Graph.BaseGraph import BaseGraph
from Core.Common.Logger import logger
from typing import Union, List, Any
from Core.Schema.ChunkSchema import TextChunk
from Core.Schema.EntityRelation import ChunkEntity, ChunkRelationship
from collections import defaultdict
import asyncio
from Core.Common.Utils import (clean_str, build_data_for_merge, csr_from_indices, csr_from_indices_list)
from Core.Utils.MergeER import MergeEntity, MergeRelationship

class ChunkGraph(BaseGraph):
    def __init__(self, config, llm, encoder, is_second_graph):
        super().__init__(config, llm, encoder)
        self._graph = NetworkXStorage()
    
    async def __graph__(self, nodes_element: list, edges_element: list):
        """
        Build the graph based on the input elements.
        """
        # Initialize dictionaries to hold aggregated node and edge information
        maybe_nodes, maybe_edges = defaultdict(list), defaultdict(list)

        # Iterate through each tuple of nodes and edges in the input elements
        for m_nodes in nodes_element:
            # Aggregate node information
            for k, v in m_nodes.items():
                maybe_nodes[k].extend(v)
        
        for m_edges in edges_element:
            # Aggregate edge information
            for k, v in m_edges.items():
                maybe_edges[tuple(sorted(k))].extend(v)

        # Asynchronously merge and upsert nodes
        await asyncio.gather(*[self._merge_nodes_then_upsert(k, v) for k, v in maybe_nodes.items()])

        # Asynchronously merge and upsert edges
        await asyncio.gather(*[self._merge_edges_then_upsert(k[0], k[1], v) for k, v in maybe_edges.items()])
        
    def _extract_entity_relationship(self, chunk_key_pair: tuple[str, TextChunk]):
        pass
    
    async def _extract_relationship(self,
                                    community_key: str, 
                                    community_value: dict[str, Any]):
        # relations are their corresponding linkings.
        # 1. summary-chunk -> raw-chunk
        # 2. sub-summary-chunk -> parent-summary-chunk
        # 3. raw-chunks
        
        # 1. summary-chunk -> raw-chunk
        maybe_edges = {
            tuple(sorted(["community-" + community_key, chunk_id])) : [
                ChunkRelationship(**dict(zip(['src_id', 'tgt_id'], sorted(["community-" + community_key, chunk_id]))))
            ]
            for chunk_id in community_value['chunk_ids']
        }
        
        # 2. sub-summary-chunk -> parent-summary-chunk
        maybe_edges.update(
            {
                tuple(sorted(["community-" + community_key, "community-" + sub_community_key])) : [
                    ChunkRelationship(**dict(zip(['src_id', 'tgt_id'], sorted(["community-" + community_key, "community-" + sub_community_key]))))
                ]
                for sub_community_key in community_value['sub_communities']
            }
        )
        
        return maybe_edges
    
    async def _extract_entity(self, 
                              community_dict: dict[str, dict[str, Any]], 
                              chunks_dict: dict[str, TextChunk]):
        # entity is either summary-chunk or raw-chunk.
        # 1. raw chunks entity
        nodes_elements = {
            chunk_id : [
                ChunkEntity(chunk_id=chunk_id, content=chunk.content) 
            ]
            for chunk_id, chunk in chunks_dict.items()
        }
        # 2. summary chunks entity
        nodes_elements.update(
            {
                "community-" + community_key : [
                    ChunkEntity(chunk_id="community-" + community_key, content=community_value['report_string'])
                ]
                for community_key, community_value in community_dict.items()
            }
        )
        
        # import pdb; pdb.set_trace()
        
        return [
            nodes_elements
        ]

    async def _build_graph(self, chunks):
        try:
            community_dict: dict[str, dict[str, Any]] = chunks[0]
            chunks_dict: dict[str, TextChunk] = chunks[1]
            
            edges_elements = await asyncio.gather(
                *[self._extract_relationship(community_key, community_value) 
                  for community_key, community_value in community_dict.items()
                ]
            ) # list[dict[tuple[str, str], List[ChunkRelationship]]]
            
            nodes_elements = await self._extract_entity(community_dict, chunks_dict) # list[dict[str, List[ChunkEntity]]]
            
            # Build graph based on the extracted entities and triples
            await self.__graph__(nodes_element=nodes_elements, edges_element=edges_elements)
        except Exception as e:
            logger.exception(f"Error building graph: {e}")
        finally:
            logger.info("Constructing graph finished")
            
    async def _merge_nodes_then_upsert(self, entity_name: str, nodes_data: List[ChunkEntity]) -> None:
        existing_node = await self._graph.get_node(entity_name)

        existing_data = build_data_for_merge(existing_node) if existing_node else defaultdict(list)
        # Groups node properties by their keys for upsert operation.
        upsert_nodes_data = defaultdict(list)
        for node in nodes_data:
            for node_key, node_value in node.as_dict.items():
                upsert_nodes_data[node_key].append(node_value)

        chunk_id = (MergeEntity.merge_source_ids(existing_data["chunk_id"],
                                                  upsert_nodes_data["chunk_id"]))
        
        content = (MergeEntity.merge_content(existing_data["content"],
                                             upsert_nodes_data["content"]))
        
        node_data = dict(chunk_id=chunk_id, content=content)

        # Upsert the node with the merged data
        await self._graph.upsert_node(entity_name, node_data=node_data)

    async def _merge_edges_then_upsert(self, src_id: str, tgt_id: str, edges_data: List[ChunkRelationship]) -> None:
        
        edge_data = dict(src_id=src_id, tgt_id=tgt_id)
        # Upsert the edge with the merged data
        await self._graph.upsert_edge(src_id, tgt_id, edge_data=edge_data)
            
    async def get_nodes_data(self, chunks: list[str]):
        #get_node
        nodes_data = await asyncio.gather(*[self.get_node(node_id=chunk_id) for chunk_id in chunks])
        
        # import pdb
        # pdb.set_trace()
        
        return nodes_data