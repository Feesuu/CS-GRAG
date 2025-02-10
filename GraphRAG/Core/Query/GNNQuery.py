from Core.Query.BaseQuery import BaseQuery
from Core.Common.Logger import logger
from Core.Common.Constants import Retriever
from Core.Prompt import QueryPrompt
# from Core.Prompt.QueryPrompt
import asyncio
class GNNQuery(BaseQuery):
    def __init__(self, config, retriever_context):
        super().__init__(config, retriever_context)

    async def _retrieve_relevant_contexts(self, query):
        # 1. parse entities from query
        #entities = await self.extract_query_entities(query)
        entities = query
        # 2. link entities to KG and get the chunk id (both raw_chunk and summary_chunk)
        chunk_dic =  await self._retriever.retrieve_relevant_content(type=Retriever.CHUNK,
                                                                   query=query,
                                                                   seed_entities=entities,
                                                                   mode="structure_encoding") # dict[str]
        
        # 3. merge them 
        combined_chunks = chunk_dic['raw_chunk'] + chunk_dic['summary_chunk']
        return combined_chunks

    async def generation_qa(self, query, context):
        pass

    async def generation_summary(self, query, context):
        pass
    
    async def pre_query(self, query):
        # await asyncio.gather(*[self._retrieve_relevant_contexts(query=query) for query in queries])
        combined_chunks = await self._retrieve_relevant_contexts(query=query)
        return combined_chunks