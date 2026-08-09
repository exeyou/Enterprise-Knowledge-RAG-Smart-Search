from typing import List
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.postprocessor.cohere_rerank import CohereRerank

from backend.config import settings


class HybridRetriever:

    def __init__(self, vector_store: QdrantVectorStore, all_nodes: List):
        index = VectorStoreIndex.from_vector_store(vector_store)
        self.dense_retriever = index.as_retriever(similarity_top_k=10)

        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=all_nodes, similarity_top_k=10
        )

        self.fusion_retriever = QueryFusionRetriever(
            retrievers=[self.dense_retriever, self.bm25_retriever],
            similarity_top_k=10,
            num_queries=1,
            mode="reciprocal_rerank",
        )

        self.reranker = (
            CohereRerank(
                api_key=settings.COHERE_API_KEY,
                top_n=5,
                model="rerank-v3.5",
            )
            if settings.COHERE_API_KEY
            else None
        )

    def retrieve(self, query_str: str) -> List[NodeWithScore]:
        query_bundle = QueryBundle(query_str)
        candidate_nodes = self.fusion_retriever.retrieve(query_bundle)

        if self.reranker:
            final_nodes = self.reranker.postprocess_nodes(
                candidate_nodes, query_bundle
            )
        else:
            final_nodes = candidate_nodes[:5]

        return final_nodes