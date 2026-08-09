import qdrant_client.qdrant_fastembed

if not hasattr(qdrant_client.qdrant_fastembed, "IDF_EMBEDDING_MODELS"):
    qdrant_client.qdrant_fastembed.IDF_EMBEDDING_MODELS = {}

from typing import List
from qdrant_client import QdrantClient
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from backend.config import settings


class KnowledgeIndexer:

    def __init__(self):
        if settings.QDRANT_URL and settings.QDRANT_API_KEY:
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY
            )
        else:
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT
            )

        self.embed_model = HuggingFaceEmbedding(
            model_name=settings.DEFAULT_EMBEDDING_MODEL
        )

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=settings.QDRANT_COLLECTION_NAME
        )

    def add_nodes(self, nodes: List[TextNode]):
        if not nodes:
            return

        for node in nodes:
            node.embedding = self.embed_model.get_text_embedding(
                node.get_content(metadata_mode="embed")
            )

        self.vector_store.add(nodes)