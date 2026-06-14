from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings


INDEXED_PAYLOAD_FIELDS = {
    "metadata.document_id": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.filename": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.page": qmodels.PayloadSchemaType.INTEGER,
}


class VectorStoreManager:
    @lru_cache(maxsize=1)
    def get_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": settings.hf_device},
            encode_kwargs={"normalize_embeddings": True},
        )

    @lru_cache(maxsize=1)
    def get_client(self):
        settings.storage_dir.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(settings.storage_dir))

    def ensure_collection(self, recreate=False, collection_name=None):
        client = self.get_client()
        name = collection_name or settings.qdrant_collection

        exists = client.collection_exists(name)

        if exists and recreate:
            client.delete_collection(name)
            exists = False

        if not exists:
            dim = len(self.get_embeddings().embed_query("dimension probe"))

            client.create_collection(
                collection_name=name,
                vectors_config=qmodels.VectorParams(
                    size=dim,
                    distance=qmodels.Distance.COSINE,
                ),
            )

        collection_info = client.get_collection(name)
        payload_schema = collection_info.payload_schema or {}

        for field, schema in INDEXED_PAYLOAD_FIELDS.items():
            if payload_schema.get(field) is None:
                client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=schema,
                )

    def get_vector_store(self, collection_name=None):
        return QdrantVectorStore(
            client=self.get_client(),
            collection_name=collection_name or settings.qdrant_collection,
            embedding=self.get_embeddings(),
        )


store_manager = VectorStoreManager()


def get_vector_store(collection_name=None):
    return store_manager.get_vector_store(collection_name)


def ensure_collection(recreate=False, collection_name=None):
    return store_manager.ensure_collection(
        recreate=recreate,
        collection_name=collection_name,
    )
        
def list_documents(collection_name=None):
    client = store_manager.get_client()
    name = collection_name or settings.qdrant_collection

    if not client.collection_exists(name):
        return []

    points, _ = client.scroll(
        collection_name=name,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )

    docs = {}

    for point in points:
        payload = point.payload or {}
        metadata = payload.get("metadata", {})

        document_id = metadata.get("document_id")
        filename = metadata.get("filename")
        page = metadata.get("page")

        if not filename:
            continue

        key = document_id or filename

        if key not in docs:
            docs[key] = {
                "document_id": document_id,
                "filename": filename,
                "pages": 0,
                "chunks": 0,
            }

        docs[key]["chunks"] += 1

        if isinstance(page, int):
            docs[key]["pages"] = max(docs[key]["pages"], page)

    return list(docs.values())
