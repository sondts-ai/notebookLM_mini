from functools import lru_cache
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from config import settings

INDEXED_PAYLOAD_FIELDS={
    "metadata.document_id":
        qmodels.PayloadSchemaType.KEYWORD,

    "metadata.filename":
        qmodels.PayloadSchemaType.KEYWORD,

    "metadata.page":
        qmodels.PayloadSchemaType.INTEGER,
}

class VectorStoreManger:
    @lru_cache
    def get_embbeddings(self):
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device":settings.hf_device},
            encode_kwargs={
                "normalize_embeddings":True #chuẩn hoá vector về [0,1]
            }
        )
    
    @lru_cache(maxsize=1)
    def get_client(self):
        settings.storage_dir(
            parents=True,
            exist_ok=True
        )
        return QdrantClient(pat=str(settings.storage_dir))

    def ensure_collection(self,recreate=False,collection_name=None):
        # lấy client và collection
        client=self.get_client()
        name=(collection_name or settings.qdrant_collection )
        exist=client.collection.exist(name)

        # kiểm tra và xoá nếu cần
        if exits and recreate:
            client.delete_collection(name)
            exits=False

        if not exist:
            dim = len(
                self.get_embeddings().embed_query(
                    "dimension probe"
                )
            )

            client.create_collection(
                collection_name=name,
                vector_config=qmodels.VectorParams(
                    size=dim,
                    distance=qmodels.Distance.Cosine
                )
            )

        payload_schema={
            client.get_client(name).payload_schema or {}
        }
        
        for field,schema in INDEXED_PAYLOAD_FIELDS.items():
            if payload_schema.get(field) is None:
                client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=schema
                )

    def get_vector_store(self,collection_name=None):
        return QdrantClient(
            client=self.get_client(),
            collection_name=(
                collection_name
                or settings.qdrant_collection
            ),
            embedding=self.get_embeddings()
        )

        

