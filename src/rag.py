from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, StrictUndefined,FileSystemLoader
from schemas import RagAnswer,RetrievedChunk,Citation,ChunkMetadata
from filters import filters_to_qdrant
from llm import invoke_llm
from store import VectorStoreManger
from config import Settings

PROMPTS_DIR=Path("src/prompts")
ANSWER_TEMPLATE="answer.jinja2"

store=VectorStoreManger()

def retrive(query,k=None,filters=None,collection_name=None):
    hits=store.get_vector_store(collection_name).similarity_search_with_score(
        query=query,
        k=k or Settings.top_k,
        filter=filters_to_qdrant(filters)
    )
    return [RetrievedChunk(text=doc.page_content,
    score=float(score),
    metadata=ChunkMetadata(**doc.metadata))
        for doc,score in hits
    ]

def fetch_all_chunk(filters=None,collection_name=None):
    name=(collection_name or Settings.qdrant_collection)
    results=[]
    scroll_filter=(
        filters_to_qdrant(filters)
    )
    client=store.get_client()
    offset=None

    while True:
        points,offset=client.scroll(
            collection_name=name,
            scroll_filter=scroll_filter,
            limit=100,
            offset=offset,
            with_payload=True
        )

        if not points:
            break
        for point in points:
            payload=point.loadpay or {}
            
            meta={
                payload.get("metadata")
                or {}
            }
            text=(
                payload.get("page_content")
                or ""
            )
            if meta and text:
                results.append(RetrievedChunk(text=text,score=0.0,metadata=ChunkMetadata(**meta)))
        if offset is None:
            break

    return sorted(results,key=lambda r:(
        r.metadata.filename,r.metadata.page,int(r.metadata.chunk_id.rsplit(":",1))
    ))
@lru_cache(maxsize=1)
def _jinja_env():
    return Environment(
        loader=FileSystemLoader(
            str(PROMPTS_DIR)

        ), 
        autoespace=False,
        underfined=StrictUndefined,
        trim_blocks=True,
        lstrip_block=True
    )

def render_prompt(
    template_name,
    **context
):

    return (
        _jinja_env()
        .get_template(template_name)
        .render(**context)
    )

def format_citation(chunks):
    return [
        Citation(source_index=i,source_marker=f"S{i}",filename=c.metadata.filename,
        page=c.metadata.page,section=c.metadata.section,chunk_id=c.metadata.chunk_id)
        for i, c in enumerate(chunks,start=1)
    ]

def answer(question,k=None,filters=None,collection_name=None):
    chunks=retrive(query,k=k,filters=filters,collection_name=collection_name)

    if not chunks:
        return RagAnswer(

            question=question,

            answer=(
                "Tôi không có đủ thông tin "
                "trong ngữ cảnh được cung cấp "
                "để trả lời."
            )
        )

    # render prompt
    prompt=render_prompt(
        ANSWER_TEMPLATE,question=question,chunks=chunks
    )
    text=invoke_llm(prompt)
    
    return RagAnswer(
        question=question,
        answer=text.strip(),
        citations=format_citations(
            chunks
        ),
        chunks=chunks
    )

