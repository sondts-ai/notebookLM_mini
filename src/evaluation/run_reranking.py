from sentence_transformers import CrossEncoder
from rag import render_prompt,retrive,ANSWER_TEMPLATE,format_citation
from llm import invoke_llm
from schemas import RagAnswer

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

def answer_reranking(question:str,collection_name:str,reranker:CrossEncoder,initial_k:int=5,
rerank_k=15,filters:dict[str, object] | None = None)->RagAnswer:
    chunks=retrive(question,initial_k,filters,collection_name)

    if not chunks:
        return RagAnswer(
            question=question,
            answer="Tôi không có đủ thông tin trong ngữ cảnh được cung cấp để trả lời."
        )
    scores=reranker.predict([[question,chunk.text] for chunk in chunks])
    for chunk,score in zip(chunks,scores):
        chunk.score=float(score)
    
    reranked=sorted(chunks,key=lambda c:c.score, reverse=True)[:rerank_k]

    prompt=render_prompt(ANSWER_TEMPLATE,question=question,chunks=reranked)
    text=invoke_llm(prompt)

    return RagAnswer(
        question=question,
        answer=text.strip(),
        citations=format_citation(reranked),
        chunks=reranked
    )
