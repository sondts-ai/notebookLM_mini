import json 
from  pydantic import ValidationError
from config import Settings

from rag import retrive,fetch_all_chunk,render_prompt,format_citation
from llm import invoke_llm

from schemas import Summary,QuizItem,QuizSet,Flashcard,FlashcardSet

SUMMARY_SINGLE_TEMPLATE = (
    "summary_single.jinja2"
)
SUMMARY_MAP_TEMPLATE=(
    "summary_map.jinja2"
)
SUMMARY_REDUCE_TEMPLATE = (
    "summary_reduce.jinja2"
)
QUIZ_TEMPLATE = "quiz.jinja2"
FLASHCARDS_TEMPLATE = (
    "flashcards.jinja2"
)

def _resolve_target(document,query,filters,k,retrive_k):
    effective_filters=dict(filters or {})

    if document:
        effective_filters["filename"]=document
    if query:
        chunks=retrive(query,k=k or retrive_k,filters=effective_filters)
        return chunks,"query",query
    if effective_filters:
        chunks=fetch_all_chunk(filters=effective_filters)
        scope="document" if document else "filter"
        target = ", ".join(f"{k}={v}" for k, v in effective_filters.items())
        return chunks, scope, target
    return fetch_all_chunk(filters=None),"corpus",None

def _parse_json(text):
    cleaned = text.strip()

    # remove markdown fences
    if cleaned.startswith("```"):
        cleaned = (cleaned.split("\n", 1)[-1].removesuffix("```").strip())

    obj = json.loads(cleaned)

    if not isinstance(obj,(dict, list)):
        raise RuntimeError(
            "Expected JSON object or array."
        )
    return obj

def _validate_summary_payload(payload):
    summary=str(payload.get("summary","").strip())
    key_points=[str(x).strip() for x in payload.get("key_points",[]) if str(x).strip()]
    if not summary:
        raise RuntimeError(
            "Missing summary text."
        )
    return summary, key_points

def _validate_items(payload,key,model_class,dedup_field,label,valid_markers):

    raw_items = payload.get(key)
    items = []
    seen = set()
    for raw in raw_items:

        try:
            item = (model_class.model_validate(raw))

        except ValidationError:
            continue

        # deduplicate
        norm = str(getattr(item,dedup_field,"")).strip().lower()

        if (not norm or norm in seen):
            continue

        seen.add(norm)

        # remove invalid citations
        markers = [
            m
            for m in item.source_markers
            if m in valid_markers
        ]

        items.append(
            item.model_copy(
                update={
                    "source_markers": markers
                }
            )
        )

    if not items:
        raise RuntimeError(
            f"No valid {label} produced."
        )
    return items

def sumamarize(document=None,query=None,filters=None,k=None):
    chunks,scope,target=_resolve_target(document,query,filters,k,Settings.summarize_retrieval_k)

    if len(chunks)<=Settings.summarize_batch_size:
        prompt=render_prompt(SUMMARY_SINGLE_TEMPLATE,chunks=chunks)
        payload=_parse_json(invoke_llm(prompt))
        summary_text,key_points=_validate_summary_payload(payload)

    else:
        partials=[]
        for start in range(0,len(chunks),Settings.summarize_batch_size):
            batch=chunks[start:start+Settings.summarize_batch_size]
            payload=_parse_json(invoke_llm(render_prompt(SUMMARY_MAP_TEMPLATE, chunks=batch)))
            summary_text, key_points = _validate_summary_payload(payload)
            partials.append({"summary":summary_text,"key_points":key_points})
        payload=_parse_json(invoke_llm(render_prompt(SUMMARY_REDUCE_TEMPLATE,partials=partials)))    
        summary_text, key_points = _validate_summary_payload(payload)

        return Summary(
        scope=scope,
        target=target,
        summary=summary_text,
        key_points=key_points,
        citations=format_citations(chunks),
        chunks=chunks
        )

def generate_quiz(document=None,query=None,count=None,k=None):
    chunks,scope,target=_resolve_target(document,query,filters,k,Settings.summarize_retrieval_k)
    n=count or Settings.quiz_default_count
    valid_markers={f"S{i}" for i in range(1, len(chunks) + 1)}
    prompt=render_prompt(QUIZ_TEMPLATE,chunks=chunks,count=n)
    payload=_parse_json(invoke_llm(prompt))
    items=_validate_items(payload, "items", QuizItem, "question", "quiz items",valid_markers)

    return QuizSet(scope=scope, target=target, items=items, chunks=chunks,
                    citations=format_citations(chunks))

def generate_flashcard(document=None,query=None,count=None,k=None):
    chunks,scope,target=_resolve_target(document,query,filters,k,Settings.summarize_retrieval_k)
    n=count or Settings.flashcards_default_count
    valid_markers={f"S{i}" for i in range(1, len(chunks) + 1)}
    prompt=render_prompt(FLASHCARDS_TEMPLATE,chunks=chunks,count=n)
    payload=_parse_json(invoke_llm(prompt))
    cards=_validate_items(payload, "cards", Flashcard, "front", "flashcards",valid_markers)

    return FlashcardSet(scope=scope,target=target, cards=cards, chunks=chunks,citations=format_citations(chunks))