from fastapi import FastAPI,Uploadfile,File
from pydantic import BaseModel,Field

from filters import MetadataFilter,filters_to_dict
from indexing import save_and_ingest_pdf
from rag import answer
from learning import sumamarize as sumamarize_learning,generate_quiz,generate_flashcard
from schemas import (
    RagAnswer,
    Summary,
    QuizSet,
    FlashcardSet,
    DocumentInfo,
    UploadResponse,
)
from store import list_documents

# request schemas
class Askrequest(BaseModel):
    question:str=Field(min_length=1)
    k:int| None=Field(default=None,ge=1,le=64)
    filters:MetadataFilter| None=None

class SummarizeRequest(BaseModel):
    document: str | None = None
    query: str | None = None
    filters: MetadataFilter | None = None
    k: int | None = Field(default=None, ge=1, le=64)


class QuizRequest(BaseModel):
    document: str | None = None
    query: str | None = None
    filters: MetadataFilter | None = None
    count: int | None = Field(default=None, ge=1, le=50)
    k: int | None = Field(default=None, ge=1, le=64)


class FlashcardsRequest(QuizRequest):
    pass

app=FastAPI(
    title="RAG Learning API",
    description="Grounded Q&A, summaries, quizzes, and flashcards over indexed PDFs.",
    verstion="0.1.0"
)

## End point

@app.get("/health")
def health():
    return {"status":"ok"}
    
@app.get("documents",response_model=list[DocumentInfo])
def documents():
    return list_documents()

@app.post("upload",response_model=UploadResponse)
async def upload(file:Uploadfile=File(...)):
    content=await file.read()

    return save_and_ingest_pdf(
        content,
        file.filename or ""
    )

@app.post("/ask", response_model=RagAnswer)
def ask(req: AskRequest):
    return answer(
        req.question,
        k=req.k,
        filters=filters_to_dict(req.filters),
    )

@app.post("/summarize",response_model=Summary)
def summarize(req:SummarizeRequest):
    return sumamarize_learning(
        document=req.document,
        query=req.query,
        filters=filters_to_dict(req.filters),
        k=req.k
    )

@app.post("/quiz", response_model=QuizSet)
def quiz(req: QuizRequest):
    return generate_quiz(
        document=req.document,
        query=req.query,
        filters=filters_to_dict(req.filters),
        count=req.count,
        k=req.k,
    )


@app.post("/flashcards", response_model=FlashcardSet)
def flashcards(req: FlashcardsRequest):
    return generate_flashcards(
        document=req.document,
        query=req.query,
        filters=filters_to_dict(req.filters),
        count=req.count,
        k=req.k,
    )

