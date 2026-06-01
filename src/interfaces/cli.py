import json
from pathlib import Path

import typer

from export import export
from filters import MetadataFilter, filters_to_dict
from indexing import ingest as ingest_data_dir
from rag import answer, retrieve
from learning import sumamarize as summarize_learning

app=typer.Typer()

def _parser_filters(filters:str|None):
    if not filters:
        return None
    data=json.load(filters)
    return filters_to_dict(MetadataFilter(**data))

def _print_answer(text:str):
    typer.echo("\nAnswer:")
    typer.echo(text)

def _print_sources(chunks):
    if not chunks:
        return 
    typer.echo("\nSource:")
    for i,chunk in enumerate(chunks,start=1):
        meta=chunk.metadata
        typer.echo(
            f"[S{i}] {meta.filename} - page {meta.page} - score={chunk.score:.4f}"
        )

def _emit(result,output:str|None,fmt:str):
    if output:
        path=Path(output)
        export(result,fmt=fmt,output=path)
        typer.echo(f"Saved to {path}")
    else:
        typer.echo(export(result, fmt=fmt))

@app.command()
def ingest(recreate: bool = False):
    count = ingest_data_dir(recreate=recreate)
    typer.echo(f"Done. {count} chunks indexed.")


@app.command()
def ask(
    question: str,
    k: int | None = None,
    filters: str | None = None,
):
    result = answer(
        question,
        k=k,
        filters=_parse_filters(filters),
    )

    _print_answer(result.answer)
    _print_sources(result.chunks)


@app.command("debug-retrieval")
def debug_retrieval(
    question: str,
    k: int | None = None,
    filters: str | None = None,
    as_json: bool = False,
):
    chunks = retrieve(
        question,
        k=k,
        filters=_parse_filters(filters),
    )

    if as_json:
        typer.echo(
            json.dumps(
                [c.model_dump() for c in chunks],
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        _print_sources(chunks)


@app.command("summarize")
def summarize(
    document: str | None = None,
    query: str | None = None,
    filters: str | None = None,
    k: int | None = None,
    output: str | None = None,
    fmt: str = "text",
):
    result = summarize_learning(
        document=document,
        query=query,
        filters=_parse_filters(filters),
        k=k,
    )

    _emit(result, output, fmt)


if __name__ == "__main__":
    app()