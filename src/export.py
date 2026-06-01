from typing import Literal
from pathlib import Path

from schemas import Summary, QuizSet, FlashcardSet


ExportFormat = Literal["text", "md", "json"]


def export(model, *, fmt: ExportFormat = "text", output: Path | None = None):
    if fmt == "json":
        text = model.model_dump_json(indent=2) + "\n"

    elif fmt in {"text", "md"}:
        text = _to_markdown(model)
        
    else:
        raise ValueError(
            f"Unknown fmt '{fmt}'. Expected 'text' | 'md' | 'json'."
        )

    if output is None:
        return text

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")

    return output