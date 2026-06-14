import hashlib
import uuid
from collections import defaultdict
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.store import get_vector_store, ensure_collection
from src.config import Settings as settings


def discover_pdfs() -> list[Path]:
    data_dir = Path(settings.data_dir)

    if not data_dir.exists():
        return []

    return [
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() == ".pdf"
    ]


class DocumentIndexer:
    def generate_document_id(self, path: Path) -> str:
        raw = f"{path.name}:{path.stat().st_size}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def generate_chunk_id(self, doc_id: str, page: int, index: int) -> str:
        return f"{doc_id}:{page}:{index}"

    def load_pdf(self, path: Path):
        pages = PyPDFLoader(str(path)).load()
        document_id = self.generate_document_id(path)

        for page in pages:
            page_number = int(page.metadata.get("page", 0)) + 1

            page.metadata = {
                "document_id": document_id,
                "filename": path.name,
                "source": str(path.resolve()),
                "page": page_number,
                "section": page.metadata.get("section"),
            }

        return pages

    def create_split(self, chunk_size: int, chunk_overlap: int):
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=False,
        )

    def build_chunks(
        self,
        pdf_path: Path,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        chunker=None,
    ):
        chunk_size = chunk_size or settings.chunk_size
        chunk_overlap = chunk_overlap or settings.chunk_overlap

        page_docs = self.load_pdf(pdf_path)

        splitter = chunker or self.create_split(chunk_size, chunk_overlap)
        chunks = splitter.split_documents(page_docs)

        per_doc_counter = defaultdict(int)

        for chunk in chunks:
            document_id = chunk.metadata["document_id"]
            index = per_doc_counter[document_id]
            per_doc_counter[document_id] += 1

            chunk.metadata = {
                "document_id": document_id,
                "filename": chunk.metadata["filename"],
                "source": chunk.metadata["source"],
                "page": chunk.metadata["page"],
                "section": chunk.metadata.get("section"),
                "chunk_id": self.generate_chunk_id(
                    document_id,
                    chunk.metadata["page"],
                    index,
                ),
            }

        return chunks

    def index_chunks(self, chunks, collection_name=None):
        if not chunks:
            return 0

        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_DNS, c.metadata["chunk_id"]))
            for c in chunks
        ]

        get_vector_store(collection_name=collection_name).add_documents(
            chunks,
            ids=ids,
        )

        return len(chunks)

    def ingest(
        self,
        recreate=False,
        collection_name=None,
        chunker=None,
        chunk_size=None,
        chunk_overlap=None,
    ):
        pdfs = discover_pdfs()

        ensure_collection(
            recreate=recreate,
            collection_name=collection_name,
        )

        all_chunks = []

        for pdf_path in pdfs:
            all_chunks.extend(
                self.build_chunks(
                    pdf_path,
                    chunker=chunker,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )

        return self.index_chunks(all_chunks, collection_name)

    def save_and_ingest_pdf(self, file_bytes, filename):
        safe_name = Path(filename).name
        data_dir = Path(settings.data_dir)
        dest = data_dir / safe_name

        data_dir.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(file_bytes)

        ensure_collection(recreate=False)

        chunks = self.build_chunks(dest)

        return {
            "filename": safe_name,
            "chunks_indexed": self.index_chunks(chunks),
        }


indexer = DocumentIndexer()


def save_and_ingest_pdf(file_bytes, filename):
    return indexer.save_and_ingest_pdf(file_bytes, filename)