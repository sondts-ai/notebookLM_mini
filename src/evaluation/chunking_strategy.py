from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

_RECURSIVE_CONFIGS = [
    ("rc_500_50", 500, 50),
    ("rc_800_100", 800, 100),
    ("rc_1000_150", 1000, 150),
    ("rc_1500_200", 1500, 200),
]

_SEMANTIC_CONFIGS = [
    ("semantic_percentile", "percentile"),
    ("semantic_std_dev", "standard_deviation"),
    ("semantic_interquartile", "interquartile"),
]


class RecursiveChunker:
    def __init__(self, chunk_size=500, chunk_overlap=50, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def _splitter(self):
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators or DEFAULT_SEPARATORS,
            is_separator_regex=False,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []

        splitter = self._splitter()
        return splitter.split_documents(documents)

    def split_text(self, text: str) -> list[str]:
        splitter = self._splitter()
        return splitter.split_text(text)


## Semantic######
class SemanticWrapper:
    def __init__(self, embeddings: Embeddings, break_point="percentile"):
        self.embeddings = embeddings
        self.break_point = break_point

    def _splitter(self):
        return SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type=self.break_point,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        splitter = self._splitter()
        return splitter.split_documents(documents)

    def split_text(self, text: str) -> list[str]:
        splitter = self._splitter()
        return splitter.split_text(text)


def evaluate_strategy(test_cases, documents, embeddings, strategy_id, chunker):
    collection_name = f"rag_chunks__{strategy_id}"

    chunks = chunker.split_documents(documents)

    for chunk in chunks:
        vector = embed(chunk.page_content)

        qdrant.insert(
            collection_name=collection_name,
            document=chunk,
            embedding=vector,
        )

    def answer_fn(question):
        docs = retrieve(
            question,
            collection_name=collection_name,
        )
        answer = llm(question, docs)
        return answer

    result = evaluation(
        test_cases,
        answer_fn=answer_fn,
    )

    return result