import json
from pathlib import Path

from chunkers.fixed import FixedChunker
from chunkers.fixed_overlap import FixedOverlapChunker
from chunkers.sentence import SentenceChunker
from chunkers.semantic import SemanticChunker


PROCESSED_DOC_PATH = Path("data/processed/rulebook_processed.json")
OUTPUT_DIR = Path("data/processed")
TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_processed_doc():
    if not PROCESSED_DOC_PATH.exists():
        raise FileNotFoundError(
            f"Processed document not found: {PROCESSED_DOC_PATH}"
        )
    with open(PROCESSED_DOC_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_chunks(chunks, filename):
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(chunks)} chunks → {path}")


def main():
    processed_doc = load_processed_doc()

    # Fixed
    fixed = FixedChunker(
        tokenizer_name=TOKENIZER_NAME,
        chunk_size=512
    )
    fixed_chunks = fixed.chunk_document(processed_doc)
    save_chunks(fixed_chunks, "fixed_chunks.json")

    # Fixed + overlap
    fixed_overlap = FixedOverlapChunker(
        tokenizer_name=TOKENIZER_NAME,
        chunk_size=512,
        overlap_ratio=0.2
    )
    fixed_overlap_chunks = fixed_overlap.chunk_document(processed_doc)
    save_chunks(fixed_overlap_chunks, "fixed_overlap_chunks.json")

    # Sentence-based
    sentence = SentenceChunker(
        tokenizer_name=TOKENIZER_NAME,
        max_chunk_tokens=512
    )
    sentence_chunks = sentence.chunk_document(processed_doc)
    save_chunks(sentence_chunks, "sentence_chunks.json")

    # Semantic
    semantic = SemanticChunker(
        embedding_model_name=TOKENIZER_NAME,
        tokenizer_name=TOKENIZER_NAME,
        similarity_threshold=0.75,
        max_chunk_tokens=512
    )
    semantic_chunks = semantic.chunk_document(processed_doc)
    save_chunks(semantic_chunks, "semantic_chunks.json")


if __name__ == "__main__":
    main()
