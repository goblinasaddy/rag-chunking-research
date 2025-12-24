import os
import json
import re
from typing import List, Dict
from pathlib import Path

import pdfplumber
import nltk

# Ensure sentence tokenizer is available
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize


class DocumentLoader:
    """
    Loads and processes long-form PDF documents into structured text.

    Output format:
    [
        {
            "doc_id": str,
            "section_id": int,
            "text": str,
            "sentences": List[str]
        },
        ...
    ]
    """

    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_pdf(self, filename: str) -> str:
        """Extract raw text from a PDF file."""
        pdf_path = self.raw_dir / filename
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages_text = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)

        return "\n".join(pages_text)

    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text normalization without destroying structure."""
        text = re.sub(r"\n{2,}", "\n\n", text)     # normalize paragraph breaks
        text = re.sub(r"[ \t]+", " ", text)       # normalize spaces
        text = text.replace("\ufeff", "")         # remove BOM
        return text.strip()

    @staticmethod
    def split_paragraphs(text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs

    def process_document(self, pdf_filename: str) -> List[Dict]:
        """
        Full processing pipeline:
        PDF → cleaned text → paragraphs → sentences
        """
        raw_text = self.load_pdf(pdf_filename)
        cleaned_text = self.clean_text(raw_text)
        paragraphs = self.split_paragraphs(cleaned_text)

        structured_data = []
        doc_id = pdf_filename.replace(".pdf", "")

        for idx, para in enumerate(paragraphs):
            sentences = sent_tokenize(para)

            structured_data.append({
                "doc_id": doc_id,
                "section_id": idx,
                "text": para,
                "sentences": sentences
            })

        return structured_data

    def save_processed_document(self, pdf_filename: str) -> str:
        """
        Process a PDF and save structured output as JSON.
        Returns path to saved file.
        """
        structured_data = self.process_document(pdf_filename)

        output_path = self.processed_dir / f"{pdf_filename.replace('.pdf', '')}_processed.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)

        return str(output_path)


# -------------------------
# Standalone usage
# -------------------------
if __name__ == "__main__":
    loader = DocumentLoader(
        raw_dir="data/raw",
        processed_dir="data/processed"
    )

    # Example usage
    pdf_name = "rulebook.pdf"
    output_file = loader.save_processed_document(pdf_name)
    print(f"Processed document saved to: {output_file}")
