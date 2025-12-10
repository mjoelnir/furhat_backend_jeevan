"""
Simple RAG (Retrieval-Augmented Generation) module.

Design choices:
- Lightweight lexical BM25 (rank_bm25) instead of heavier vector embeddings to
  avoid extra dependencies and keep startup fast on low-resource machines.
- Input corpus from DOCUMENTS_PATH; prefers QA JSONs when available for better
  trivia/document alignment, otherwise falls back to PDFs/txt.
- In-memory only: no external DB or persistent index to simplify deployment.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from my_furhat_backend.config.settings import config

logger = logging.getLogger(__name__)


class RAG:
    """
    Minimal RAG helper with BM25 over chunked documents/QA pairs.
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        pdf_path: str | None = None,  # kept for backward-compatibility, no longer required
        default_k: int = 20,
    ) -> None:
        # In the updated version, we treat DOCUMENTS_PATH as a directory
        # containing many small source documents (PDFs and/or .txt files).
        # The old pdf_path argument is kept only so existing callers do not break,
        # but is no longer used directly.
        self.documents_path = config["DOCUMENTS_PATH"]
        self.default_k = default_k

        # In-memory corpus and BM25 index
        self.chunks: List[Document] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: BM25Okapi | None = None

        self._build_index()

    def _load_documents(self) -> List[Document]:
        """
        Load source documents from DOCUMENTS_PATH.

        Priority rationale:
        1) Prefer QA JSON (qa_pairs.json) to align with trivia use-cases and
           avoid noisy PDF extraction when structured Q&A exists.
        2) Otherwise, load PDFs and plain text for a generic corpus.
        """
        base_dir = self.documents_path
        if not base_dir or not os.path.isdir(base_dir):
            logger.warning("RAG: documents directory not found at %s", base_dir)
            return []

        # --- 1) Prefer JSON QA-pair documents, if present ---
        json_pattern = os.path.join(base_dir, "*.json")
        json_paths = sorted(glob.glob(json_pattern))
        qa_docs: List[Document] = []

        for path in json_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:  # noqa: BLE001
                logger.warning("RAG: failed to load JSON %s: %s", path, exc)
                continue

            # Expect structure like {"qa_pairs": [{"question": "...", "answer": "..."}]}
            pairs = data.get("qa_pairs")
            if not isinstance(pairs, list):
                continue

            for idx, pair in enumerate(pairs):
                if not isinstance(pair, dict):
                    continue
                question = str(pair.get("question") or "").strip()
                answer = str(pair.get("answer") or "").strip()
                if not question or not answer:
                    continue

                page_content = f"Question: {question}\nAnswer: {answer}"
                qa_docs.append(
                    Document(
                        page_content=page_content,
                        metadata={
                            "source": os.path.basename(path),
                            "index": idx,
                            "type": "qa_pair",
                        },
                    )
                )

        if qa_docs:
            logger.info(
                "RAG: loaded %d QA-pair documents from %d JSON file(s) in %s",
                len(qa_docs),
                len(json_paths),
                base_dir,
            )
            # Treat QA pairs as the only corpus when present.
            return qa_docs

        # --- 2) Fallback: PDFs and plain-text snippets ---
        docs: List[Document] = []
        # 1) Load all PDFs
        pdf_pattern = os.path.join(base_dir, "*.pdf")
        pdf_paths = sorted(glob.glob(pdf_pattern))
        for path in pdf_paths:
            try:
                loader = PyPDFLoader(path)
                pdf_docs = loader.load()
                docs.extend(pdf_docs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("RAG: failed to load PDF %s: %s", path, exc)

        # 2) Load all plain-text snippets (optional, for mixed corpora)
        txt_pattern = os.path.join(base_dir, "*.txt")
        txt_paths = sorted(glob.glob(txt_pattern))
        for path in txt_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            except OSError as exc:
                logger.warning("RAG: failed to read %s: %s", path, exc)
                continue

            if not text:
                continue

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(path),
                        "path": path,
                    },
                )
            )

        if not docs:
            logger.warning(
                "RAG: no documents found in %s (no .pdf or .txt files)", base_dir
            )
            return []

        logger.info(
            "RAG: loaded %d documents from %s (pdfs: %d, txts: %d)",
            len(docs),
            base_dir,
            len(pdf_paths),
            len(txt_paths),
        )
        return docs

    def _chunk_documents(self, docs: List[Document]) -> List[Document]:
        """
        Split documents into overlapping chunks for retrieval.

        Chosen settings: chunk_size 800, overlap 150 to preserve context while
        keeping chunks compact for BM25 scoring.
        """
        if not docs:
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
        )
        chunks = splitter.split_documents(docs)
        logger.info("RAG: created %d chunks", len(chunks))
        return chunks

    def _build_index(self) -> None:
        """
        Tokenize chunks and build the BM25 index in memory.

        BM25 was chosen over embeddings to minimize dependencies and runtime
        cost; sufficient for small corpora and trivia Q&A retrieval.
        """
        docs = self._load_documents()
        self.chunks = self._chunk_documents(docs)

        if not self.chunks:
            logger.warning("RAG: no chunks available; BM25 index will be empty.")
            self.tokenized_corpus = []
            self.bm25 = None
            return

        def tokenize(text: str) -> List[str]:
            return (text or "").lower().split()

        self.tokenized_corpus = [
            tokenize(chunk.page_content) for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("RAG: BM25 index built over %d chunks", len(self.chunks))

    def query(self, query_text: str, k: int | None = None) -> List[Document]:
        """
        Run a BM25 lexical search against the indexed chunks.

        Lightweight, in-memory retrieval; returns top-k chunks (default k=self.default_k).
        """
        if not query_text or self.bm25 is None or not self.chunks:
            return []

        k = k or self.default_k

        query_tokens = query_text.lower().split()
        # BM25Okapi.get_top_n returns the top-n documents directly
        top_docs: List[Document] = self.bm25.get_top_n(
            query_tokens, self.chunks, n=k
        )
        return top_docs


