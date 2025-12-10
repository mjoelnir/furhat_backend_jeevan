"""
Simplified DocumentAgent for the NorwAI backend.

Design choices:
- Keep RAG retrieval lightweight (BM25) via rag_flow; no vector DB or caching layers.
- Default to Ollama-backed LLM to avoid heavy GPU deps; can swap via create_llm config.
- Minimal prompt building with optional language steering for bilingual trivia flows.

Pipeline: question -> RAG.retrieve(context) -> build prompt -> LLM -> answer
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document

from my_furhat_backend.config.settings import config
from my_furhat_backend.models.llm_factory import create_llm
from my_furhat_backend.RAG.rag_flow import RAG

logger = logging.getLogger(__name__)


class DocumentAgent:
    """
    Minimal RAG+LLM agent.

    Public API:
        - run(question: str) -> str
        - engage(document_name: str, answer: str) -> str
        - clear_all_caches() -> None  (no-op for compatibility)
    """

    def __init__(self) -> None:
        # Initialize a simple RAG helper for semantic search over the NorwAI PDF.
        self.rag = RAG()

        # Initialize the Ollama-backed LLM using the configured model and system prompt.
        self.llm = create_llm(
            "ollama",
            model=config.get("OLLAMA_MODEL", "llama3.2:latest"),
            base_url=config.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            system_prompt=config.get("OLLAMA_SYSTEM_PROMPT"),
        )

    @staticmethod
    def _build_context(docs: List[Document], max_chars: int = 4000) -> str:
        """
        Turn retrieved documents into a compact textual context for the LLM.
        """
        if not docs:
            return "No relevant document context could be retrieved for this question."

        segments: List[str] = []
        for doc in docs:
            content = (doc.page_content or "").strip()
            if not content:
                continue
            metadata = doc.metadata or {}
            page = metadata.get("page")
            # Keep a simple textual prefix instead of bracketed page labels to avoid
            # awkward symbols in spoken output.
            if isinstance(page, int):
                prefix = f"Page {page + 1}: "
            else:
                prefix = ""
            segments.append(f"{prefix}{content}")

        context = "\n\n".join(segments)
        if len(context) > max_chars:
            context = context[:max_chars] + "..."
        return context or "No relevant document context could be retrieved for this question."

    @staticmethod
    def _build_prompt(question: str, context: str, preferred_language: str | None = None) -> str:
        """
        Construct a single-string prompt for the underlying LLM.
        """
        system = config.get("OLLAMA_SYSTEM_PROMPT", "")

        # Optional language steering. The trivia Q&A pairs are typically in Norwegian,
        # but the user may speak a different language. You MUST respond in the user's
        # language, translating any Norwegian text or facts as needed.
        lang_hint = None
        if preferred_language:
            pl = preferred_language.lower()
            if pl.startswith("norw") or pl in {"no", "nb", "nn"}:
                lang_hint = (
                    "The user is speaking Norwegian. You MUST respond only in Norwegian. "
                    "If the context or Q&A pairs are written in another language, translate "
                    "them into natural Norwegian before answering. Never respond in English."
                )
            elif pl.startswith("eng") or pl in {"en", "en-us", "en-gb"}:
                lang_hint = (
                    "The user is speaking English. You MUST respond only in English. "
                    "Most of the context and Q&A pairs may be written in Norwegian; "
                    "translate all relevant information into natural English before answering. "
                    "Never respond in Norwegian."
                )
        prompt_parts = [
            system,
            "",
            "You are given context from documents or question–answer pairs.",
            "Answer the user's question using only this context.",
            lang_hint,
            'If the context does not contain the answer, say "I couldn’t find that in the documents you provided."',
            "",
            "Context:",
            context,
            "",
            f"Question: {question}",
            "",
            "Answer:",
        ]
        return "\n".join(part for part in prompt_parts if part is not None)

    def run(self, initial_input: str, preferred_language: str | None = None) -> str:
        """
        Main entry point used by the FastAPI backend (/ask, /transcribe).
        """
        question = (initial_input or "").strip()
        if not question:
            return "I didn't receive a question to answer."

        try:
            # 1) Retrieve relevant chunks from the vector store
            docs = self.rag.query(question, k=20)

            # 2) Build a compact textual context
            context = self._build_context(docs)

            # 3) Build a simple prompt and query the LLM
            prompt = self._build_prompt(question, context, preferred_language)
            raw_response = self.llm.query(prompt)

            return str(raw_response).strip()

        except Exception as exc:  # noqa: BLE001
            logger.exception("Error in DocumentAgent.run: %s", exc)
            return f"I encountered an error while answering your question: {exc}"

    def engage(self, document_name: str, answer: str) -> str:
        """
        Generate a single follow-up question based on the last answer.

        This is used by the /engage endpoint to keep the conversation going.
        """
        try:
            prompt = (
                "You are helping a user explore a NorwAI document.\n"
                "Based on the previous answer, suggest ONE natural follow-up question "
                "that stays on topic and encourages deeper exploration.\n\n"
                f"Document: {document_name}\n"
                f"Previous answer: {answer}\n\n"
                "Follow-up question:"
            )
            raw_response = self.llm.query(prompt)
            return str(raw_response).strip()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error in DocumentAgent.engage: %s", exc)
            return "I couldn't generate a follow-up question right now."

    def trivia_turn(
        self,
        phase: str,
        question: str,
        answer: str,
        user_answer: str | None = None,
        preferred_language: str | None = None,
    ) -> str:
        """
        Generate localized trivia utterances for the robot.

        phase:
            - "ask":     Turn the raw trivia question into a natural question in the
                         user's language (do NOT reveal the answer).
            - "feedback": Explain briefly if the user's answer was correct or not,
                         state the correct answer, and optionally invite another round.
        """
        phase = (phase or "").strip().lower()
        q = (question or "").strip()
        a = (answer or "").strip()
        ua = (user_answer or "").strip()

        if not q or not a:
            return "I don't have a valid trivia question and answer to use."

        # Reuse the language-hinting logic from _build_prompt
        lang_hint = None
        if preferred_language:
            pl = preferred_language.lower()
            if pl.startswith("norw") or pl in {"no", "nb", "nn"}:
                lang_hint = (
                    "The user is speaking Norwegian. You MUST respond only in Norwegian. "
                    "If the trivia question and answer are written in another language, "
                    "translate them into natural Norwegian before speaking. Never respond in English."
                )
            elif pl.startswith("eng") or pl in {"en", "en-us", "en-gb"}:
                lang_hint = (
                    "The user is speaking English. You MUST respond only in English. "
                    "Most trivia questions and answers may be written in Norwegian; "
                    "translate all relevant information into natural English before speaking. "
                    "Never respond in Norwegian."
                )

        if phase == "ask":
            prompt = (
                f"{lang_hint or ''}\n\n"
                "You are running a Norwegian trivia game as a social robot.\n"
                "You are given ONE trivia question (originally Norwegian):\n\n"
                f"Question: {q}\n"
                f"Correct answer: {a}\n\n"
                "Task:\n"
                " - Produce exactly ONE short, natural-sounding question in the user's language.\n"
                " - Do NOT reveal or hint at the correct answer.\n"
                " - Do NOT add explanations, commentary, or extra instructions.\n"
                "Return only the question sentence you will speak aloud."
            )
        elif phase == "feedback":
            prompt = (
                f"{lang_hint or ''}\n\n"
                "You are running a Norwegian trivia game as a social robot.\n"
                "You are given one trivia question, its correct answer, and what the user answered:\n\n"
                f"Question: {q}\n"
                f"Correct answer: {a}\n"
                f"User answer: {ua}\n\n"
                "Task:\n"
                " - In the user's language, say briefly whether they were correct or not (speak directly to 'you/your', never say 'the user').\n"
                " - Clearly state the correct answer.\n"
                " - Be kind and lenient; if their answer is close, acknowledge that politely.\n"
                " - Optionally add ONE short follow-up sentence inviting them to try another question.\n"
                "Keep the total output to at most two short sentences."
            )
        else:
            return "I didn't recognise this trivia phase."

        try:
            raw_response = self.llm.query(prompt)
            return str(raw_response).strip()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error in DocumentAgent.trivia_turn: %s", exc)
            return "I couldn't generate the trivia line right now."

    def clear_all_caches(self) -> None:
        """
        Compatibility stub.

        The simplified agent does not maintain long-lived semantic caches,
        but this method is kept so existing tooling or docs calling it
        will not break.
        """
        logger.info("DocumentAgent.clear_all_caches called; no caches to clear in simplified agent.")


