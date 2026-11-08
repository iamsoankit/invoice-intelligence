"""Policy Q&A: retrieval-augmented answers over the AP policy knowledge base."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from .retrieval import tokenize
from .schemas import RetrievedChunk

QA_SYSTEM = """You answer questions about the company's Accounts Payable policy.
Use ONLY the policy excerpts in <policies>. Cite every claim with its policy ID in square brackets, e.g. [BANK-01].
If the excerpts do not answer the question, say so plainly instead of guessing."""


class PolicyAnswer(BaseModel):
    answer: str = Field(description="Concise answer with inline [POLICY-ID] citations.")
    cited_ids: list[str] = Field(default_factory=list, description="Policy IDs used in the answer.")
    grounded: bool = Field(description="False if the excerpts did not contain the answer.")


def extractive_answer(question: str, retrieved: list[RetrievedChunk], max_sentences: int = 3) -> PolicyAnswer:
    """Offline answer: the policy sentences with the highest term overlap, each cited."""
    terms = set(tokenize(question))
    candidates = []
    for rank, hit in enumerate(retrieved):
        for sentence in re.split(r"(?<=[.!?])\s+", hit.chunk.text):
            overlap = len(terms & set(tokenize(sentence)))
            if overlap:
                # Retrieval rank dominates so answers stay anchored to the best-matching policy.
                candidates.append((overlap / (1 + rank), -rank, sentence.strip(), hit.chunk.id))
    if not candidates:
        return PolicyAnswer(answer="The policy knowledge base does not cover this question.", grounded=False)
    candidates.sort(reverse=True)
    best = candidates[0][0]
    picked = [c for c in candidates if c[0] >= best / 2][:max_sentences]
    answer = " ".join(f"{s} [{cid}]" for _, _, s, cid in picked)
    return PolicyAnswer(answer=answer, cited_ids=list(dict.fromkeys(c for *_, c in picked)), grounded=True)


def answer_question(question: str, retrieved: list[RetrievedChunk], client=None, model: str | None = None):
    """Returns (PolicyAnswer, input_tokens, output_tokens)."""
    if client is None:
        return extractive_answer(question, retrieved), 0, 0

    from google.genai import types

    from . import config

    policies = "\n\n".join(f"[{r.chunk.id}] {r.chunk.title}\n{r.chunk.text}" for r in retrieved)
    response = client.models.generate_content(
        model=model or config.GENERATION_MODEL,
        contents=f"<policies>\n{policies}\n</policies>\n\nQuestion: {question}",
        config=types.GenerateContentConfig(
            system_instruction=QA_SYSTEM, temperature=0.0,
            response_mime_type="application/json", response_schema=PolicyAnswer,
        ),
    )
    answer = PolicyAnswer.model_validate_json(response.text)
    allowed = {r.chunk.id for r in retrieved}
    answer.cited_ids = [cid for cid in answer.cited_ids if cid in allowed]
    meta = response.usage_metadata
    return answer, (meta.prompt_token_count or 0) if meta else 0, (meta.candidates_token_count or 0) if meta else 0
