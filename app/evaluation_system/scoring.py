"""Keyword-based correctness check (stdlib only)."""
import re


def _found(answer: str, alternative: str) -> bool:
    # (?<!\w)...(?!\w): whole-word match that also works for terms like "$0.05" or "o(log n)"
    pattern = rf"(?<!\w){re.escape(alternative)}(?!\w)"
    return re.search(pattern, answer, re.IGNORECASE) is not None


def keyword_score(answer: str, keywords: tuple[tuple[str, ...], ...]) -> float:
    """Fraction of keyword groups for which at least one alternative appears."""
    if not keywords:
        return 0.0
    hits = sum(1 for group in keywords if any(_found(answer, alt) for alt in group))
    return hits / len(keywords)