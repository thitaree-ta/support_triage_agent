"""Finding help articles that match a question.

Method used here: BM25, a classic keyword-scoring formula (the idea behind many search boxes):
  - an article scores higher when it contains the query words often,
  - rare words count more than common ones,
  - very long articles are not unfairly favoured.
Words are first simplified ("charges", "charged" -> "charg") so different forms still match.

It sits behind the small `Retriever` interface, so it can later be replaced by an embedding
(meaning-based) search without touching the tool or the agent.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol, Sequence

_STOPWORDS = frozenset(
    "a an the and or of to in on for is are was were be it my i we you your with at by from this "
    "that do does can how what why".split()
)


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    body: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class KBHit:
    article: Article
    score: float


class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> list[KBHit]: ...


def _stem(token: str) -> str:
    """Tiny suffix stripper so 'charges/charged/charging' meet. Good enough for a mock KB."""
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        token = token[:-1]
    if token.endswith("ing") and len(token) > 5:
        token = token[:-3]
    elif token.endswith("ed") and len(token) > 5:
        token = token[:-2]
    if token.endswith("e") and len(token) > 3:
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


class KeywordRetriever:
    """BM25 over title (weighted x2), tags and body. English only: the agent translates queries."""

    def __init__(
        self,
        articles: Sequence[Article],
        k1: float = 1.5,
        b: float = 0.75,
        min_score: float = 1.0,
    ) -> None:
        self._articles = list(articles)
        self._k1, self._b, self._min_score = k1, b, min_score
        self._docs = [
            Counter(tokenize(f"{a.title} {a.title} {' '.join(a.tags)} {a.body}")) for a in self._articles
        ]
        self._lengths = [sum(doc.values()) for doc in self._docs]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._doc_freq: Counter[str] = Counter()
        for doc in self._docs:
            self._doc_freq.update(doc.keys())

    def search(self, query: str, top_k: int) -> list[KBHit]:
        terms = set(tokenize(query))
        n_docs = len(self._articles)
        hits: list[KBHit] = []
        for article, doc, length in zip(self._articles, self._docs, self._lengths):
            score = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if not tf:
                    continue
                df = self._doc_freq[term]
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                norm = tf + self._k1 * (1 - self._b + self._b * length / self._avg_len)
                score += idf * tf * (self._k1 + 1) / norm
            if score >= self._min_score:
                hits.append(KBHit(article, score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: max(top_k, 0)]
