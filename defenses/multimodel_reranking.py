"""
Multi-Model Re-ranking Defense
Target attack: LM_targeted

Adversarial docs are optimized against a specific retriever (Contriever).
They don't transfer well to a different ranking model (BM25).
Docs that rank high in Contriever but low in BM25 are flagged and removed.
"""

import math
import re
from collections import Counter
from typing import List


def _tokenize(text: str) -> List[str]:
    return re.findall(r'\w+', text.lower())


def _bm25_scores(query: str, docs: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    """Lightweight BM25 implementation — no external dependencies."""
    tokenized_docs = [_tokenize(d) for d in docs]
    tokenized_query = _tokenize(query)
    avgdl = sum(len(d) for d in tokenized_docs) / max(len(tokenized_docs), 1)
    N = len(tokenized_docs)

    # IDF for each query term
    idf = {}
    for term in set(tokenized_query):
        df = sum(1 for d in tokenized_docs if term in d)
        idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

    scores = []
    for doc_tokens in tokenized_docs:
        tf_map = Counter(doc_tokens)
        dl = len(doc_tokens)
        score = 0.0
        for term in tokenized_query:
            tf = tf_map.get(term, 0)
            score += idf.get(term, 0) * (tf * (k1 + 1)) / (
                tf + k1 * (1 - b + b * dl / max(avgdl, 1))
            )
        scores.append(score)
    return scores


class MultiModelReranker:
    """
    Defense against LM_targeted attack.

    Re-ranks retrieved documents using BM25 alongside the original dense
    retriever scores. Documents that score high in dense retrieval but low
    in BM25 are considered suspicious and removed.

    Args:
        alpha (float): Weight for dense score in combined ranking (0-1).
                       Lower alpha = more weight on BM25 = stricter defense.
        anomaly_threshold (float): If BM25 rank is this many positions worse
                                   than dense rank, the doc is flagged.
        top_k (int): Number of docs to keep after re-ranking.
    """

    def __init__(self, alpha: float = 0.5, anomaly_threshold: int = 3, top_k: int = 5, bm25_min_ratio: float = 0.1):
        self.alpha = alpha
        self.anomaly_threshold = anomaly_threshold
        self.top_k = top_k
        self.bm25_min_ratio = bm25_min_ratio  # docs scoring below this fraction of max BM25 are flagged

    def defend(self, query: str, docs: List[str], dense_scores: List[float]) -> List[str]:
        """
        Apply multi-model re-ranking defense.

        Args:
            query: The user query string.
            docs: Retrieved document texts (already in top-k order).
            dense_scores: Corresponding dense retriever scores.

        Returns:
            Filtered and re-ranked list of documents.
        """
        if not docs:
            return docs

        bm25 = _bm25_scores(query, docs)

        # Normalize both score lists to [0, 1]
        def normalize(scores):
            mn, mx = min(scores), max(scores)
            if mx == mn:
                return [1.0] * len(scores)
            return [(s - mn) / (mx - mn) for s in scores]

        norm_dense = normalize(dense_scores)
        norm_bm25 = normalize(bm25)

        # Strategy 1: absolute BM25 score filter
        # Adversarial docs crafted for embedding similarity often have low
        # absolute BM25 scores because they don't contain the query's keywords.
        # Docs with near-zero BM25 score are suspicious.
        max_bm25 = max(bm25) if max(bm25) > 0 else 1.0
        filtered = []
        for i, doc in enumerate(docs):
            bm25_ratio = bm25[i] / max_bm25
            # Strategy 2: rank gap anomaly (secondary check)
            dense_rank = sorted(range(len(docs)), key=lambda j: norm_dense[j], reverse=True).index(i) \
                if i in range(len(docs)) else 0

            # Flag if BM25 score is very low relative to the best BM25 doc
            if bm25_ratio < self.bm25_min_ratio:
                print(f"[MultiModelReranker] Flagged doc {i} "
                      f"(bm25={bm25[i]:.3f}, ratio={bm25_ratio:.3f} < threshold={self.bm25_min_ratio})")
                continue

            combined = self.alpha * norm_dense[i] + (1 - self.alpha) * norm_bm25[i]
            filtered.append((combined, doc))

        if not filtered:
            print("[MultiModelReranker] Warning: all docs flagged — returning top BM25 doc only.")
            best_idx = max(range(len(docs)), key=lambda i: bm25[i])
            return [docs[best_idx]]

        filtered.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in filtered[:self.top_k]]
