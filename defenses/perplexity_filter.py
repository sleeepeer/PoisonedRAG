"""
Perplexity Filter Defense
Target attack: HotFlip

HotFlip produces token sequences that fool the retriever but break natural
language fluency. A language model perplexity scorer catches these because
token-flipped text has abnormally high perplexity.
"""

import math
from typing import List, Optional


class PerplexityFilter:
    """
    Defense against HotFlip attack.

    Scores each retrieved document using a language model perplexity scorer.
    Documents with perplexity above a threshold are considered adversarial
    and removed from the context before the LLM query.

    Args:
        threshold (float): Perplexity cutoff. Docs above this are removed.
                           Default 200 works well for HotFlip artifacts.
        model_name (str): HuggingFace model to use for perplexity scoring.
                          GPT-2 is fast and sufficient.
    """

    def __init__(self, threshold: float = 200.0, model_name: str = "gpt2"):
        self.threshold = threshold
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy-load the perplexity model on first use."""
        if self._model is not None:
            return
        try:
            import torch
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast
            self._tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
            self._model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self._model.eval()
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._model.to(self._device)
        except Exception as e:
            print(f"[PerplexityFilter] Warning: Could not load {self.model_name}: {e}")
            self._model = None

    def _perplexity(self, text: str) -> float:
        """Compute perplexity of a single text string."""
        import torch
        self._load_model()
        if self._model is None:
            return 0.0  # fail open if model unavailable

        encodings = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        input_ids = encodings.input_ids.to(self._device)

        with torch.no_grad():
            outputs = self._model(input_ids, labels=input_ids)
            loss = outputs.loss
        return math.exp(loss.item())

    def defend(self, docs: List[str]) -> List[str]:
        """
        Filter out documents with abnormally high perplexity.

        Args:
            docs: List of retrieved document texts.

        Returns:
            Filtered list with high-perplexity (likely adversarial) docs removed.
        """
        if not docs:
            return docs

        filtered = []
        for i, doc in enumerate(docs):
            ppl = self._perplexity(doc)
            if ppl > self.threshold:
                print(f"[PerplexityFilter] Removed doc {i} (perplexity={ppl:.1f} > threshold={self.threshold})")
            else:
                filtered.append(doc)

        if not filtered:
            print("[PerplexityFilter] All docs filtered — LLM will answer from own knowledge.")
            return []  # empty list signals main.py to skip context entirely

        return filtered
