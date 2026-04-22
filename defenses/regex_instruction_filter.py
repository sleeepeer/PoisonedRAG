"""
Regex Instruction Filter Defense
Target attack: Prompt Injection

Prompt injection embeds explicit override instructions in retrieved docs
(e.g. SYSTEM OVERRIDE, [CRITICAL], AUTHORITATIVE SOURCE).
These patterns are directly detectable via regex — cheapest defense in the stack.
"""

import re
from typing import List, Tuple

# Patterns matching known prompt injection templates used in this project
# and common real-world injection patterns
INJECTION_PATTERNS = [
    r"SYSTEM\s+OVERRIDE",
    r"AUTHORITATIVE\s+SOURCE",
    r"\[CRITICAL\]",
    r"VERIFIED\s+FACT",
    r"internal\s+documentation.*OFFICIAL",
    r"Disregard\s+all\s+other\s+sources",
    r"Ignore\s+conflicting\s+information",
    r"Follow\s+this\s+exclusively",
    r"takes?\s+precedence\s+over",
    r"All\s+other\s+sources\s+are\s+outdated",
    r"Use\s+this\s+answer",
    r"ignore\s+(all\s+)?(previous|prior|other)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|other)\s+instructions?",
    r"you\s+(must|should|shall)\s+(now\s+)?answer",
    r"override\s+(previous\s+)?instructions?",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def _is_injected(text: str) -> Tuple[bool, str]:
    """Returns (is_injected, matched_pattern)."""
    for pattern in _COMPILED:
        m = pattern.search(text)
        if m:
            return True, m.group(0)
    return False, ""


class RegexInstructionFilter:
    """
    Defense against Prompt Injection attack.

    Scans each retrieved document for known injection patterns before
    passing context to the LLM. Flagged documents are removed entirely.

    Args:
        custom_patterns (list[str], optional): Additional regex patterns to check.
        sanitize (bool): If True, strip the injection phrase instead of removing
                         the whole document. Default False (remove whole doc).
    """

    def __init__(self, custom_patterns: List[str] = None, sanitize: bool = False):
        self.sanitize = sanitize
        self._patterns = list(_COMPILED)
        if custom_patterns:
            self._patterns += [re.compile(p, re.IGNORECASE) for p in custom_patterns]

    def _clean(self, text: str) -> str:
        """Strip injection phrases from text (used when sanitize=True)."""
        for pattern in self._patterns:
            text = pattern.sub("[REMOVED]", text)
        return text.strip()

    def defend(self, docs: List[str]) -> List[str]:
        """
        Filter or sanitize documents containing injection patterns.

        Args:
            docs: List of retrieved document texts.

        Returns:
            Cleaned list with injected documents removed (or sanitized).
        """
        if not docs:
            return docs

        filtered = []
        for i, doc in enumerate(docs):
            injected, matched = _is_injected(doc)
            if injected:
                if self.sanitize:
                    cleaned = self._clean(doc)
                    print(f"[RegexFilter] Sanitized doc {i} — matched: '{matched}'")
                    filtered.append(cleaned)
                else:
                    print(f"[RegexFilter] Removed doc {i} — matched: '{matched}'")
            else:
                filtered.append(doc)

        if not filtered:
            print("[RegexFilter] All docs filtered — LLM will answer from own knowledge.")
            return []

        return filtered
