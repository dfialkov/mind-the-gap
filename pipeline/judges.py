"""Answer extraction (regex) and Claude-API CoT/answer acknowledgement judges."""
import re

_BOXED = re.compile(r"\\boxed\{\s*([ABCD])\s*\}")
_ANSWER_IS = re.compile(
    r"(?:answer(?:\s+is)?|final answer|correct answer)\s*[:\*]*\s*\(?([ABCD])\)?",
    re.IGNORECASE,
)
_LETTER = re.compile(r"\b([ABCD])\b")


def extract_answer(response: str) -> str | None:
    """Best-effort A/B/C/D extraction from the post-`</think>` response text."""
    if not response:
        return None
    m = _BOXED.search(response)
    if m:
        return m.group(1)
    m = _ANSWER_IS.search(response)
    if m:
        return m.group(1).upper()
    matches = _LETTER.findall(response)
    return matches[-1] if matches else None
