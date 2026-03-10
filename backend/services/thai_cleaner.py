"""Thai text cleaning service using PyThaiNLP."""

import re

try:
    from pythainlp.util import normalize as thai_normalize
    from pythainlp.tokenize import word_tokenize
    HAS_PYTHAINLP = True
except ImportError:
    HAS_PYTHAINLP = False


def clean_thai_text(raw: str) -> str:
    """Clean and normalize Thai text.

    - Normalize Thai characters (fix floating vowels, etc.)
    - Remove excessive whitespace
    - Re-segment words for proper spacing
    """
    if not raw or not raw.strip():
        return ""

    text = raw.strip()

    # Remove null bytes and control characters (except newlines/tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Normalize unicode
    if HAS_PYTHAINLP:
        text = thai_normalize(text)

    # Collapse multiple spaces/tabs into single space (preserve newlines)
    text = re.sub(r'[^\S\n]+', ' ', text)

    # Collapse multiple newlines into double newline
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


def segment_thai_text(text: str) -> str:
    """Re-segment Thai text using PyThaiNLP word tokenizer."""
    if not HAS_PYTHAINLP or not text:
        return text

    # Only apply to lines that contain Thai characters
    thai_pattern = re.compile(r'[\u0E00-\u0E7F]')
    lines = text.split('\n')
    result_lines = []

    for line in lines:
        if thai_pattern.search(line):
            tokens = word_tokenize(line, engine="newmm")
            result_lines.append("".join(tokens))
        else:
            result_lines.append(line)

    return '\n'.join(result_lines)
