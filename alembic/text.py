"""Safe text normalization shared by document ingestion and dataset cleaning."""

import re
import unicodedata


_REMOVABLE_INVISIBLE = frozenset(
    {
        "\u00ad",  # soft hyphen
        "\u200b",  # zero-width space
        "\u200e",  # left-to-right mark
        "\u200f",  # right-to-left mark
        "\u202a",  # bidi embedding / override controls
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2060",  # word joiner
        "\u2061",
        "\u2062",
        "\u2063",
        "\u2064",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
        "\ufeff",  # byte-order mark embedded in decoded text
        "\ufffd",  # Unicode replacement character from broken decoding
    }
)


def normalize_text(text: str) -> str:
    """Normalize noisy Unicode without removing meaningful document syntax.

    The operation is deliberately conservative: Markdown punctuation, formulas,
    emoji, and the zero-width joiner/non-joiner used by some languages are kept.
    Only transport noise, unsafe controls, and layout inconsistencies are removed.
    """

    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)

    normalized: list[str] = []
    for char in text:
        if char in _REMOVABLE_INVISIBLE:
            continue
        if char == "\t":
            normalized.append("    ")
            continue
        if char == "\n":
            normalized.append(char)
            continue

        category = unicodedata.category(char)
        if category in {"Cc", "Cs"}:  # controls and lone surrogate code points
            continue
        if category == "Zs":  # NBSP and other Unicode horizontal spaces
            normalized.append(" ")
            continue
        normalized.append(char)

    result = "".join(normalized)
    result = "\n".join(line.rstrip() for line in result.split("\n"))
    result = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", result)
    return result.strip()
