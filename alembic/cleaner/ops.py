import re
import hashlib

_URL_PATTERN = re.compile(
    r'https?://[^\s<>"\'{}|\\^`\[\]]+|www\.[^\s<>"\'{}|\\^`\[\]]+',
    re.IGNORECASE,
)

_HTML_PATTERN = re.compile(r'<[^>]+>')

_EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

_MARKDOWN_LINK_PATTERN = re.compile(r'\[([^\]]*)\]\([^)]+\)')
_MARKDOWN_IMAGE_PATTERN = re.compile(r'!\[[^\]]*\]\([^)]+\)')


def remove_urls(text: str) -> str:
    return _URL_PATTERN.sub('', text)


def remove_html(text: str) -> str:
    return _HTML_PATTERN.sub('', text)


def remove_emails(text: str) -> str:
    return _EMAIL_PATTERN.sub('', text)


def remove_markdown(text: str) -> str:
    text = _MARKDOWN_IMAGE_PATTERN.sub('', text)
    text = _MARKDOWN_LINK_PATTERN.sub(r'\1', text)
    return text


def clean_text(text: str, remove_html_tags: bool = True, remove_url_links: bool = True, remove_email_addr: bool = True) -> str:
    if remove_html_tags:
        text = remove_html(text)
    if remove_url_links:
        text = remove_urls(text)
    if remove_email_addr:
        text = remove_emails(text)
    return text


_SPECIAL_CHARS_PATTERN = re.compile(r'[^a-zA-Z0-9\u4e00-\u9fff\u3400-\u4dbf\s.,!?;:\'"()\-_+=/\\@#$%^&*\[\]{}|<>`~]')


def special_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    count = len(_SPECIAL_CHARS_PATTERN.findall(text))
    return count / len(text)


def word_repetition_ratio(text: str, window: int = 5) -> float:
    words = text.lower().split()
    if len(words) < window:
        return 0.0
    seen = set()
    repeat = 0
    for i in range(len(words) - window + 1):
        ngram = tuple(words[i:i + window])
        if ngram in seen:
            repeat += 1
        else:
            seen.add(ngram)
    return repeat / max(len(words) - window + 1, 1)


def char_repetition_ratio(text: str, min_len: int = 5) -> float:
    if len(text) < min_len:
        return 0.0
    max_run = 1
    current_run = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    if max_run < min_len:
        return 0.0
    return max_run / len(text)


def compute_dedup_key(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()
