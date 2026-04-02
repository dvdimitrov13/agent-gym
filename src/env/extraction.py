import requests
import trafilatura


JINA_READER_URL = "https://r.jina.ai/"


def _extract_with_jina(url: str, timeout: int = 15) -> str | None:
    """Use Jina Reader API to get clean Markdown. Returns None on failure."""
    try:
        resp = requests.get(
            f"{JINA_READER_URL}{url}",
            headers={"Accept": "text/markdown"},
            timeout=timeout,
        )
        if resp.status_code == 200 and resp.text.strip():
            return resp.text.strip()
    except Exception:
        pass
    return None


def _extract_with_trafilatura(url: str) -> str | None:
    """Use trafilatura as local fallback. Returns None on failure."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return None
        text = trafilatura.extract(downloaded, include_links=False, include_tables=True)
        return text if text else None
    except Exception:
        return None


def fetch_and_extract(url: str, timeout: int = 15) -> str:
    """Fetch a URL and extract content. Tries Jina Reader first (Markdown),
    falls back to trafilatura (plain text). Returns extracted text or error string."""
    # Try Jina Reader first — returns clean Markdown with structure
    result = _extract_with_jina(url, timeout=timeout)
    if result:
        return result

    # Fallback to trafilatura — local, no API dependency
    result = _extract_with_trafilatura(url)
    if result:
        return result

    return f"[Error: could not extract content from {url}]"
