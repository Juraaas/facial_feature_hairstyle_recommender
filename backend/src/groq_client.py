import os
import time
from groq import Groq, RateLimitError, APIConnectionError, APITimeoutError
from src.logger import get_logger

log = get_logger(__name__)

_client = None
_paused_until = 0.0


def get_client():
    global _client
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    if _client is None:
        _client = Groq(api_key=key, timeout=10.0, max_retries=0)
    return _client


def available() -> bool:
    return time.monotonic() >= _paused_until


def handle_error(e: Exception, where: str):
    global _paused_until
    if isinstance(e, RateLimitError):
        retry = None
        resp = getattr(e, "response", None)
        if resp is not None:
            retry = resp.headers.get("retry-after")
        try:
            wait = min(float(retry), 120) if retry else 30
        except ValueError:
            wait = 30
        _paused_until = time.monotonic() + wait
        log.warning("GROQ_RATE_LIMITED where=%s retry_after=%s pause=%ss", where, retry, wait)
    elif isinstance(e, (APIConnectionError, APITimeoutError)):
        log.warning("GROQ_UNAVAILABLE where=%s", where)
    else:
        log.exception("GROQ_ERROR where=%s", where)