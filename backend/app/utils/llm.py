from app.config import settings

_claude_client = None

def claude_client():
    """One shared Anthropic client for every call site: connections are reused, and a request
    that hangs (dropped network) fails after claude_timeout_sec; the SDK retries twice."""
    global _claude_client
    if _claude_client is None:
        import anthropic
        _claude_client = anthropic.Anthropic(api_key=settings.claude_api_key,
                                             timeout=settings.claude_timeout_sec, max_retries=2)
    return _claude_client
