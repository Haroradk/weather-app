"""
The pipeline's only two uses of an LLM, behind one small interface:
structured extraction (text in, typed fields out) and embeddings (text in,
vector out). Kept in one module so every Gemini call gets the same retry
behaviour and the same models.

Treat everything that comes back as untrusted input: the extraction schema
constrains its *shape*, but not whether its *content* is true - that's
what the checks in discussion_silver.py are for.
"""

from functools import lru_cache

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from config import EMBEDDING_DIMENSIONS, GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL, GEMINI_EXTRACTION_MODEL


def is_available() -> bool:
    return bool(GEMINI_API_KEY)


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY)


def _is_retryable(exception: BaseException) -> bool:
    # 503 overload and 429 rate limit fix themselves with time; 400/403 don't.
    if isinstance(exception, genai_errors.ServerError):
        return True
    return isinstance(exception, genai_errors.ClientError) and exception.code == 429


_retry = retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    reraise=True,
)


@_retry
def extract(prompt: str, schema):
    """Returns an instance of `schema` (a Pydantic model). Gemini's
    structured output guarantees valid JSON of that shape, so there's no
    free-text parsing - but the values themselves still need checking."""
    response = _client().models.generate_content(
        model=GEMINI_EXTRACTION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0,
        ),
    )
    return response.parsed


@_retry
def embed(texts: list, task_type: str = "RETRIEVAL_DOCUMENT") -> list:
    """One vector per text, in one request. task_type matters: documents
    and search queries are embedded slightly differently so a short
    question can still land near the long passage that answers it."""
    result = _client().models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBEDDING_DIMENSIONS),
    )
    return [e.values for e in result.embeddings]
