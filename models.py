"""The two models the system uses, both reached through OpenRouter.

OpenRouter speaks the same HTTP API as OpenAI, so the official `openai`
client works: we just point it at a different base URL.
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # copy .env into os.environ (real env vars win if both are set)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


def embed(texts: list[str]) -> list[list[float]]:
    """Turn each text into a vector (a list of EMBED_DIM floats), same order as input."""
    vectors = []
    # The API caps how much you can send per request, so big documents go in batches.
    for i in range(0, len(texts), 100):
        resp = client.embeddings.create(model=os.environ["EMBED_MODEL"], input=texts[i:i + 100])
        # resp.data is a list of openai `Embedding` objects, one per input text.
        # Each has .index (which input it belongs to) and .embedding (the floats).
        items = sorted(resp.data, key=lambda item: item.index)
        vectors += [item.embedding for item in items]
    return vectors


def chat(system: str, user: str) -> str:
    """One question in, one answer out. No conversation memory."""
    resp = client.chat.completions.create(
        model=os.environ["LLM_MODEL"],
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,  # least random: we want faithful answers, not creative ones
    )
    choice = resp.choices[0]
    # "stop" = the model finished its answer. Anything else ("length", "error", ...) means
    # the text may be cut off; never pass a half answer off as a whole one.
    if choice.finish_reason != "stop":
        raise RuntimeError(f"LLM stopped early (finish_reason={choice.finish_reason!r}), please retry")
    return choice.message.content


if __name__ == "__main__":
    # Self-check: `python models.py`
    import math

    def cosine(a, b):
        return math.sumprod(a, b) / (math.hypot(*a) * math.hypot(*b))

    cat, kitten, tax = embed([
        "The cat sat on the mat.",
        "A kitten is resting on a rug.",
        "Quarterly tax filings are due in April.",
    ])
    print(f"dimensions: {len(cat)} (EMBED_DIM={os.environ['EMBED_DIM']})")
    print(f"first 5 numbers of 'cat': {[round(x, 4) for x in cat[:5]]}")
    print(f"length of 'cat' vector: {math.hypot(*cat):.4f}")
    print(f"cat vs kitten similarity: {cosine(cat, kitten):.3f}")
    print(f"cat vs tax similarity:    {cosine(cat, tax):.3f}")
    assert len(cat) == int(os.environ["EMBED_DIM"]), "EMBED_DIM in .env doesn't match the model"
    assert cosine(cat, kitten) > cosine(cat, tax), "similar sentences should score higher"

    print("LLM:", chat("Answer in five words or fewer.", "what is the capital of france"))
    print("models OK")
