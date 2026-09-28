"""Find the stored chunks most relevant to a question."""
from models import embed
from storage import connect

TOP_K = 5  # how many chunks to hand to the LLM at most

# Cosine distance cut-off: chunks further away than this are treated as "not relevant".
# Measured with `python retrieval.py` on the sample handbook (text-embedding-3-small):
# answerable questions scored 0.53-0.70, unrelated ones 0.87+. It can only filter
# OFF-TOPIC questions; "on-topic but not in the docs" is the LLM prompt's job (step 6).
# ponytail: calibrated on one small document; re-run the calibration on your real docs,
# and again if you change EMBED_MODEL or CHUNK_SIZE.
MAX_DISTANCE = 0.80


def search(question: str, k: int = TOP_K) -> list[dict]:
    """The k nearest chunks to the question, nearest first, each with its distance."""
    [q] = embed([question])  # same model as the chunks, so the vectors are comparable
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT d.source, c.page, c.chunk_index, c.content,
                   c.embedding <=> %(q)s::vector AS distance
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            -- ORDER BY the distance expression + LIMIT is the pattern the HNSW index speeds up
            ORDER BY c.embedding <=> %(q)s::vector
            LIMIT %(k)s
            """,
            {"q": q, "k": k},
        ).fetchall()
    keys = ("source", "page", "chunk_index", "content", "distance")
    return [dict(zip(keys, row)) for row in rows]


def retrieve(question: str) -> list[dict]:
    """search() minus anything too far away to be relevant. May return []."""
    return [c for c in search(question) if c["distance"] <= MAX_DISTANCE]


if __name__ == "__main__":
    import sys

    questions = sys.argv[1:] or [
        # answerable from the sample handbook
        "How many vacation days do full-time employees get?",
        "Who keeps track of when grinder burrs are replaced?",
        "Is there a bonus for finishing the cupping course?",
        "can i take my break at 8am",
        # on-topic for a workplace, but NOT in the handbook
        "What is the parental leave policy?",
        "How much do baristas get paid per hour?",
        # completely unrelated
        "What is the capital of France?",
        "How do I bake sourdough bread?",
    ]
    for question in questions:
        print(f"\nQ: {question}")
        for c in search(question):
            mark = "keep" if c["distance"] <= MAX_DISTANCE else "drop"
            preview = c["content"][:60].replace("\n", " ")
            print(f"  {c['distance']:.3f} {mark}  {c['source']} #{c['chunk_index']}  {preview}")

    if not sys.argv[1:]:
        # Self-check (needs samples/nimbus-handbook.md stored, see step 4)
        assert retrieve(questions[0]), "an answerable question found nothing"
        assert not retrieve("What is the capital of France?"), "an unrelated question got through"
        print("\nretrieval OK")
