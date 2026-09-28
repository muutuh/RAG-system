"""Everything that writes to Postgres: the schema, and adding documents without duplicates."""
import hashlib
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from ingestion import load_chunks
from models import embed

load_dotenv()

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS documents (
    id       bigserial PRIMARY KEY,
    source   text NOT NULL UNIQUE,   -- file name; shown in citations
    sha256   text NOT NULL UNIQUE,   -- fingerprint of the file's exact bytes
    added_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          bigserial PRIMARY KEY,
    -- ON DELETE CASCADE: deleting a document deletes its chunks automatically.
    document_id bigint NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index int NOT NULL,
    page        int,                  -- NULL for .txt/.md
    content     text NOT NULL,
    embedding   vector({int(os.environ["EMBED_DIM"])}) NOT NULL,
    UNIQUE (document_id, chunk_index)
);

-- HNSW index for fast approximate nearest-neighbour search by cosine distance.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def connect() -> psycopg.Connection:
    """Open a connection, creating pgvector and our tables if they don't exist yet."""
    # autocommit: each statement saves immediately, unless inside `with conn.transaction()`.
    # connect_timeout: fail with an error instead of hanging if Postgres isn't reachable.
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True, connect_timeout=10)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)  # teach psycopg to convert `vector` values to/from Python
    # ponytail: schema check on every connect; fine for one user, use migrations if this grows.
    conn.execute(SCHEMA)
    return conn


def add_document(path: str | Path) -> str:
    """Chunk, embed and store a file. Calling it again on the same file does nothing.

    Identity rules:
      - same bytes as any stored document -> skip (even under another name)
      - same file name, different bytes   -> replace the old version
    """
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with connect() as conn:
        row = conn.execute("SELECT source FROM documents WHERE sha256 = %s", (digest,)).fetchone()
        if row:
            return f"skipped {path.name}: identical to stored '{row[0]}'"

        chunks = load_chunks(path)
        # Slow network call happens BEFORE we touch any rows, so a failure here changes nothing.
        vectors = embed([c["text"] for c in chunks])

        # All-or-nothing: if anything below fails, the old version stays exactly as it was.
        with conn.transaction():
            replaced = conn.execute("DELETE FROM documents WHERE source = %s", (path.name,)).rowcount
            doc_id = conn.execute(
                "INSERT INTO documents (source, sha256) VALUES (%s, %s) RETURNING id",
                (path.name, digest),
            ).fetchone()[0]
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (document_id, chunk_index, page, content, embedding)"
                    " VALUES (%s, %s, %s, %s, %s)",
                    [(doc_id, c["index"], c["page"], c["text"], v) for c, v in zip(chunks, vectors)],
                )

    return f"{'updated' if replaced else 'added'} {path.name}: {len(chunks)} chunks"


def list_documents() -> list[tuple]:
    """(source, chunk_count, last_page, added_at) for every stored document, oldest first."""
    with connect() as conn:
        return conn.execute(
            """
            SELECT d.source, count(c.id), max(c.page), d.added_at
            FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id
            ORDER BY d.added_at
            """
        ).fetchall()


if __name__ == "__main__":
    import sys
    import tempfile

    if len(sys.argv) > 1:
        # `python storage.py file1.pdf file2.md` -> store those files
        for p in sys.argv[1:]:
            print(add_document(p))
        sys.exit()

    # `python storage.py` -> self-check of the duplicate rules, cleans up after itself
    def counts():
        with connect() as conn:
            return conn.execute("SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks)").fetchone()

    before = counts()
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "dedup-test.md"
        f.write_text("Version one of the test document. " * 100)
        print(add_document(f))
        after_add = counts()
        assert after_add[0] == before[0] + 1

        print(add_document(f))
        assert counts() == after_add, "same file twice must not add rows"

        copy = Path(tmp) / "renamed-copy.md"
        copy.write_bytes(f.read_bytes())
        print(add_document(copy))
        assert counts() == after_add, "same content under another name must not add rows"

        f.write_text("Version two is short.")
        print(add_document(f))
        assert counts() == (after_add[0], before[1] + 1), "new version must replace old chunks"

    with connect() as conn:
        conn.execute("DELETE FROM documents WHERE source = 'dedup-test.md'")
    assert counts() == before
    print("storage OK")
