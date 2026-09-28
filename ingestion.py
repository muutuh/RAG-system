"""Turn a file into small, overlapping, labelled pieces of text ("chunks")."""
from pathlib import Path

from pypdf import PdfReader

CHUNK_SIZE = 1000    # max characters per chunk (~250 tokens)
CHUNK_OVERLAP = 200  # characters repeated between neighbouring chunks
SUPPORTED = (".pdf", ".txt", ".md")


def load_pages(path: Path) -> list[tuple[int | None, str]]:
    """Return (page_number, text) pairs. .txt/.md have no pages, so page is None."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        # extract_text() reads the PDF's text layer; it returns "" for scanned images.
        return [(n, page.extract_text() or "") for n, page in enumerate(PdfReader(path).pages, start=1)]
    if suffix in SUPPORTED:  # .txt / .md
        # errors="replace": a stray non-UTF-8 byte becomes a replacement character instead of crashing.
        return [(None, path.read_text(encoding="utf-8", errors="replace"))]
    raise ValueError(f"unsupported file type: {path.name} (use .pdf, .txt or .md)")


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Cut text into pieces of at most `size` chars, preferring natural break points."""
    text = text.strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Cut at the best break in the second half of the window:
            # paragraph > line > sentence > word. Never mid-word.
            for sep in ("\n\n", "\n", ". ", " "):
                cut = text.rfind(sep, start + size // 2, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        # Step back `overlap` chars so a sentence cut at the edge also appears whole
        # in the next chunk, then move forward to a word start.
        start = max(end - overlap, start + 1)
        while start < end and not text[start - 1].isspace():
            start += 1
    return [c for c in chunks if c]


def load_chunks(path: str | Path) -> list[dict]:
    """File -> [{"index": 0, "page": 1, "text": "..."}, ...] in reading order."""
    path = Path(path)
    chunks = []
    # ponytail: chunks never cross a page boundary, so each cites exactly one page.
    # A paragraph split across two pages becomes two chunks; merge pages if answers suffer.
    for page, text in load_pages(path):
        # PDFs sometimes contain NUL characters; Postgres refuses them in text columns.
        for piece in split_text(text.replace("\x00", "")):
            chunks.append({"index": len(chunks), "page": page, "text": piece})
    if not chunks:
        raise ValueError(f"no text found in {path.name} (a scanned PDF needs OCR first)")
    return chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # `python ingestion.py some.pdf` -> show how that file gets chunked
        for c in load_chunks(sys.argv[1]):
            print(f"\n--- chunk {c['index']} | page {c['page']} | {len(c['text'])} chars ---")
            print(c["text"])
    else:
        # `python ingestion.py` -> self-check on synthetic text
        sentences = [f"Sentence number {i} says something." for i in range(300)]
        text = "\n\n".join(" ".join(sentences[i:i + 5]) for i in range(0, 300, 5))
        chunks = split_text(text)
        assert all(len(c) <= CHUNK_SIZE for c in chunks), "a chunk is too big"
        assert set(" ".join(chunks).split()) == set(text.split()), "words lost or cut in half"
        assert all(chunks[i + 1][:30] in chunks[i] for i in range(len(chunks) - 1)), "no overlap"
        print(f"split_text OK: {len(text)} chars -> {len(chunks)} chunks")
