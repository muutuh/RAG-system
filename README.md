# Document Q&A with RAG

Ask questions about your own documents (PDF, `.txt`, `.md`) and get answers that come **only** from those documents, with citations down to the file, page and chunk. If the answer isn't in your documents, it says so instead of guessing.

Built as a learning project: small, readable modules, with no framework layers hiding what happens.

```
> python cli.py ask When is the midterm and how much is it worth?

The midterm exam is on October 14, 2026, and it is worth 25% [2].

Sources:
  [2] 261_CS481_Syllabus.pdf, page 3, chunk 7
```

## How it works

```
ADD A DOCUMENT
file ─► sha256 already stored? ─yes─► skip
          │ no
          ▼
     pages ─► overlapping chunks (~1000 chars) ─► embeddings ─► Postgres + pgvector
                                                     (one transaction; same file name = replace)

ASK A QUESTION  (LangGraph)
question ─► retrieve: embed, top-5 nearest chunks by cosine distance, drop distance > 0.80
               │
               ├─ nothing left ──► refuse ("couldn't find it in your documents")
               │
               └─ chunks ──► generate: LLM answers ONLY from numbered sources, cites [n],
                             or replies NOT_FOUND ──► answer + cited sources
```

Guessing is blocked in two layers:
1. **Distance threshold:** off-topic questions never reach the LLM.
2. **Prompt:** on-topic questions whose answer isn't in the documents get `NOT_FOUND`.

| File | Role |
|---|---|
| `models.py` | `embed()` and `chat()` via OpenRouter (OpenAI-compatible API) |
| `ingestion.py` | Load PDF/txt/md, split into overlapping chunks with page numbers |
| `storage.py` | Schema, duplicate-safe `add_document()`, HNSW vector index |
| `retrieval.py` | Nearest-neighbour search + relevance threshold |
| `graph.py` | LangGraph flow: `retrieve → generate / refuse`, citation mapping |
| `cli.py` | `add`, `ask`, `list` commands |

## Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (runs Postgres + pgvector)
- Python 3.12+
- An [OpenRouter](https://openrouter.ai) API key. Default models cost about $0.003 to embed a 300-page book and about $0.0004 per question.

## Setup

```bash
git clone <this-repo-url>
cd <repo-folder>
```

Copy `.env.example` to `.env`, then put your OpenRouter key in `.env`:

```bash
cp .env.example .env
```

Start Postgres (data persists in a Docker volume):

```bash
docker compose up -d
```

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

On Windows, use `.venv\Scripts\pip` and `.venv\Scripts\python` instead of `.venv/bin/...`.

## Usage

Activate the virtual environment first: `source .venv/bin/activate`, or `.venv\Scripts\Activate.ps1` on Windows.

```bash
python cli.py add path/to/file.pdf path/to/folder   # folders are searched for .pdf/.txt/.md
python cli.py list                                  # stored documents
python cli.py ask When is the midterm?              # one question
python cli.py ask                                   # interactive mode; empty line quits
```

Adding a file that is already stored does nothing. Adding a changed file with the same name replaces the old version.

To start over with an empty database, run `docker compose down -v`. This **deletes all stored documents**.

## Self-checks

Each module runs a small check when executed directly. Checks from `retrieval.py` onwards need the sample document stored first:

```bash
python cli.py add samples
python models.py      # embeddings + LLM reachable, similar sentences score higher
python ingestion.py   # chunker: no oversized chunks, no lost or split words, overlap present
python storage.py     # duplicate rules (cleans up after itself)
python retrieval.py   # distance table for calibration questions
python graph.py       # answerable / on-topic-but-unanswerable / off-topic
```

`samples/nimbus-handbook.md` describes a fictional company, so an LLM can only know its facts by retrieving them.

## Tuning

| Setting | Where | Default | Notes |
|---|---|---|---|
| Chunk size / overlap | `ingestion.py` | 1000 / 200 chars | Smaller = sharper matches, less context per chunk |
| `TOP_K` | `retrieval.py` | 5 | Chunks handed to the LLM |
| `MAX_DISTANCE` | `retrieval.py` | 0.80 | Calibrate with `python retrieval.py "your questions"` on your own documents |
| Models | `.env` | gemini-2.5-flash-lite, text-embedding-3-small | Changing `EMBED_MODEL` requires re-ingesting |

## Known limitations

- Chunks don't cross page boundaries, and splitting ignores headings (no structure-aware chunking yet).
- Scanned PDFs without a text layer need OCR first.
- Documents are identified by file name: two different files with the same name replace each other.
- No conversation memory: each question is answered on its own.

## Ideas for improvement

1. An evaluation set of questions with expected answers, run after every change
2. Structure-aware chunking: split at headings, prefix each chunk with its heading
3. Hybrid search: Postgres full-text + vectors, for exact terms like course codes and names
4. Reranking the top 20 down to the best 5
5. Query rewriting: a LangGraph loop that retries retrieval with a rephrased question
6. Conversation memory with a LangGraph checkpointer
7. Checking that cited numbers and dates really appear in the cited chunk
