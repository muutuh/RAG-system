"""Command-line entry point.

    python cli.py add <file-or-folder> [...]   store documents (.pdf / .txt / .md)
    python cli.py ask <question>               answer one question
    python cli.py ask                          interactive: keep asking until an empty line
    python cli.py list                         show stored documents
"""
import argparse
from pathlib import Path

import openai
import psycopg
from pypdf.errors import PdfReadError

from graph import ask, label
from ingestion import SUPPORTED
from storage import add_document, list_documents


def expand(paths: list[str]):
    """Yield files; folders are searched (including subfolders) for supported files.
    Needed on Windows: PowerShell doesn't expand `docs\\*.pdf` for Python."""
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED)
        else:
            yield p


def cmd_add(args):
    for path in expand(args.paths):
        try:
            print(add_document(path))
        except (OSError, ValueError, PdfReadError) as e:  # one bad file shouldn't stop the batch
            print(f"error {path.name}: {e}")


def answer(question: str):
    try:
        result = ask(question)
    except (openai.APIError, RuntimeError) as e:  # network/model trouble: report, don't crash
        print(f"error: {e}")
        return
    print(f"\n{result['answer']}")
    if result["sources"]:
        print("\nSources:")
        for s in result["sources"]:
            print(f"  [{s['n']}] {label(s)}")


def cmd_ask(args):
    if args.question:
        answer(" ".join(args.question))
        return
    print("Ask about your documents. Press Enter on an empty line to quit.")
    try:
        while question := input("\n> ").strip():
            answer(question)
    except (EOFError, KeyboardInterrupt):  # Ctrl+Z / Ctrl+C quits quietly
        print()


def cmd_list(args):
    docs = list_documents()
    if not docs:
        print("No documents yet. Add some with: python cli.py add <file-or-folder>")
    for source, chunks, pages, added in docs:
        pages_text = f"{pages} pages, " if pages else ""
        print(f"{source}  ({pages_text}{chunks} chunks, added {added:%Y-%m-%d %H:%M})")


def main():
    parser = argparse.ArgumentParser(description="Ask questions about your documents.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("add", help="store files or folders (.pdf/.txt/.md)")
    p.add_argument("paths", nargs="+")
    p = sub.add_parser("ask", help="answer a question (no question = interactive mode)")
    p.add_argument("question", nargs="*")  # quotes optional: `ask When is the midterm?`
    sub.add_parser("list", help="show stored documents")
    args = parser.parse_args()

    commands = {"add": cmd_add, "ask": cmd_ask, "list": cmd_list}
    try:
        commands[args.command](args)
    except psycopg.OperationalError as e:
        print(f"Can't reach the database. Is Docker running? Try: docker compose up -d\n({e})")


if __name__ == "__main__":
    main()
