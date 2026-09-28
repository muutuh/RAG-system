"""The question-answering flow, as a LangGraph graph.

    START -> retrieve -> relevant chunks? --yes--> generate -> END
                                          --no---> refuse   -> END
"""
import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from models import chat
from retrieval import retrieve

NOT_FOUND_MESSAGE = "I couldn't find the answer to that in your documents."

RULES = """You answer questions using ONLY the numbered sources you are given.

Rules:
- Use no outside knowledge, even if you know the answer.
- After every claim, cite the source number(s) in square brackets, like [1] or [2][3].
- If the sources do not contain the answer, reply with exactly: NOT_FOUND
- If the sources answer only part of the question, answer that part and say plainly what is not covered.
- Copy numbers, dates and names exactly as they appear in the sources.
- Be concise."""


class State(TypedDict, total=False):
    """The shared state every node reads from and writes to."""
    question: str        # set by the caller
    chunks: list[dict]   # set by retrieve
    answer: str          # set by generate or refuse
    sources: list[dict]  # set by generate: only the chunks the answer actually cites


def label(chunk: dict) -> str:
    """'syllabus.pdf, page 3, chunk 7' (no page for .txt/.md)."""
    page = f", page {chunk['page']}" if chunk["page"] is not None else ""
    return f"{chunk['source']}{page}, chunk {chunk['chunk_index']}"


# --- nodes: plain functions that take the state and return the keys they change ---

def retrieve_node(state: State) -> State:
    return {"chunks": retrieve(state["question"])}


def generate(state: State) -> State:
    chunks = state["chunks"]
    sources_text = "\n\n".join(f"[{n}] ({label(c)})\n{c['content']}" for n, c in enumerate(chunks, 1))
    answer = chat(RULES, f"Sources:\n\n{sources_text}\n\nQuestion: {state['question']}").strip()

    if answer == "NOT_FOUND":  # second safety layer: relevant-looking chunks, but no answer in them
        return {"answer": NOT_FOUND_MESSAGE, "sources": []}

    # Map the [n] markers the LLM wrote back to real chunks; ignore numbers that don't exist.
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    sources = [{"n": n, **chunks[n - 1]} for n in cited if 1 <= n <= len(chunks)]
    return {"answer": answer, "sources": sources}


def refuse(state: State) -> State:
    return {"answer": NOT_FOUND_MESSAGE, "sources": []}


# --- edge logic: decides which node runs next ---

def has_relevant_chunks(state: State) -> str:
    return "generate" if state["chunks"] else "refuse"


builder = StateGraph(State)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate)
builder.add_node("refuse", refuse)
builder.add_edge(START, "retrieve")
builder.add_conditional_edges("retrieve", has_relevant_chunks, ["generate", "refuse"])
builder.add_edge("generate", END)
builder.add_edge("refuse", END)
graph = builder.compile()


def ask(question: str) -> State:
    """Run the whole flow; returns the final state (answer + sources)."""
    return graph.invoke({"question": question})


if __name__ == "__main__":
    import sys

    def run(question: str) -> State:
        # stream() yields each node's changes as it finishes, so we can show the path taken.
        path, state = [], {}
        for update in graph.stream({"question": question}, stream_mode="updates"):
            for node, changes in update.items():
                path.append(node)
                state.update(changes)
        print(f"\nQ: {question}\n   path: {' -> '.join(path)}\nA: {state['answer']}")
        for s in state["sources"]:
            print(f"   [{s['n']}] {label(s)}")
        return state

    if sys.argv[1:]:
        for q in sys.argv[1:]:
            run(q)
    else:
        # Self-check on the sample handbook (must be stored, see step 4)
        found = run("How many vacation days do full-time employees get?")
        assert "23" in found["answer"] and found["sources"], "should answer 23 days, with a citation"
        not_in_docs = run("How much do baristas get paid per hour?")
        assert not_in_docs["answer"] == NOT_FOUND_MESSAGE, "on-topic but unanswerable: LLM must refuse"
        off_topic = run("What is the capital of France?")
        assert off_topic["answer"] == NOT_FOUND_MESSAGE, "off-topic: threshold must refuse"
        print("\ngraph OK")
