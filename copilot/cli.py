"""Command-line interface.

python -m copilot.cli ingest
python -m copilot.cli search "Nike gross margin fiscal 2025"
python -m copilot.cli ask "How did Nike's gross margin change in fiscal 2025?"
python -m copilot.cli agent "By what percentage did Nike's net income fall in fiscal 2025?"
"""

import argparse

from copilot import config


def cmd_ingest(_: argparse.Namespace) -> None:
    """Chunk and embed the reports and write the Chroma index."""
    from copilot.ingest import build_index, load_corpus

    chunks = load_corpus()
    print(f"Chunked {len(config.DOCUMENTS)} reports into {len(chunks)} chunks. Embedding...")
    build_index(chunks)
    print(f"Index written to {config.CHROMA_DIR}")


def cmd_search(args: argparse.Namespace) -> None:
    """Print the retrieved passages for a question."""
    from copilot.retrieval import build_retriever

    for rank, hit in enumerate(build_retriever(args.retriever).search(args.question, k=args.k), start=1):
        print(f"{rank}. {hit.score:.3f}  {hit.chunk.citation}\n   {hit.chunk.text[:200]}...\n")


def cmd_ask(args: argparse.Namespace) -> None:
    """Answer a question with single-shot RAG (no tools)."""
    from copilot.generate import answer_question
    from copilot.retrieval import build_retriever

    hits = build_retriever(args.retriever).search(args.question, k=args.k)
    answer = answer_question(args.question, hits)
    print(f"Answer: {answer.text}\n")
    print("Sources: " + "  ".join(f"[{i}] {s}" for i, s in enumerate(answer.sources, start=1)))


def cmd_agent(args: argparse.Namespace) -> None:
    """Answer a question with the tool-calling agent and print its trace."""
    from copilot.agent import ResearchAgent

    result = ResearchAgent().run(args.question)
    print(f"Answer: {result.answer}")
    if result.sources:
        print("Sources: " + "  ".join(f"[{i}] {s}" for i, s in enumerate(result.sources, start=1)))
    print("Tool calls: " + (", ".join(f"{name} ({n})" for name, n in result.tool_calls.items()) or "none"))
    print(
        f"Fallback triggered: {str(result.fallback_triggered).lower()}"
        + (f" ({result.fallback_reason})" if result.fallback_reason else "")
    )


def main() -> None:
    """Parse command-line arguments and run the chosen command."""
    parser = argparse.ArgumentParser(prog="copilot", description="Q&A over athletic apparel annual reports")
    sub = parser.add_subparsers(required=True)
    sub.add_parser("ingest", help="chunk and embed the reports in data/").set_defaults(func=cmd_ingest)
    for name, func, help_text in [
        ("search", cmd_search, "show the retrieved passages (no API key needed)"),
        ("ask", cmd_ask, "answer a question with citations"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("question")
        p.add_argument("-k", type=int, default=config.TOP_K, help="number of passages to retrieve")
        p.add_argument(
            "--retriever",
            default=config.RETRIEVER_MODE,
            choices=["vector", "bm25", "hybrid", "hybrid_rerank", "hybrid_rerank_company"],
        )
        p.set_defaults(func=func)
    p = sub.add_parser("agent", help="tool-calling agent with calculator and fallback guardrail")
    p.add_argument("question")
    p.set_defaults(func=cmd_agent)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
