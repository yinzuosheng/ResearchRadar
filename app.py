import argparse
from dotenv import load_dotenv
load_dotenv()

from agent.react_agent import ReactAgent
from utils.pipeline import collect_and_ingest, collect_papers_and_ingest, generate_brief, run_daily_brief
from utils.scheduler import start_scheduler


def _split_lines(text: str | None) -> list[str] | None:
    if not text:
        return None
    return [line.strip() for line in text.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Intel RAG Agent")
    subparsers = parser.add_subparsers(dest="command")

    collect_parser = subparsers.add_parser("collect", help="Search and ingest sources")
    collect_parser.add_argument("--topics", help="Newline-separated topics")
    collect_parser.add_argument("--provider", choices=["tavily", "bing"], help="Search provider")
    collect_parser.add_argument("--max-results", type=int, help="Max results per topic")

    papers_parser = subparsers.add_parser("collect-papers", help="Search and ingest papers")
    papers_parser.add_argument("--queries", help="Newline-separated paper queries")
    papers_parser.add_argument(
        "--provider",
        choices=["semantic_scholar", "openalex", "arxiv"],
        help="Paper provider",
    )
    papers_parser.add_argument("--max-results", type=int, help="Max results per query")
    papers_parser.add_argument(
        "--include-pdf",
        action="store_true",
        default=None,
        help="Fetch PDF full text",
    )

    brief_parser = subparsers.add_parser("brief", help="Generate a daily brief")
    brief_parser.add_argument("--topics", help="Newline-separated topics")
    brief_parser.add_argument("--push", action="store_true", help="Push the brief after generation")

    ask_parser = subparsers.add_parser("ask", help="Ask the agent a question")
    ask_parser.add_argument("question", help="User question")

    subparsers.add_parser("schedule", help="Start daily scheduler")

    args = parser.parse_args()

    if args.command == "collect":
        topics = _split_lines(args.topics)
        result = collect_and_ingest(topics, args.provider, args.max_results)
        print(result)
    elif args.command == "collect-papers":
        queries = _split_lines(args.queries)
        result = collect_papers_and_ingest(
            queries,
            args.provider,
            args.max_results,
            args.include_pdf,
        )
        print(result)
    elif args.command == "brief":
        topics = _split_lines(args.topics)
        if args.push:
            brief = run_daily_brief(topics, push=True)
        else:
            brief = generate_brief(topics)
        print(brief)
    elif args.command == "ask":
        agent = ReactAgent()
        output = agent.run(args.question)
        print(output)
    elif args.command == "schedule":
        start_scheduler()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
