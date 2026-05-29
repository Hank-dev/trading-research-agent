from pathlib import Path

from trading_research_agent.workflows.research_graph import build_research_graph


DOC_PATH = Path("docs/WORKFLOW.md")
BEGIN = "<!-- BEGIN_BASE_RESEARCH_GRAPH -->"
END = "<!-- END_BASE_RESEARCH_GRAPH -->"


def main() -> int:
    markdown = DOC_PATH.read_text(encoding="utf-8")
    mermaid = build_research_graph().get_graph().draw_mermaid()
    replacement = f"{BEGIN}\n```mermaid\n{mermaid}\n```\n{END}"
    updated = replace_between(markdown, BEGIN, END, replacement)
    DOC_PATH.write_text(updated, encoding="utf-8")
    print(f"Updated {DOC_PATH}")
    return 0


def replace_between(text: str, begin: str, end: str, replacement: str) -> str:
    start = text.index(begin)
    finish = text.index(end, start) + len(end)
    return text[:start] + replacement + text[finish:]


if __name__ == "__main__":
    raise SystemExit(main())
