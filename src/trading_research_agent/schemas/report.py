from pydantic import BaseModel


class ResearchReport(BaseModel):
    markdown: str
    verdict: str
    reasons: list[str]
    next_tests: list[str]
    report_path: str | None = None
