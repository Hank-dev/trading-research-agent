from pydantic import BaseModel, Field


class StrategyCritique(BaseModel):
    approved: bool
    problems: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    required_changes: list[str] = Field(default_factory=list)
