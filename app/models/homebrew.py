"""Validated models for Homebrew tap export."""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class TapExportRequest(BaseModel):
    """Input used to render a Homebrew tap repository."""

    output_dir: Path
    github_owner: str = Field(min_length=1)
    source_repo: str = Field(min_length=1)
    tap_repo: str = Field(min_length=1)
    formula_name: str = Field(default="reviewbuddy", min_length=1)
    python_formula: str = Field(default="python@3.13", min_length=1)
    version: str = Field(min_length=1)
    app_description: str = Field(min_length=1)

    @field_validator("github_owner", "source_repo", "tap_repo", "formula_name")
    @classmethod
    def _strip_value(cls, value: str) -> str:
        """Normalize string inputs."""

        return value.strip()


class TapExportResult(BaseModel):
    """Rendered tap output paths."""

    output_dir: Path
    files: list[Path]
