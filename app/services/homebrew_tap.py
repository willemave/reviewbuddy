"""Render and export Homebrew tap assets."""

from __future__ import annotations

import subprocess
from pathlib import Path

from app.models.homebrew import TapExportRequest, TapExportResult


def export_tap_repository(request: TapExportRequest) -> TapExportResult:
    """Write a Homebrew tap repository to disk.

    Args:
        request: Validated export configuration.

    Returns:
        Written file paths.
    """

    formula_dir = request.output_dir / "Formula"
    workflow_dir = request.output_dir / ".github" / "workflows"
    skill_dir = request.output_dir / "skills" / "reviewbuddy-tap-maintainer"
    references_dir = skill_dir / "references"

    formula_dir.mkdir(parents=True, exist_ok=True)
    workflow_dir.mkdir(parents=True, exist_ok=True)
    references_dir.mkdir(parents=True, exist_ok=True)

    files = {
        request.output_dir / "README.md": render_tap_readme(request),
        formula_dir / f"{request.formula_name}.rb": render_formula(request),
        workflow_dir / "validate.yml": render_validate_workflow(),
        skill_dir / "SKILL.md": render_tap_skill(request),
        references_dir / "publishing.md": render_skill_publishing_reference(request),
    }

    for path, content in files.items():
        path.write_text(content, encoding="utf-8")

    return TapExportResult(output_dir=request.output_dir, files=list(files))


def detect_github_remote(repo_root: Path) -> tuple[str, str] | None:
    """Parse the GitHub owner/repo from `origin`.

    Args:
        repo_root: Repository root path.

    Returns:
        `(owner, repo)` when a GitHub origin remote exists, otherwise `None`.
    """

    try:
        completed = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            cwd=repo_root,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    remote = completed.stdout.strip()
    if remote.startswith("git@github.com:"):
        slug = remote.removeprefix("git@github.com:")
    elif remote.startswith("https://github.com/"):
        slug = remote.removeprefix("https://github.com/")
    else:
        return None

    if slug.endswith(".git"):
        slug = slug[:-4]
    if "/" not in slug:
        return None

    owner, repo = slug.split("/", 1)
    if not owner or not repo:
        return None
    return owner, repo


def render_formula(request: TapExportRequest) -> str:
    """Render the tap formula."""

    class_name = request.formula_name.replace("-", " ").replace("_", " ").title().replace(" ", "")
    source_url = build_source_tarball_url(request)
    homepage = build_source_homepage(request)
    skill_path = "#{opt_pkgshare}/skills/reviewbuddy-tap-maintainer"
    bootstrap_url = (
        f"git+https://github.com/{request.github_owner}/{request.source_repo}.git@v{request.version}"
    )
    return f'''class {class_name} < Formula
  desc "{request.app_description}"
  homepage "{homepage}"
  url "{source_url}"
  sha256 "REPLACE_WITH_RELEASE_SHA256"

  depends_on "ffmpeg"
  depends_on "uv"

  def install
    (bin/"reviewbuddy").write <<~SH
      #!/usr/bin/env bash
      set -euo pipefail
      exec "#{{Formula["uv"].opt_bin}}/uv" tool run --from "{bootstrap_url}" reviewbuddy "$@"
    SH
    pkgshare.install "skills"
    pkgshare.install "docs"
  end

  def caveats
    <<~EOS
      ReviewBuddy bootstraps the tagged CLI package through uv on first run:
        #{{Formula["uv"].opt_bin}}/uv tool run --from "{bootstrap_url}" reviewbuddy

      Additional runtime setup:
        - Install Playwright browsers after bootstrap if `reviewbuddy doctor` reports they are missing
        - Install and authenticate codex: codex login
        - Set OPENAI_API_KEY and one search provider key (EXA_API_KEY, TAVILY_API_KEY, or FIRECRAWL_API_KEY)
        - Run `reviewbuddy doctor` before first use

      Tap maintenance skill:
        {skill_path}
    EOS
  end

  test do
    assert_match "uv tool run --from", (bin/"reviewbuddy").read
    assert_predicate pkgshare/"skills/reviewbuddy-tap-maintainer/SKILL.md", :exist?
  end
end
'''


def render_tap_readme(request: TapExportRequest) -> str:
    """Render the tap repository README."""

    source_repo_url = build_source_homepage(request)
    short_tap = build_short_tap_name(request)
    formula_name = request.formula_name
    return f"""# {request.tap_repo}

Homebrew tap for [{request.source_repo}]({source_repo_url}).

## Install

```bash
brew tap {short_tap}
brew install {formula_name}
```

If another formula with the same name ever exists, use the fully qualified name:

```bash
brew install {short_tap}/{formula_name}
```

## Update the formula for a new ReviewBuddy release

1. Create and push a new tagged release in the source repo, for example `v{request.version}`.
2. Download the release tarball and compute its SHA256:
   ```bash
   curl -L {build_source_tarball_url(request)} | shasum -a 256
   ```
3. Update `Formula/{formula_name}.rb` with the new `url` and `sha256`.
4. Validate locally:
   ```bash
   brew audit --strict --online {formula_name}
   brew install --build-from-source ./Formula/{formula_name}.rb
   brew test {formula_name}
   ```
5. Commit and push the tap repo.

## Skill

This tap also ships a maintainer skill at:

```text
skills/reviewbuddy-tap-maintainer
```

Use it when an agent needs to bump the formula or validate the tap release.
"""


def render_tap_skill(request: TapExportRequest) -> str:
    """Render the tap maintenance skill."""

    short_tap = build_short_tap_name(request)
    return f"""---
name: reviewbuddy-tap-maintainer
description: Use when publishing or updating the ReviewBuddy Homebrew tap. Covers formula version bumps, SHA updates, local brew validation, and push readiness checks.
metadata: {{"openclaw":{{"requires":{{"bins":["brew","curl","shasum"]}}}}}}
---

# ReviewBuddy Homebrew Tap Maintainer

Use this skill when the task is to publish or update the ReviewBuddy Homebrew tap.

## Publishing Workflow

1. Confirm the source repo has a pushed Git tag for the target release.
2. Open `Formula/reviewbuddy.rb`.
3. Update `url` and `sha256`.
4. Run:
   ```bash
   brew audit --strict --online reviewbuddy
   brew install --build-from-source ./Formula/reviewbuddy.rb
   brew test reviewbuddy
   ```
5. If validation passes, commit and push the tap repo.

## Install Reference

- Tap: `brew tap {short_tap}`
- Install: `brew install reviewbuddy`
- Fully qualified install: `brew install {short_tap}/reviewbuddy`

## Read These References As Needed

- For the exact release and validation checklist, read `references/publishing.md`.
"""


def render_skill_publishing_reference(request: TapExportRequest) -> str:
    """Render the maintainer reference document."""

    source_url = build_source_tarball_url(request)
    return f"""# Publishing ReviewBuddy To Homebrew

## Release Inputs

- Source repository: `{build_source_homepage(request)}`
- Release tarball pattern: `{source_url}`
- Formula file: `Formula/reviewbuddy.rb`

## Release Checklist

1. Push the source tag, for example `v{request.version}`.
2. Compute the release SHA:
   ```bash
   curl -L {source_url} | shasum -a 256
   ```
3. Update the formula fields:
   - `url`
   - `sha256`
4. Validate:
   ```bash
   brew audit --strict --online reviewbuddy
   brew install --build-from-source ./Formula/reviewbuddy.rb
   brew test reviewbuddy
   ```
5. Commit and push the tap changes.

## Runtime Notes

- `reviewbuddy` still needs `codex` installed and authenticated.
- Playwright browsers are installed after brew install with:
  - `$(brew --prefix)/opt/reviewbuddy/libexec/bin/python -m playwright install`
- Required environment:
  - `OPENAI_API_KEY`
  - one search provider key: `EXA_API_KEY`, `TAVILY_API_KEY`, or `FIRECRAWL_API_KEY`
"""


def render_validate_workflow() -> str:
    """Render the tap CI workflow."""

    return """name: validate

on:
  push:
    branches:
      - main
  pull_request:

jobs:
  brew-validate:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Homebrew
        run: brew update
      - name: Audit formula
        run: brew audit --strict --online ./Formula/reviewbuddy.rb
      - name: Install formula
        run: brew install --build-from-source ./Formula/reviewbuddy.rb
      - name: Test formula
        run: brew test reviewbuddy
"""


def build_source_tarball_url(request: TapExportRequest) -> str:
    """Return the tagged GitHub tarball URL."""

    return (
        f"https://github.com/{request.github_owner}/{request.source_repo}/archive/refs/tags/"
        f"v{request.version}.tar.gz"
    )


def build_source_homepage(request: TapExportRequest) -> str:
    """Return the GitHub repository homepage URL."""

    return f"https://github.com/{request.github_owner}/{request.source_repo}"


def build_short_tap_name(request: TapExportRequest) -> str:
    """Return the short `brew tap` name."""

    repo = request.tap_repo
    if repo.startswith("homebrew-"):
        repo = repo.removeprefix("homebrew-")
    return f"{request.github_owner}/{repo}"
