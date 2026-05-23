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
    skill_dir = request.output_dir / "skills" / "researchbuddy-tap-maintainer"
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
    skill_path = "#{opt_pkgshare}/skills/research"
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
    pkgshare.install "constraints.txt"
    pkgshare.install "skills"
    pkgshare.install "docs"

    (bin/"researchbuddy").write <<~SH
      #!/usr/bin/env bash
      set -euo pipefail
      export RESEARCHBUDDY_SKILL_DIR="#{{opt_pkgshare}}/skills/research"
      exec "#{{Formula["uv"].opt_bin}}/uv" tool run --python "3.11" --constraints "#{{opt_pkgshare}}/constraints.txt" --from "{bootstrap_url}" researchbuddy "$@"
    SH
  end

  def caveats
    <<~EOS
      ResearchBuddy bootstraps the tagged CLI package through uv on first run:
        #{{Formula["uv"].opt_bin}}/uv tool run --python 3.11 --constraints "#{{opt_pkgshare}}/constraints.txt" --from "{bootstrap_url}" researchbuddy

      Additional runtime setup:
        - Run `researchbuddy doctor --fix` to create local state and install Playwright browsers into the uv tool runtime
        - Install and authenticate codex: codex login
        - Configure at least one search provider key through environment, ~/.hermes/.env, ~/.openclaw/.env plus ~/.openclaw/openclaw.json, or manual ResearchBuddy .env
        - Optionally set SEARCH_PROVIDER to override auto-selection
        - In OpenClaw, first check ~/.openclaw/openclaw.json and reuse an existing exa/tavily/firecrawl key when the user approves
        - ResearchBuddy auto-loads provider config from ~/.hermes/.env, ~/.openclaw/.env, and ~/.openclaw/openclaw.json without copying credentials
        - Install the OpenClaw skill with: researchbuddy skills install openclaw --scope shared
        - Run `researchbuddy doctor` before first use

      Bundled research skill:
        {skill_path}
    EOS
  end

  test do
    assert_match "tool run --python", (bin/"researchbuddy").read
    assert_match "--constraints", (bin/"researchbuddy").read
    assert_path_exists pkgshare/"constraints.txt"
    assert_path_exists pkgshare/"skills/research/SKILL.md"
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

If you are installing into OpenClaw, check `~/.openclaw/openclaw.json` before asking for search-provider credentials. When it already has `exa`, `tavily`, or `firecrawl` configured under `plugins.entries.<provider>.config.webSearch.apiKey` or the legacy `tools.web.search` paths, ask whether ResearchBuddy should reuse that existing provider/key and let `researchbuddy doctor --fix` auto-load it without copying credentials.

## Update the formula for a new ResearchBuddy release

1. Create and push a new tagged release in the source repo, for example `v{request.version}`.
2. Regenerate `constraints.txt` from the source lockfile:
   ```bash
   uv export --format requirements-txt --no-hashes --no-dev --no-emit-project -o constraints.txt
   ```
3. Download the release tarball and compute its SHA256:
   ```bash
   curl -L {build_source_tarball_url(request)} | shasum -a 256
   ```
4. Update `Formula/{formula_name}.rb` with the new `url` and `sha256`.
5. Validate locally:
   ```bash
   brew audit --strict --online {formula_name}
   brew install --build-from-source ./Formula/{formula_name}.rb
   brew test {formula_name}
   ```
6. Commit and push the tap repo.

## Skill

This tap also ships a maintainer skill at:

```text
skills/researchbuddy-tap-maintainer
```

Use it when an agent needs to bump the formula or validate the tap release.
"""


def render_tap_skill(request: TapExportRequest) -> str:
    """Render the tap maintenance skill."""

    short_tap = build_short_tap_name(request)
    return f"""---
name: researchbuddy-tap-maintainer
description: Use when publishing or updating the ResearchBuddy Homebrew tap. Covers formula version bumps, SHA updates, local brew validation, and push readiness checks.
metadata: {{"openclaw":{{"requires":{{"bins":["brew","curl","shasum"]}}}}}}
---

# ResearchBuddy Homebrew Tap Maintainer

Use this skill when the task is to publish or update the ResearchBuddy Homebrew tap.

## Publishing Workflow

1. Confirm the source repo has a pushed Git tag for the target release.
2. Open `Formula/researchbuddy.rb`.
3. Update `url` and `sha256`.
4. Run:
   ```bash
   brew audit --strict --online researchbuddy
   brew install --build-from-source ./Formula/researchbuddy.rb
   brew test researchbuddy
   ```
5. If validation passes, commit and push the tap repo.

## Install Reference

- Tap: `brew tap {short_tap}`
- Install: `brew install researchbuddy`
- Fully qualified install: `brew install {short_tap}/researchbuddy`

## Read These References As Needed

- For the exact release and validation checklist, read `references/publishing.md`.
"""


def render_skill_publishing_reference(request: TapExportRequest) -> str:
    """Render the maintainer reference document."""

    source_url = build_source_tarball_url(request)
    return f"""# Publishing ResearchBuddy To Homebrew

## Release Inputs

- Source repository: `{build_source_homepage(request)}`
- Release tarball pattern: `{source_url}`
- Formula file: `Formula/researchbuddy.rb`

## Release Checklist

1. Push the source tag, for example `v{request.version}`.
2. Compute the release SHA:
   ```bash
   curl -L {source_url} | shasum -a 256
   ```
3. Regenerate `constraints.txt` from the source `uv.lock` and ensure the formula installs it.
4. Update the formula fields:
   - `url`
   - `sha256`
5. Validate:
   ```bash
   brew audit --strict --online researchbuddy
   brew install --build-from-source ./Formula/researchbuddy.rb
   brew test researchbuddy
   ```
6. Commit and push the tap changes.

## Runtime Notes

- `researchbuddy` still needs `codex` installed and authenticated.
- Playwright browsers are installed into the uv tool runtime by `researchbuddy doctor --fix`.
- The Homebrew wrapper passes `--constraints "$(brew --prefix)/opt/researchbuddy/share/researchbuddy/constraints.txt"` so first-run uv resolution stays aligned with the source lockfile.
- Required environment:
  - one search provider key: `EXA_API_KEY`, `TAVILY_API_KEY`, or `FIRECRAWL_API_KEY`
  - optional override: `SEARCH_PROVIDER`
- Additional auto-detected config sources:
  - `~/.hermes/.env`
  - `~/.openclaw/.env`
  - `~/.openclaw/openclaw.json`
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
        run: brew audit --strict --online ./Formula/researchbuddy.rb
      - name: Install formula
        run: brew install --build-from-source ./Formula/researchbuddy.rb
      - name: Test formula
        run: brew test researchbuddy
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
