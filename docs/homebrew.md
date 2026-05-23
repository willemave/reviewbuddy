# ResearchBuddy Homebrew Tap

ResearchBuddy can be published as a third-party Homebrew tap.

## Generate the tap repo

From the source repository:

```bash
researchbuddy tap export
```

By default this writes a sibling repository at:

```text
../homebrew-researchbuddy
```

You can override the target:

```bash
researchbuddy tap export --output-dir /absolute/path/to/homebrew-researchbuddy
```

## What gets generated

- `Formula/researchbuddy.rb`
- `README.md`
- `.github/workflows/validate.yml`
- `skills/researchbuddy-tap-maintainer/SKILL.md`
- `skills/researchbuddy-tap-maintainer/references/publishing.md`

## Publish flow

1. Create and push a tagged release in the source repo, for example `v0.1.7`.
2. Regenerate the dependency constraints from the source lockfile:
   ```bash
   uv export --format requirements-txt --no-hashes --no-dev --no-emit-project -o constraints.txt
   ```
3. Compute the release tarball SHA:
   ```bash
   curl -L https://github.com/<owner>/researchbuddy/archive/refs/tags/v0.1.7.tar.gz | shasum -a 256
   ```
4. Update `Formula/researchbuddy.rb` with the real `sha256`.
5. Validate the tap:
   ```bash
   brew audit --strict --online researchbuddy
   brew install --build-from-source ./Formula/researchbuddy.rb
   brew test researchbuddy
   ```
6. Push the tap repo to `https://github.com/<owner>/homebrew-researchbuddy`.

## User install

```bash
brew tap <owner>/researchbuddy
brew install researchbuddy
```

## Hermes / OpenClaw paste block

Paste this into Hermes or another OpenClaw-style agent:

```text
Install ResearchBuddy with Homebrew:

brew tap willemave/researchbuddy
brew install researchbuddy

Before asking for any new search-provider key, check ~/.openclaw/openclaw.json.
If exa, tavily, or firecrawl is already configured there under `plugins.entries.<provider>.config.webSearch.apiKey` or the legacy `tools.web.search` paths, ask whether ResearchBuddy should reuse that existing provider/key.
If the user says yes, do not collect a duplicate secret. Let ResearchBuddy doctor --fix use the OpenClaw config without copying credentials into ResearchBuddy .env.

researchbuddy doctor --fix
researchbuddy list

Install the bundled OpenClaw skill:
researchbuddy skills install openclaw --scope shared

For a single OpenClaw workspace instead:
researchbuddy skills install openclaw --scope workspace --workspace /path/to/workspace

Do not start research runs until `researchbuddy doctor` passes.
```
