# ReviewBuddy Homebrew Tap

ReviewBuddy can be published as a third-party Homebrew tap.

## Generate the tap repo

From the source repository:

```bash
reviewbuddy tap export
```

By default this writes a sibling repository at:

```text
../homebrew-reviewbuddy
```

You can override the target:

```bash
reviewbuddy tap export --output-dir /absolute/path/to/homebrew-reviewbuddy
```

## What gets generated

- `Formula/reviewbuddy.rb`
- `README.md`
- `.github/workflows/validate.yml`
- `skills/reviewbuddy-tap-maintainer/SKILL.md`
- `skills/reviewbuddy-tap-maintainer/references/publishing.md`

## Publish flow

1. Create and push a tagged release in the source repo, for example `v0.1.1`.
2. Compute the release tarball SHA:
   ```bash
   curl -L https://github.com/<owner>/reviewbuddy/archive/refs/tags/v0.1.1.tar.gz | shasum -a 256
   ```
3. Update `Formula/reviewbuddy.rb` with the real `sha256`.
4. Validate the tap:
   ```bash
   brew audit --strict --online reviewbuddy
   brew install --build-from-source ./Formula/reviewbuddy.rb
   brew test reviewbuddy
   ```
5. Push the tap repo to `https://github.com/<owner>/homebrew-reviewbuddy`.

## User install

```bash
brew tap <owner>/reviewbuddy
brew install reviewbuddy
```

## Hermes / OpenClaw paste block

Paste this into Hermes or another OpenClaw-style agent:

```text
Install ReviewBuddy with Homebrew:

brew tap willemave/reviewbuddy
brew install reviewbuddy
reviewbuddy commands --agent
reviewbuddy doctor

If you support SKILL.md directories, load this bundled skill:
$(brew --prefix)/opt/reviewbuddy/share/reviewbuddy/skills/reviewbuddy-cli

Read:
- $(brew --prefix)/opt/reviewbuddy/share/reviewbuddy/skills/reviewbuddy-cli/SKILL.md

Do not start research runs until `reviewbuddy doctor` passes.
```
