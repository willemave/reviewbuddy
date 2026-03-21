## Application Overview

Writing Buddy is an AI-powered writing assistant that transforms notes/outlines into polished prose. It uses a pipeline of specialized pydantic-ai agents for expansion, critique, and rewriting.

**Key Components:**
- **CLI** (`app/cli.py`): Typer-based run and follow-up commands
- **API** (`app/main.py`): FastAPI REST endpoints for React frontend
- **Agents** (`app/agents/`): Expander, Critics (3), Rewriters (2), Researcher
- **Research** (`app/research/`): MCP tools + deep research providers (OpenAI/Google)

**Workflows:**
| Type | Flow |
|------|------|
| Basic | outline → expand → critique → rewrite |
| Research | outline → research → expand (with citations) → critique → rewrite |
| Quick | paste text → critique → rewrite |

See [docs/architecture.md](docs/architecture.md) for detailed architecture documentation.

---

## 1. Python / FastAPI Coding Rules

* **Functions over classes**.
* **Full type hints**; validate with **Pydantic v2** models. Use `typing` for complex types.
* **RORO** pattern (receive object, return object).
* `lower_snake_case` for files/dirs; verbs in variables (`is_valid`, `has_permission`).
* Guard-clause error handling; early returns over nested `else`.
* **Docstrings**: Use Google-style for all public functions/classes.
* **Constants**: Define in `app/constants.py` or module-level UPPER_CASE.

---

## 2. FastAPI Best Practices

* Use **lifespan** context, not `startup/shutdown` events.
* Inject DB/session with dependencies; use `Annotated` for cleaner signatures.
* Middleware order matters: logging → tracing → CORS → error capture.

---

## 3. Code Quality & Safety

* **No hardcoded secrets**; use `pydantic-settings` for config management.
* **Input validation**: Always validate at boundaries (API, external services).
* **SQL injection prevention**: Use parameterized queries, never f-strings.
* **Graceful degradation**: Circuit breakers for external services.
* **Error context**: Include request IDs, user context in error logs.

---

## 4. Testing Requirements

* **Write tests for all new functionality** in `app/tests/` using idiomatic pytest.
* Test structure mirrors app structure: `tests/routers/`, `tests/services/`, etc.
* Test file naming: `test_<module_name>.py`.
* **Test categories**:
  - Unit tests: isolated function/class testing
  - Integration tests: API endpoints with test DB
  - Contract tests: external service interactions
* Use pytest fixtures for setup/teardown.
* **TestClient** from FastAPI for endpoint testing.
* Mock external dependencies with `pytest-mock` or `unittest.mock`.
* **Run tests**: `pytest app/tests/ -v`
* **Test data**: Use factories or fixtures, never production data.

---

## 5. Development Workflow

* **Pre-commit hooks**: `ruff` for linting/formatting
* **Environment management**: `.env.example` template; never commit `.env`. Use `app/core/settings.py` and Pydantic for settings.
* **Database migrations**: Alembic with descriptive revision messages.
* **Error responses**: Consistent format with error codes, messages, details.
* **Versioned releases**: If the user asks to "commit and push" changes that should ship through Homebrew/tap, also bump the app version in `pyproject.toml` and `app/constants.py`, create/push a new git tag, and update the tap formula to point at the new tagged release and SHA.

---

## 6. Package & Dev Tools

### Package Management (uv)
```bash
uv sync                    # Install all dependencies
uv add <package>           # Add dependency
uv add --dev <package>     # Add dev dependency
source .venv/bin/activate  # Activate venv
```

### Database
```bash
alembic upgrade head       # Apply migrations
alembic revision -m "..."  # Create migration
```

### Code Quality
```bash
ruff check .               # Lint
ruff format .              # Format
pytest app/tests/ -v       # Run tests
```

### Running the App
```bash
# Setup
uv sync && . .venv/bin/activate
cp .env.example .env                 # Add your API keys
alembic upgrade head                 # Apply migrations

# CLI
python -m app.cli run "your prompt"

# API server (for React frontend)
uvicorn app.main:app --reload        # Port 8000

# Frontend (separate terminal)
cd frontend && npm install && npm run dev  # Port 5173
```

---

## 7. Preferred Dev Tools

* **LLM internet search**: Use the EXA MCP `web_search_exa` tool for any web/internet lookups (and `get_code_context_exa` for external API/library docs).
* **LLM code search**: Use the Morph MCP `warp_grep` tool for repository code searches before opening files.

| Tool | Purpose | Example |
|------|---------|---------|
| **fd** | Fast file finder | `fd -e py foo` |
| **rg** | Fast code search | `rg "TODO"` |
| **ast-grep (sg)** | AST-aware search | `sg -p 'if ($A) { $B }'` |
| **jq** | JSON processor | `cat data.json \| jq '.items'` |
| **fzf** | Fuzzy finder | `history \| fzf` |
| **bat** | Better cat | `bat file.py` |
| **eza** | Modern ls | `eza -l --git` |
| **httpie** | HTTP client | `http GET api/foo` |
| **delta** | Better git diff | `git diff` (with config) |


**Keep all replies short, technical, and complete.**

**Always run `ruff check` on touched Python files (or the repo) after a set of changes, and fix violations before final handoff whenever possible.**
