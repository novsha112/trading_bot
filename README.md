# trading_bot

Modular cryptocurrency trading bot: backtest → paper → testnet → live.
Primary exchange: Bybit. First strategy: Grid (long / short / neutral).

- Project rules: [`CLAUDE.md`](CLAUDE.md)
- Architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- Roadmap: [`docs/ROADMAP.md`](docs/ROADMAP.md)

Current status: **Phase 0 — Project Bootstrap**. No trading functionality yet.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync                  # creates .venv and installs runtime + dev dependencies
cp .env.example .env     # local secrets; .env is git-ignored
```

## Checks

```bash
uv run ruff check .          # lint
uv run ruff format --check . # formatting
uv run mypy                  # type checking (strict)
uv run pytest                # tests
```

All four must pass before a commit.

## Logging

`app.monitoring.logging.configure_logging()` sets up structlog (JSON or console)
for both structlog and stdlib loggers. Secrets are masked by field name
(`api_key`, `secret`, `token`, `signature`, `authorization`, ...) and by value:
pass the actual secret values via `secrets=` so they are masked anywhere,
including exception messages and tracebacks.

## Layout

```text
app/          application code (packages are added phase by phase)
tests/unit/   unit tests, including module dependency rules
docs/         architecture and roadmap
```
