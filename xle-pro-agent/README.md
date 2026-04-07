# XLERobot Software-Only Orchestrator (Prototype)

This is a simple, dockerized Python app that uses a **LangGraph agent workflow** with:

- Pluggable LLM backend via config (`vertex` or `ollama`)
- **LangSmith** for execution tracing
- **Streamlit** for a lightweight UI that shows planner/tool/response trace events

It is intentionally **software-only** and makes **no calls to external hardware**.

## What this prototype does

The workflow processes a user directive into a task-oriented result:

1. Planner node asks the LLM which software tool to run.
2. Tool node executes one of three local tools:
   - `build_task_list`
   - `estimate_effort`
   - `rewrite_directive`
3. Responder node returns a final response with structured output.

The app also displays:
- graph topology (`mermaid`)
- local tool/graph trace events in the UI
- full run traces in LangSmith (if configured)

## Project layout

```text
.
├── src/xlerobot_soft_orchestrator/
│   ├── agent.py
│   ├── config.py
│   ├── main.py
│   ├── tools.py
│   └── ui.py
├── tests/
│   ├── test_agent.py
│   └── test_tools.py
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── pyproject.toml
```

## Requirements

- Python `3.11` or `3.12`
- For Vertex: a Google Cloud project with Vertex AI enabled and credentials
- For Ollama (bare metal): local Ollama runtime installed and running
- LangSmith API key (optional but recommended for tracing)

## Environment setup

1. Copy env template:

```bash
cp .env.example .env
```

2. Fill values in `.env`:

- `LLM_PROVIDER` (`vertex` or `ollama`)
- `LLM_MODEL` (optional override for either provider)
- `VERTEX_PROJECT_ID`
- `VERTEX_LOCATION` (default `us-central1`)
- `VERTEX_MODEL` (default `gemini-2.5-flash`)
- `OLLAMA_BASE_URL` (default `http://localhost:11434`)
- `OLLAMA_MODEL` (default `llama3.2`)
- `LANGSMITH_TRACING` (`true` or `false`)
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT`
- `LANGSMITH_ENDPOINT`

3. If using Vertex, set Google credentials in your shell:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
```

Windows PowerShell:

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\absolute\path\to\service-account.json"
```

For Docker Compose in this repo, place the key at `secret/gcp-creds.json`.
The compose file mounts it to `/secret/gcp-creds.json` in the container and sets
`GOOGLE_APPLICATION_CREDENTIALS=/secret/gcp-creds.json`.

4. If using Ollama on bare metal:

```bash
ollama serve
ollama pull llama3.2
```

## Local run

Install dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Run Streamlit UI:

```bash
streamlit run src/xlerobot_soft_orchestrator/ui.py
```

Optional CLI run:

```bash
python -m xlerobot_soft_orchestrator.main "Create a simple morning triage plan"
```

## Docker run

Build and run:

```bash
docker compose up --build
```

Then open:

- [http://localhost:8501](http://localhost:8501)

## TDD notes

Unit tests cover:

- deterministic tool behavior
- planner routing with a fake LLM
- fallback handling for malformed planner output

This keeps tests fast and cloud-independent while production still uses Vertex AI.

## LangSmith tracing

When `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` is set:

- graph runs are sent to LangSmith
- node execution can be inspected in trace view

The Streamlit UI also shows local trace events for quick debugging.
