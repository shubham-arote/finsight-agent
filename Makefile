# finsight — laptop workflow (reference-course pattern: the Makefile wraps only what
# you run locally; every GCP command lives in deploy/gcp/ + docs/deploy.md so you can
# read what it does before you run it).
#
# The flow you probably want:
#   make up        — whole stack in Docker (agent + MCP sidecar + Qdrant), demo profile
#   make verify    — e2e proof against it: ingest → ask → cited, VERIFIED answer
#   make demo      — local dev server on :8010 (no Docker; picks up your .env)
#   make gate      — the offline test suite (runs with zero API keys)
#
# Works in Git Bash / WSL / Cloud Shell. PowerShell users: run the underlying
# commands from README.md directly.

# Auto-load .env so targets pick up keys without `set -a; source .env` first.
ifneq (,$(wildcard .env))
include .env
export
endif

BASE ?= http://localhost:8000

help:
	@echo "Local stack (Docker; demo profile = fast ingest, model-integrated if .env has keys):"
	@echo "  make up        — build + start agent + MCP sidecar + Qdrant"
	@echo "  make verify    — e2e proof: ingest -> ask -> cited, verified answer (BASE=$(BASE))"
	@echo "  make logs      — tail the stack"
	@echo "  make down      — stop containers (volumes survive)"
	@echo "  make nuke      — stop + wipe volumes (fresh Qdrant/collection — fixes schema drift)"
	@echo ""
	@echo "No-Docker demo:"
	@echo "  make demo      — uvicorn on :8010 with your .env (cloud mode if keys present)"
	@echo ""
	@echo "  make vertex    — preflight the Vertex AI path (project, ADC, a real call)"
	@echo ""
	@echo "Quality gates (offline, no keys needed):"
	@echo "  make gate      — full pytest suite (eval floors gated)"
	@echo "  make eval          — agent eval report (evals/reports/)"
	@echo "  make evals-gate    — REGRESSION GATE vs committed floors (evals/baseline.json)"
	@echo "  make evals-ratchet — raise the floors after a genuine improvement"
	@echo "  make bench     — FinRAGBench-V sampled run (needs data/finragbench_v/, see evals/)"
	@echo ""
	@echo "GCP: PROJECT_ID=<id> deploy/gcp/setup.sh && deploy/gcp/deploy.sh   (docs/deploy.md)"

up:
	docker compose up --build -d
	@echo "→ http://localhost:8000   (make verify for the e2e proof)"

verify:
	uv run python scripts/verify_stack.py $(BASE)

logs:
	docker compose logs -f --tail 50

down:
	docker compose stop

nuke:
	docker compose down -v
	@echo "volumes wiped — next 'make up' starts a fresh collection"

demo:
	uv run uvicorn finsight.server:app --host 127.0.0.1 --port 8010

vertex:
	uv run python scripts/check_vertex.py

gate:
	uv run pytest -q

eval:
	uv run python -m evals.run_agent_eval

evals-gate:
	uv run python -X utf8 -m evals.gate

evals-gate-judge:
	uv run python -X utf8 -m evals.gate --judge

evals-ratchet:
	uv run python -X utf8 -m evals.gate --ratchet

bench:
	uv run python -X utf8 -m evals.run_benchmark --data-dir data/finragbench_v --sample 25 --doc-scoped

.PHONY: help up verify logs down nuke demo vertex gate eval evals-gate evals-gate-judge evals-ratchet bench
