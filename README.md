<![CDATA[<div align="center">

# 🧪 LLM Evaluation Harness

**Production-ready framework for systematic LLM testing, regression detection, and quality assurance**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776ab.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/ArivunidhiA/LLM-Evaluation-Harness/actions/workflows/test.yml/badge.svg)](https://github.com/ArivunidhiA/LLM-Evaluation-Harness/actions)

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [API](#-api-reference) • [CLI](#-cli-usage)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [CLI Usage](#-cli-usage)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Deployment](#-deployment)

---

## 🎯 Overview

LLM Evaluation Harness is a comprehensive testing framework designed to **evaluate, benchmark, and monitor LLM outputs at scale**. It enables teams to catch quality regressions before deployment, measure model performance across diverse test suites, and maintain consistent output quality over time.

**Key Highlights:**
- ⚡ Evaluate **1000+ test cases in under 10 minutes** with async concurrency
- 📊 Generate **professional HTML reports** with Plotly visualizations
- 🔄 **Regression detection** against versioned baselines
- 🛡️ Production-grade reliability with circuit breakers and graceful degradation

---

## ✨ Features

| Category | Features |
|----------|----------|
| **Metrics** | Exact Match, Semantic Similarity (embeddings), LLM-as-Judge, BLEU, ROUGE, Latency, Toxicity, Format Validation |
| **Execution** | 50+ concurrent requests, exponential backoff retries, rate limiting (TPM/RPM), circuit breaker |
| **Observability** | Structured JSON logging, Prometheus metrics, OpenTelemetry tracing, Sentry integration |
| **Deployment** | Docker/Docker Compose, GitHub Actions CI/CD, Kubernetes-ready |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LLM EVAL HARNESS                            │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │   CLI    │    │   API    │    │ Reporter │    │ Metrics  │      │
│  │ (Typer)  │    │(FastAPI) │    │  (HTML)  │    │  Engine  │      │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘      │
│       │               │               │               │             │
│       └───────────────┼───────────────┼───────────────┘             │
│                       ▼                                             │
│              ┌─────────────────┐                                    │
│              │    Evaluator    │◄──── Test Cases (JSON)             │
│              │   (Orchestrator)│                                    │
│              └────────┬────────┘                                    │
│                       │                                             │
│              ┌────────▼────────┐                                    │
│              │  Async Executor │                                    │
│              │ ┌─────────────┐ │                                    │
│              │ │Rate Limiter │ │                                    │
│              │ │Circuit Break│ │                                    │
│              │ │Retry Logic  │ │                                    │
│              │ └─────────────┘ │                                    │
│              └────────┬────────┘                                    │
│                       │                                             │
├───────────────────────┼─────────────────────────────────────────────┤
│  EXTERNAL             ▼                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐         │
│  │ OpenAI   │   │PostgreSQL│   │  Redis   │   │ Sentry   │         │
│  │   API    │   │    DB    │   │  Cache   │   │  Errors  │         │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘         │
└─────────────────────────────────────────────────────────────────────┘
```

| Component | Responsibility |
|-----------|----------------|
| **Evaluator** | Orchestrates test execution, builds prompts, collects results |
| **Executor** | Manages concurrency, retries, rate limits, circuit breaker |
| **Metrics Engine** | Computes all evaluation metrics (10+ built-in) |
| **API** | REST endpoints for runs, results, and reports |
| **CLI** | Command-line interface for local and CI usage |

---

## 🛠 Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Runtime** | Python 3.11+, asyncio |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Database** | PostgreSQL, SQLAlchemy 2.0 (async), Alembic |
| **Cache** | Redis |
| **LLM** | OpenAI API (GPT-4, embeddings) |
| **Observability** | structlog, Prometheus, OpenTelemetry, Sentry |
| **DevOps** | Docker, GitHub Actions, Make |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 14+
- Redis 7+
- OpenAI API key

### Installation

```bash
# Clone repository
git clone https://github.com/ArivunidhiA/LLM-Evaluation-Harness.git
cd LLM-Evaluation-Harness

# Install package
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Start services (Docker)
docker-compose up -d

# Run database migrations
make db-migrate

# Start API server
llm-eval serve --port 8000
```

### Run Your First Evaluation

```bash
# Run a test suite
llm-eval run --suite qa --workers 50

# Compare against baseline
llm-eval compare run123 baseline_v1 --output report.html

# View results
llm-eval results run123 --format table
```

---

## 💻 CLI Usage

```bash
llm-eval <command> [options]

Commands:
  serve     Start the API server
  run       Execute an evaluation suite
  compare   Compare two runs for regressions
  generate  Generate test cases using GPT-4
  results   View run results
```

| Command | Example |
|---------|---------|
| Start server | `llm-eval serve --port 8000` |
| Run evaluation | `llm-eval run --suite summarization --workers 50` |
| Compare runs | `llm-eval compare run123 run456 --output diff.html` |
| Generate tests | `llm-eval generate --suite qa --count 300` |

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/evaluate/metrics` | GET | List available metrics |
| `/evaluate/run` | POST | Start new evaluation |
| `/evaluate/runs` | GET | List all runs |
| `/evaluate/runs/{id}` | GET | Get run details |
| `/evaluate/runs/{id}/report` | GET | Download HTML report |
| `/evaluate/compare` | POST | Compare two runs |

**Example Request:**
```bash
curl -X POST http://localhost:8000/evaluate/run \
  -H "Content-Type: application/json" \
  -d '{"suite": "qa", "model": "gpt-4-turbo", "workers": 50}'
```

---

## ⚙️ Configuration

All settings via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection |
| `MAX_CONCURRENT_REQUESTS` | `50` | Parallel API calls |
| `DEFAULT_MODEL` | `gpt-4-turbo` | Default LLM model |
| `RATE_LIMIT_RPM` | `500` | Requests per minute |

---

## 🐳 Deployment

```bash
# Development
docker-compose up -d

# Production (build and push)
docker build -t llm-eval:latest .
docker push ghcr.io/your-org/llm-eval:latest
```

**CI/CD Workflows:**
- `test.yml` — Run tests on every PR
- `regression.yml` — Nightly regression tests with Slack alerts
- `release.yml` — Build Docker image on tags

---

<div align="center">

**Built by [Arivunidhi A](https://github.com/ArivunidhiA)** • MIT License

</div>
]]>