"""CLI entrypoints for the evaluation harness."""

from __future__ import annotations

import typer
import uvicorn

from llm_eval.api import app as api_app


app = typer.Typer(add_completion=False)


@app.command()
def serve(port: int = 8000) -> None:
    """Start the API server."""

    uvicorn.run(api_app, host="0.0.0.0", port=port)


@app.command()
def run(suite: str, baseline: str | None = None, workers: int = 50) -> None:
    """Run an evaluation suite."""

    typer.echo(f"run not implemented: suite={suite}, baseline={baseline}, workers={workers}")


@app.command()
def compare(run_id: str, baseline_id: str, output: str = "report.html") -> None:
    """Compare two runs and output report."""

    typer.echo(f"compare not implemented: {run_id} vs {baseline_id} -> {output}")


@app.command()
def generate(suite: str, count: int, output: str = "test_suites/") -> None:
    """Generate test cases for a suite."""

    typer.echo(f"generate not implemented: suite={suite}, count={count}, output={output}")


@app.command()
def results(run_id: str, format: str = "table") -> None:
    """View results for a run."""

    typer.echo(f"results not implemented: run_id={run_id}, format={format}")
