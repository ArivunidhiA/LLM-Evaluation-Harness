"""FastAPI application with evaluation endpoints."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from llm_eval.metrics import AVAILABLE_METRICS

app = FastAPI(title="LLM Eval Harness")


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/evaluate/metrics")
async def list_metrics() -> dict[str, list[str]]:
    return {"metrics": AVAILABLE_METRICS}


@app.post("/evaluate/run")
async def start_run() -> dict[str, str]:
    raise HTTPException(status_code=501, detail="Not implemented yet")


@app.get("/evaluate/runs")
async def list_runs() -> dict[str, str]:
    raise HTTPException(status_code=501, detail="Not implemented yet")


@app.get("/evaluate/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, str]:
    raise HTTPException(status_code=501, detail="Not implemented yet")


@app.get("/evaluate/runs/{run_id}/report")
async def get_report(run_id: str) -> dict[str, str]:
    raise HTTPException(status_code=501, detail="Not implemented yet")


@app.post("/evaluate/compare")
async def compare_runs() -> dict[str, str]:
    raise HTTPException(status_code=501, detail="Not implemented yet")
