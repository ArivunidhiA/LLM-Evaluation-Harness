"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from llm_eval.api import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns OK."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestMetricsEndpoint:
    """Tests for metrics listing endpoint."""

    def test_list_metrics(self, client):
        """Test listing available metrics."""
        response = client.get("/evaluate/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert "exact_match" in data["metrics"]
        assert "semantic_similarity" in data["metrics"]


class TestRunEndpoints:
    """Tests for evaluation run endpoints."""

    def test_start_run_not_implemented(self, client):
        """Test that start run returns 501 (not implemented yet)."""
        response = client.post("/evaluate/run")
        assert response.status_code == 501

    def test_list_runs_not_implemented(self, client):
        """Test that list runs returns 501."""
        response = client.get("/evaluate/runs")
        assert response.status_code == 501

    def test_get_run_not_implemented(self, client):
        """Test that get run returns 501."""
        response = client.get("/evaluate/runs/test-run-id")
        assert response.status_code == 501

    def test_get_report_not_implemented(self, client):
        """Test that get report returns 501."""
        response = client.get("/evaluate/runs/test-run-id/report")
        assert response.status_code == 501

    def test_compare_runs_not_implemented(self, client):
        """Test that compare runs returns 501."""
        response = client.post("/evaluate/compare")
        assert response.status_code == 501
