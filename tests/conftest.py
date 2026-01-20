"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_test_case():
    """Provide a sample test case for testing."""
    return {
        "test_id": "test_001",
        "suite": "qa",
        "prompt_template": "Answer the question: {question}",
        "input_variables": {"question": "What is 2+2?"},
        "expected_output": "4",
        "evaluation_metrics": ["exact_match"],
        "thresholds": {"exact_match": 1.0},
        "tags": ["math", "simple"],
        "version": "1.0",
    }
