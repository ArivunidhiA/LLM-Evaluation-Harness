# LLM Eval Harness

Production-ready evaluation harness for LLM test suites with metrics, regression
analysis, reports, API, and CLI.

## Quick start
- Install deps: `pip install -e .`
- Start API: `llm-eval serve --port 8000`
- Run eval: `llm-eval run --suite summarization --workers 50`

## Structure
- `src/llm_eval`: core package
- `scripts`: utilities (test case generation)
- `test_suites`: generated test cases
- `reports`: HTML/JSON reports
