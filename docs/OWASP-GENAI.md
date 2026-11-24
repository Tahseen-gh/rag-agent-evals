# OWASP GenAI/LLM Risks — What we address

- Prompt Injection: treat retrieved text as untrusted, clearly delimited from instructions in the generation prompt (`rag/generation.py`); the agent's only tool is read-only corpus search, so even a successful injection has nothing destructive to reach (least-privilege tool design, see `data/corpus/prompt_injection_attacks.txt`).
- Sensitive Data Exposure: logs are structured and minimal (request id, method, path, status, duration, client IP, user agent — see `fastapi_app/app/middleware.py`), but there is no PII redaction anywhere in the codebase; this is noted as a threat but not implemented.
- Supply Chain: SBOM via CycloneDX; dependabot recommended.
- Model Behavior: RAGAS evals (optional) to monitor faithfulness/relevance.
- Abuse Handling: implemented, not just recommended — a per-client/path token-bucket rate limiter (`fastapi_app/app/middleware.py::_InMemoryRateLimiter`, wired via `settings.rate_limit_qps` / `RATE_LIMIT_QPS`, default 5 qps, returns 429) and a request body size cap (`fastapi_app/app/main.py::BodySizeLimitMiddleware`, wired via `settings.request_body_max_bytes` / `REQUEST_BODY_MAX_BYTES`, default 100KB, returns 413).
