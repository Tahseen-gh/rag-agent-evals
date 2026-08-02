# OWASP GenAI/LLM Risks — What we address

- Prompt Injection: treat retrieved text as untrusted, clearly delimited from instructions in the generation prompt (`rag/generation.py`); the agent's only tool is read-only corpus search, so even a successful injection has nothing destructive to reach (least-privilege tool design, see `data/corpus/prompt_injection_attacks.txt`).
- Sensitive Data Exposure: keep logs minimal; redact obvious PII before storing.
- Supply Chain: SBOM via CycloneDX; dependabot recommended.
- Model Behavior: RAGAS evals (optional) to monitor faithfulness/relevance.
- Abuse Handling: small rate-limit & size limit recommended at gateway (not included here).
