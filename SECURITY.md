# SECURITY.md

**Reporting a vulnerability:** Please open a private security advisory on
this repository (GitHub → Security tab → "Report a vulnerability").

## Threat model (LLM/RAG-focused)
- Prompt injection & data exfil → sanitize/strip HTML, never execute user content.
- Poisoned corpus → record provenance; allow block/rollback of specific docs.
- Secrets leakage → `.env` is gitignored; use GitHub Actions secrets; run detect-secrets.
- Supply chain → SBOM uploaded from CI; pin/lock deps when possible.

## Mitigations implemented
- Pre-commit: ruff, mypy (type errors), trailing whitespace/EOF fixers, detect-secrets.
- CI: tests + coverage ≥80%, retrieval evals gated on regression, SBOM artifact.
- Service: health check; optional OTEL traces; no write-side endpoints by default.

## Known pip-audit findings (triaged, not yet fully resolved)
`requirements.txt` pins `fastapi==0.128.0`, which resolves to `starlette==0.50.0`.
`pip-audit` still flags a handful of CVEs on that `starlette` version whose fix
requires the `1.x` line. Fully resolving them means moving three
interdependent pins together: `fastapi` (>=0.140, which pulls in
`starlette>=1.0`), `prometheus-fastapi-instrumentator` (>=8.0, which *requires*
`starlette>=1.0`), and `schemathesis` (the pinned `3.x` series requires
`starlette<1`, so the `api-fuzz.yml` workflow needs schemathesis `4.x`, a
separate major-version migration with CLI/API changes of its own). Doing that
properly needs end-to-end verification of the fuzz workflow against
schemathesis 4.x, not just a pin bump, so it's left as follow-up work rather
than rushed through here. In the meantime this is a known, bounded gap, not a
silent one: `pip-audit` is explicitly non-blocking in CI (see README
"Security") specifically so findings like this can be triaged rather than
hidden by a red build.
