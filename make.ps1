param([Parameter(Mandatory=$true)][ValidateSet("setup","test","serve","lint","format","typecheck","docker-build","docker-run")]$t)
function py { param($args) & python @args }
switch ($t) {
  "setup"        { py -m pip install -r requirements.txt }
  "test"         { py -m pytest -q }
  "serve"        { uvicorn fastapi_app.app.main:app --reload }
  "lint"         { ruff check . }
  "format"       { ruff format . }
  "typecheck"    { mypy . }
  "docker-build" { docker build -t domain-rag-evals:local . }
  "docker-run"   { docker run --rm -p 8000:8000 domain-rag-evals:local }
}
