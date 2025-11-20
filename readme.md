## Como usar (local)
1. Criar e ativar venv:
```bash
python -m venv .venv
source .venv/bin/activate # ou .venv\Scripts\activate no Windows
pip install -r requirements.txt
```
2. Rodar API:
```bash
uvicorn discovery.api:app --reload --port 8000
```
3. Usar endpoints:
- `POST /ingest` para submeter texto/paper (json: {"text": "..."})
- `POST /search` para buscar por query (json: {"query": "..."})
- `POST /hypotheses/generate` para gerar hipóteses simples (json: {"context_ids": [...], "domain": "biomed"})


--- FILE: requirements.txt ---
fastapi==0.95.2
uvicorn==0.22.0
pydantic==1.10.11
pdfminer.six==20221105
sentence-transformers==2.2.2
transformers==4.35.2
scikit-learn==1.2.2
faiss-cpu==1.7.4
pytest==7.4.0


# Notes: some packages (faiss) may require platform-specific wheels. If faiss fails, vectorstore falls back to in-memory brute force.


--- FILE: pyproject.toml ---
[tool.poetry]
name = "discovery_ai_project"
version = "0.1.0"
description = "Skeleton for discovery AI project"
authors = ["You <you@example.com>"]


[tool.poetry.dependencies]
python = "^3.9"


--- FILE: scripts/run.sh ---
#!/usr/bin/env bash
set -e
if [ -f .venv/bin/activate ]; then
source .venv/bin/activate
fi
uvicorn discovery.api:app --reload --port 8000


--- FILE: src/discovery/__init__.py ---
"""discovery package init"""


--- FILE: src/discovery/config.py ---
from pydantic import BaseSettings


class Settings(BaseSettings):
EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
HOST: str = "0.0.0.0"
PORT: int = 8000


settings = Settings()