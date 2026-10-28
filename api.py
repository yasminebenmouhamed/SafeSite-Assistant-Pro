"""FastAPI service for SafeSite Assistant."""
from functools import lru_cache
from pathlib import Path
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from safesite.config import Settings
from safesite.engine import RAGEngine
from safesite.ingestion import safe_filename

app = FastAPI(title='SafeSite Assistant API', version='0.1.0', description='Source-grounded workplace safety document RAG')

@lru_cache(maxsize=1)
def engine():
    return RAGEngine(Settings())

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=3000)
    source: str | None = None
    mode: str = Field(default='hybrid', pattern='^(similarity|mmr|hybrid)$')
    k: int = Field(default=4, ge=1, le=20)

class Source(BaseModel):
    file: str
    page: int | str
    excerpt: str

class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: int
    retrieval_mode: str

@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'SafeSite Assistant'}

@app.get('/documents')
def documents():
    return {'documents': engine().list_documents()}

@app.post('/ask', response_model=AskResponse)
def ask(payload: AskRequest):
    try:
        return engine().ask(payload.question, source=payload.source, mode=payload.mode, k=payload.k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Don't leak keys, stack traces or provider credentials through API.
        raise HTTPException(status_code=503, detail='LLM or retrieval service unavailable. See server logs.') from exc

@app.post('/ingest')
def ingest(files: list[UploadFile] = File(...)):
    paths = []
    root = Path('storage/uploads')
    root.mkdir(parents=True, exist_ok=True)
    try:
        for upload in files:
            try:
                filename = safe_filename(upload.filename or '')
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            dest = root / filename
            # Bound each upload to 15 MB.
            content = upload.file.read(15 * 1024 * 1024 + 1)
            if len(content) > 15 * 1024 * 1024:
                raise HTTPException(status_code=413, detail='File too large (15 MB max)')
            dest.write_bytes(content)
            paths.append(dest)
        return engine().ingest(paths)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail='Document ingestion failed; see server logs.') from exc
