"""Index local example documents: python ingest.py"""
from pathlib import Path
from safesite.config import Settings
from safesite.engine import RAGEngine
from safesite.ingestion import ALLOWED_EXTENSIONS

if __name__ == '__main__':
    cfg = Settings()
    files = sorted(p for p in cfg.data_dir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS)
    if not files:
        raise SystemExit(f'No PDF, Markdown or TXT files in {cfg.data_dir}')
    print(RAGEngine(cfg).ingest(files))
