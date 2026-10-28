"""Document loading and repeatable, deduplicated ingestion."""
from __future__ import annotations
import hashlib
import re
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

ALLOWED_EXTENSIONS = {'.pdf', '.md', '.txt'}


def load_documents(path: Path) -> list[Document]:
    path = Path(path)
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f'Unsupported extension: {path.suffix}')
    if path.suffix.lower() == '.pdf':
        import fitz
        pages: list[Document] = []
        with fitz.open(str(path)) as pdf:
            for index, page in enumerate(pdf):
                text = page.get_text('text').strip()
                if text:
                    pages.append(Document(page_content=text, metadata={'source': path.name, 'page': index + 1}))
        return pages
    return [Document(page_content=path.read_text(encoding='utf-8'), metadata={'source': path.name, 'page': 1})]


def split_documents(documents: list[Document], size: int, overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=['\n\n', '\n', '. ', ' ', '']
    )
    return splitter.split_documents(documents)


def stable_chunk_id(doc: Document, index: int) -> str:
    raw = f"{doc.metadata['source']}|{doc.metadata.get('page', 1)}|{index}|{doc.page_content}"
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def safe_filename(name: str) -> str:
    """Keep a simple filename; never allow user-provided filesystem paths."""
    base = Path(name.replace('\\', '/')).name
    cleaned = re.sub(r'[^a-zA-Z0-9._-]', '_', base)
    if not cleaned or cleaned in {'.', '..'} or cleaned.startswith('.'):
        raise ValueError('Invalid filename')
    if Path(cleaned).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError('Only PDF, Markdown and TXT files are supported')
    return cleaned
