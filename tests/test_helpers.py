import pytest
from langchain_core.documents import Document
from safesite.ingestion import safe_filename, split_documents, stable_chunk_id
from safesite.retrieval import tokenize, rrf_merge, lexical_search


def test_filename_sanitization():
    assert safe_filename('../../good.pdf') == 'good.pdf'
    assert safe_filename('C:\\fake\\report.md') == 'report.md'
    with pytest.raises(ValueError):
        safe_filename('evil.exe')


def test_stable_ids_and_chunking():
    docs = [Document(page_content='A long handbook paragraph. ' * 30, metadata={'source': 'demo.txt', 'page': 1})]
    chunks = split_documents(docs, 150, 20)
    assert len(chunks) > 1
    assert stable_chunk_id(chunks[0], 0) == stable_chunk_id(chunks[0], 0)
    assert stable_chunk_id(chunks[0], 0) != stable_chunk_id(chunks[0], 1)


def test_rrf_deduplicates():
    a = Document(page_content='badge reception', metadata={'source': 'a.md', 'page': 1})
    b = Document(page_content='training office', metadata={'source': 'b.md', 'page': 1})
    merged = rrf_merge([[a,b], [a]], 2)
    assert len(merged) == 2
    assert merged[0].page_content == 'badge reception'


def test_lexical_retrieval():
    docs = [Document(page_content='training coordinator'), Document(page_content='equipment registration')]
    assert lexical_search('equipment registration', docs, 1)[0] == docs[1]


def test_tokenize():
    assert tokenize('Incident: IN-04!') == ['incident', 'in', '04']
