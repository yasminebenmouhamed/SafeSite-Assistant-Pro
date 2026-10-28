"""Streamlit client for the FastAPI backend. Launch API before UI."""
import os
import requests
import streamlit as st

API = os.getenv('SAFESITE_API_URL', 'http://localhost:8000')
st.set_page_config(page_title='SafeSite Assistant', page_icon='🦺', layout='wide')
st.title('🦺 SafeSite Assistant')
st.caption('Document-grounded workplace safety Q&A • Local Ollama or OpenRouter • Source citations')

with st.sidebar:
    st.header('Documents')
    uploaded = st.file_uploader('Upload PDF, Markdown or TXT', type=['pdf', 'md', 'txt'], accept_multiple_files=True)
    if st.button('Index selected files', disabled=not uploaded):
        with st.spinner('Indexing documents...'):
            try:
                response = requests.post(API + '/ingest', files=[('files', (f.name, f.getvalue())) for f in uploaded], timeout=180)
                response.raise_for_status()
                st.success(str(response.json()))
            except requests.RequestException as exc:
                st.error(f'Upload failed: {exc}')
    try:
        response = requests.get(API + '/documents', timeout=15)
        response.raise_for_status()
        names = response.json()['documents']
    except requests.RequestException:
        names = []
        st.warning('Start the FastAPI server first (port 8000).')
    selected = st.selectbox('Search documents', ['All documents'] + names)
    mode = st.selectbox('Retriever', ['hybrid', 'similarity', 'mmr'])
    k = st.slider('Retrieved chunks (k)', min_value=1, max_value=10, value=4)
    if st.button('Clear conversation'):
        st.session_state.messages = []
        st.rerun()

if 'messages' not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if message.get('sources'):
            with st.expander('Evidence and citations'):
                for s in message['sources']:
                    st.markdown(f"**{s['file']} · p.{s['page']}**")
                    st.caption(s['excerpt'])

question = st.chat_input('Ask a question about the indexed safety documents...')
if question:
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)
    with st.chat_message('assistant'):
        with st.spinner('Retrieving evidence and generating answer...'):
            try:
                response = requests.post(API + '/ask', json={
                    'question': question, 'source': None if selected == 'All documents' else selected,
                    'mode': mode, 'k': k
                }, timeout=180)
                response.raise_for_status()
                result = response.json()
                st.markdown(result['answer'])
                st.caption(f"{result['latency_ms']} ms • {result['retrieval_mode']} retrieval")
                with st.expander('Evidence and citations'):
                    for s in result['sources']:
                        st.markdown(f"**{s['file']} · p.{s['page']}**")
                        st.caption(s['excerpt'])
                st.session_state.messages.append({'role': 'assistant', 'content': result['answer'], 'sources': result['sources']})
            except requests.RequestException as exc:
                st.error(f'API request failed: {exc}')
