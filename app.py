"""Streamlit interface for extraction, document Q&A, and tool-using research."""
import json
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Document AI Workbench", page_icon="📄", layout="wide")
st.title("Document AI Workbench")
st.caption("Extract structured fields, ask questions over your documents, or research with tools.")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    if api_key: os.environ["OPENAI_API_KEY"] = api_key
    mode = st.radio("Workspace", ["Extract JSON", "Document Q&A", "Research agent"])

if mode == "Extract JSON":
    from extractor import extract
    kind = st.selectbox("Document type", ["invoice", "resume"])
    instructions = st.text_input("Extra extraction guidance", placeholder="For example: return dates in YYYY-MM-DD format")
    text = st.text_area("Paste document text", height=300)
    if st.button("Extract", type="primary", disabled=not text):
        with st.spinner("Extracting and validating fields…"):
            try: st.code(json.dumps(extract(text, kind, instructions), indent=2, ensure_ascii=False), language="json")
            except Exception as exc: st.error(str(exc))
elif mode == "Document Q&A":
    from rag import index_folder, answer_stream
    folder = Path(os.getenv("DOCUMENTS_DIR", "documents"))
    folder.mkdir(parents=True, exist_ok=True)
    st.write(f"Add PDF, TXT, or Markdown files to `{folder.resolve()}`.")
    uploads = st.file_uploader("Add documents", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if uploads:
        for upload in uploads:
            safe_name = Path(upload.name).name
            (folder / safe_name).write_bytes(upload.getvalue())
        st.success(f"Saved {len(uploads)} file(s) in the document folder.")
    if st.button("Index / refresh documents", type="primary"):
        with st.spinner("Reading documents and creating embeddings…"):
            try: st.success(f"Indexed {index_folder(folder, reset=True)} text chunks.")
            except Exception as exc: st.error(str(exc))
    if "rag_messages" not in st.session_state: st.session_state.rag_messages = []
    for item in st.session_state.rag_messages:
        with st.chat_message(item["role"]): st.markdown(item["content"])
    if prompt := st.chat_input("Ask a question about the indexed documents"):
        st.session_state.rag_messages.append({"role":"user", "content":prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            try: reply = st.write_stream(answer_stream(prompt))
            except Exception as exc: reply = f"Could not answer: {exc}"
            if isinstance(reply, str): st.markdown(reply)
        st.session_state.rag_messages.append({"role":"assistant", "content":reply})
else:
    from agent import run_agent
    st.info("The agent can search the web, calculate, and read text files under AGENT_FILES_DIR (default: current folder).")
    if "agent_messages" not in st.session_state: st.session_state.agent_messages = []
    for item in st.session_state.agent_messages:
        with st.chat_message(item["role"]): st.markdown(item["content"])
    if prompt := st.chat_input("Ask a multi-step research question"):
        st.session_state.agent_messages.append({"role":"user", "content":prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            action_area = st.container()
            def show_action(event): action_area.caption(event)
            try: reply=run_agent(prompt, emit=show_action)
            except Exception as exc: reply=f"Could not complete request: {exc}"
            st.markdown(reply)
        st.session_state.agent_messages.append({"role":"assistant", "content":reply})
