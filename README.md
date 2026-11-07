# Document AI Workbench

A Python project with four workflows: schema-validated invoice/resume extraction, citation-grounded document Q&A, a tool-using research agent, and a Streamlit chat interface.

## Setup

Requires Python 3.10+ and an OpenAI API key. From this folder:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. The model defaults to `gpt-4o-mini`; override it with `OPENAI_MODEL`. Embeddings use `text-embedding-3-small` by default and can be changed with `OPENAI_EMBEDDING_MODEL`.

## 1. Extract validated JSON

Save unstructured text as UTF-8, then run:

```powershell
python extractor.py invoice invoice.txt --instructions "Use ISO dates" --output invoice.json
python extractor.py resume resume.txt
```

The output is constrained by a Pydantic schema and parsed into that model before serialization. Missing fields are represented as `null` or empty lists. The source document is treated as data, not prompt instructions.

## 2. RAG document Q&A

Place `.pdf`, `.txt`, or `.md` files in `documents/`, then launch the app and select **Document Q&A**. Click **Index / refresh documents**, then ask questions. PDF pages are indexed individually; answers are instructed to cite source filename and page. The persistent Chroma database is stored in `.chroma/` (override with `CHROMA_PATH`). Scanned PDFs need OCR before indexing.

## 3. Tool-using research agent

```powershell
python agent.py "Compare the populations of these cities and calculate their difference"
```

The agent can search DuckDuckGo, evaluate basic arithmetic expressions, and read text files. Set `AGENT_FILES_DIR` to the directory it may read from. File paths are restricted to that directory. The console shows each tool action and observation; it does not expose hidden model reasoning.

## 4. Streamlit chat UI

```powershell
streamlit run app.py
```

The app offers extraction, document Q&A, and agent modes. It preserves chat history per browser session. Document Q&A streams answer tokens after retrieval; extraction and agent responses appear when each request completes.

## Configuration

See `.env.example`. Never commit `.env` or share API keys. API calls and embedding generation incur provider charges. Chroma stores vectors locally; `CHROMA_PATH` can point to another location.
