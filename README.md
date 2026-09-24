# Hasamex Agent

**Live Demo:** [Open the Hasamex Agent](https://hasamex-transcript-agent-fashvz32pxvqtfuho95s3q.streamlit.app/)

An **agentic RAG application** for analyzing expert interviews in the European Robotic Surgery Market.

The system retrieves relevant evidence from expert transcripts and generates answers with **source experts, timestamps, and supporting quotes**.

## Features

* Q&A over expert interview transcripts
* Market-specific retrieval for **France, United Kingdom, and Germany**
* Source and timestamp traceability
* Cross-transcript analysis
* Conversational follow-up questions
* Persistent conversation state

## Tech Stack

* **LangGraph** — agent orchestration
* **OpenAI** — LLM and embeddings
* **ChromaDB** — vector database with metadata filtering
* **Streamlit** — web interface
* **SQLite** — conversation persistence

## Running Locally

### 1. Environment

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key
```

In `app.py`, load the API key using `os` instead of Streamlit secrets:

```python
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

### 2. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

### 3. Run LangGraph Studio

The project can also be run with LangGraph Studio:

```bash
langgraph dev
```

When running through LangGraph Studio, remove the checkpointer from the graph compilation since Studio provides its own persistence.

## Project Structure

```text
hasamex-agent/
├── app.py
├── graph/
│   └── graph.py
├── data/
├── chroma_db/
├── requirements.txt
└── .env
```
