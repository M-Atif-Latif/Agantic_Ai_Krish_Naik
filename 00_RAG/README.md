 # Retrieval-Augmented Generation (RAG) & Conversational Memory Lab

This folder is a practical, end‑to‑end playground of modern RAG pipelines, multi‑document chat, real‑time vector workflows, and LangChain memory strategies. Each notebook focuses on a specific concept (vector stores, AstraDB, FAISS, Pinecone, MongoDB integration, Gemini / OpenAI models, conversational memory variants) with clean, documented, and API‑key‑sanitized examples.

> All secrets/API keys have been removed. Replace placeholders like `WRITE_YOUR_OWN_KEY` or environment variable lookups with your own credentials locally.

## 📚 Table of Contents
1. Overview
2. Tech Stack
3. Notebook Matrix (What to run & Why)
4. Memory Strategies Explained
5. Vector Store & RAG Pipelines
6. Setup & Environment
7. Secrets Management Best Practices
8. Extending the Project
9. Troubleshooting
10. Author & Contact

---
## 1. Overview
The goal is to contrast multiple approaches for grounding LLM responses in external knowledge while preserving conversational continuity. You can:
- Chat with multiple heterogeneous documents (PDF, PPTX, TXT, DOC-like extracts).
- Use different embedding backends & vector stores (AstraDB, FAISS, Pinecone, in‑memory, Chroma).
- Explore conversational memory options (buffer, window, entity) and when to apply each.
- Integrate MongoDB + Pinecone for real‑time insertion & retrieval flows.
- Experiment with Gemini (Google Generative AI) and OpenAI style chat models under a unified LangChain interface.

---
## 2. Tech Stack
Core: Python, LangChain, SentenceTransformers / OpenAI / Gemini embeddings, Google Generative AI, Hugging Face models.
Vector & Storage: AstraDB, Pinecone, FAISS, (optional) Chroma, MongoDB.
Utilities: Unstructured, PyPDFLoader, pptx parsing, environment variable handling.
Orchestration: LangChain ConversationChain, memory classes, custom prompt templates.

---
## 3. Notebook Matrix
| Notebook | Focus Area | Key Concepts |
|----------|------------|--------------|
| `Chat_With_Multiple_Doc(pdfs,_docs,_txt,_pptx)_using_AstraDB_and_Langchain.ipynb` | Multi‑document RAG with AstraDB | Loading mixed formats, chunking, embeddings, AstraDB vector store, contextual QA |
| `RAG_Application_using_Langchain_OpenAI_API_and_FAISS.ipynb` | Lightweight FAISS RAG | Local vector index, OpenAI (or drop‑in alt) embeddings, retrieval chain |
| `RAG_with_Conversation.ipynb` | Conversational RAG | Memory + retrieval fusion |
| `Mongodb_with_Pinecone_Realtime_RAG_Pipeline_yt.ipynb` & `Part2` | Streaming / realtime doc ingest | MongoDB integration, Pinecone upserts, retrieval freshness |
| `Mongodb_with_Pinecone_Realtime_RAG_Pipeline_yt (1).ipynb` | Variant / refinement | Parameter tweaks, stability checks |
| `Conversational_Summary_Memory.ipynb` | Summarization memory | Rolling compression strategy |
| `ConversationEntityMemory.ipynb` | Entity‑centric memory | Entity extraction, entity store inspection |
| `chatbot_using_langchain_with_memory.ipynb` | Basic chatbot + memory | Buffer memory baseline |
| `Langchain_memory_classes (1).ipynb` | Memory patterns deep dive | Buffer vs Window vs Entity comparison |
| `LCEL(Langchain_Expression_Language) (1).ipynb` | LCEL patterns | Chain composition / declarative graphs |
| `RAG_with_Conversation.ipynb` | Retrieval + chat continuity | Combining retrieved context w/ conversation history |
| `RAG App using Langchain Mistral in-memory/` | In‑memory embedding demo | Fast prototyping, no external vector DB |

If duplicates exist (e.g., suffixed `(1)` versions), they represent iterative refinements or pedagogical variants.

---
## 4. Memory Strategies (Quick Guide)
| Memory Type | Strength | Trade‑Off | Use When |
|-------------|----------|-----------|----------|
| Buffer | Full fidelity | Token bloat | Short sessions / debugging |
| Buffer Window | Bounded prompt | Loses long‑range facts | Cost‑sensitive multi‑turn chat |
| Summary (Summarization) | Compressed history | Possible detail loss | Long sessions, need gist retention |
| Entity Memory | Semantic entity recall | Extra LLM calls | Assistants tracking people/objects |
| Hybrid (Window + Entity) | Balanced context | Complexity | Mixed recency + factual persistence |

---
## 5. Vector Store & RAG Pipelines
| Store | Characteristics | Notes |
|-------|-----------------|-------|
| AstraDB | Serverless Cassandra + vector | Good for multi‑modal scale |
| Pinecone | Managed high-perf vectors | Real‑time upsert friendly |
| FAISS | Local, fast, in‑memory/disk | Great for prototypes / offline |
| Chroma (optional) | Developer‑friendly local DB | Rapid iteration |
| In‑Memory (dict/list) | Zero dependency | Not persistent, demo only |

Retrieval Pattern (canonical):
1. Ingest -> chunk (size & overlap tuned to semantic coherence).
2. Embed -> store (batch to reduce latency).
3. Query -> embed question.
4. Retrieve top‑k.
5. Compose prompt (question + retrieved context + memory state).
6. LLM generate.
7. (Optional) Memory update / logging / monitoring.

---
## 6. Setup & Environment
### A. Clone & Environment
```bash
git clone <your-fork-or-repo-url>
cd Agantic_Ai_Krish_Naik/RAG
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
```

### B. Core Installation (pick what you need)
```bash
pip install langchain langchain-community langchain-openai langchain-google-genai
pip install sentence-transformers unstructured pypdf python-pptx
pip install faiss-cpu pinecone-client pymongo cassio
```
Optional / experimentation:
```bash
pip install chromadb tiktoken python-dotenv
```

### C. Environment Variables (examples)
| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI / compatible endpoint key |
| `GOOGLE_API_KEY` | Gemini (Generative AI) access |
| `ASTRA_DB_APPLICATION_TOKEN` | AstraDB auth |
| `ASTRA_DB_ID` / `ASTRA_DB_REGION` | Astra DB routing |
| `PINECONE_API_KEY` | Pinecone vector index |
| `MONGODB_URI` | Mongo ingestion pipeline |

Provide them securely (shell export, `.env` not committed, secret managers). Never hardcode.

---
## 7. Secrets Management Best Practices
- Use `.env` + `python-dotenv` locally; environment variables in production.
- Rotate keys periodically; restrict scopes.
- Mask notebooks before sharing (already sanitized here).
- For Colab: `from google.colab import userdata` or `getpass` pattern.
- Consider secret managers (Vault, AWS SM, GCP Secret Manager) for deployment.

---
## 8. Extending the Project
| Idea | Description |
|------|-------------|
| Add Retrieval Evaluator | Use synthetic Q/A pairs + accuracy metrics to tune chunking & embeddings. |
| Instrument Token Usage | Log prompt/completion tokens per call for cost dashboards. |
| Hybrid Search | Combine dense + sparse (BM25) scoring. |
| Add Re-Ranking | Use cross-encoder to refine top‑k context. |
| Long-Term Memory Store | Persist entity summaries to Redis or Postgres for continuity. |
| Guardrails | Add moderation / PII redaction before storage. |
| Streaming UI | Build a Gradio / Streamlit conversational front-end. |

---
## 9. Troubleshooting
| Symptom | Likely Cause | Quick Fix |
|---------|--------------|-----------|
| Empty retrieval results | Wrong embeddings / index mismatch | Re-embed with consistent model name |
| High latency | Large k / slow embedding model | Reduce k, batch embeddings |
| Memory not updating | Wrong memory key mapping | Inspect `chain.prompt.template` & memory variables |
| Entity facts missing | Model not extracting | Switch to higher quality LLM / adjust prompt |
| OOM / token errors | Oversized history | Apply window or summarization memory |

---
## 10. About the Author

<div style="background-color: #f8f9fa; border-left: 5px solid #28a745; padding: 20px; margin-bottom: 20px; border-radius: 5px;">
  <h2 style="color: #28a745; margin-top: 0; font-family: 'Poppins', sans-serif;">Muhammad Atif Latif</h2>
  <p style="font-size: 16px; color: #495057;">Data Scientist & Machine Learning Engineer</p>
  
  <p style="font-size: 15px; color: #6c757d; margin-top: 15px;">
    Passionate about building AI solutions that solve real-world problems. Specialized in machine learning, 
    deep learning, and data analytics with experience implementing production-ready models.
  </p>
</div>

## Connect With Me

<div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 15px;">
  <a href="https://github.com/m-Atif-Latif" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-Follow-212121?style=for-the-badge&logo=github" alt="GitHub">
  </a>
  <a href="https://www.kaggle.com/matiflatif" target="_blank">
    <img src="https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=for-the-badge&logo=kaggle" alt="Kaggle">
  </a>
  <a href="https://www.linkedin.com/in/muhammad-atif-latif-13a171318" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin" alt="LinkedIn">
  </a>
  <a href="https://x.com/mianatif5867" target="_blank">
    <img src="https://img.shields.io/badge/Twitter-Follow-1DA1F2?style=for-the-badge&logo=twitter" alt="Twitter">
  </a>
  <a href="https://www.instagram.com/its_atif_ai/" target="_blank">
    <img src="https://img.shields.io/badge/Instagram-Follow-E4405F?style=for-the-badge&logo=instagram" alt="Instagram">
  </a>
  <a href="mailto:muhammadatiflatif67@gmail.com">
    <img src="https://img.shields.io/badge/Email-Contact-D14836?style=for-the-badge&logo=gmail" alt="Email">
  </a>
</div>

If this helped you, consider starring the repo and sharing improvements via PR.

> Future enhancement: Add a LICENSE file (MIT/Apache 2.0 recommended) and `requirements.txt` export for reproducibility.

---
