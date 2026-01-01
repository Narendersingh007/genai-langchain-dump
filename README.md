# GenAI & LangChain Experiments

This repository contains hands-on experiments and practice code for building **GenAI applications using LangChain**.  
It focuses on understanding core concepts like retrievers, vector stores, text splitters, chains, and runnables.

##  Project Structure
- `DocLoaders/` – Loading and preprocessing documents  
- `text_splitters/` – Chunking strategies for documents  
- `retrievers/` – Different retrieval techniques (BM25, vector, hybrid)  
- `vector_store/` – Vector database experiments (FAISS / Chroma / etc.)  
- `chains/` – LangChain chains and pipelines  
- `runnables/` – Runnable-based workflows  
- `output_parsers/` – Structured output handling  
- `models/` – LLM interaction examples  

## Setup
```bash
python -m venv vir311
source vir311/bin/activate
pip install -r requirements.txt
```
## 🧪 What I’m Exploring
- Different retrieval strategies and when to use them
- Trade-offs between sparse, dense, and hybrid retrieval
- Chunking strategies and their impact on retrieval quality
- Modular LangChain components using runnables
- Basic RAG pipelines and experimentation workflows

## 📌 Current Status
This repository is actively evolving as I experiment with new ideas, patterns, and tools in the GenAI ecosystem.  
Code quality and structure are prioritized over completeness.

## 🛠️ Tech Stack
- Python
- LangChain
- Vector stores (FAISS / Chroma)
- Jupyter Notebooks
- Hugging Face / API-based LLMs

## 🔒 Notes
- Environment variables and credentials are stored in `.env` (ignored by Git)
- Heavy artifacts like embeddings and caches are not committed

## 📈 Future Improvements
- Add end-to-end RAG demo
- Introduce reranking and compression steps
- Benchmark different retrievers
- Improve documentation with examples

---

⭐ If you find this repository useful, feel free to explore or fork it.
