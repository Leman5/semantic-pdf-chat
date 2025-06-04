# 🧠 LangChain PDF QA System using Streamlit

![license](https://img.shields.io/github/license/Leman5/semantic-pdf-chat)
![python](https://img.shields.io/badge/Python-3.10%2B-blue)
![streamlit](https://img.shields.io/badge/Streamlit-🌟-brightgreen)

A  **Question‑Answering app** that lets you upload a PDF and ask natural‑language questions about its contents. The app semantically chunks the document, embeds those chunks, retrieves only the 3 most relevant ones, compresses them with an LLM, and finally feeds the concise context to GPT‑4 to craft an answer.

### Preview
![App Screenshot](./Screenshot%20from%202025-06-04%2010-18-09.png)

---

## ⚡ Quick Start

```bash
# 1. Clone & install
$ git clone https://github.com/Leman5/semantic-pdf-chat.git
$ cd semantic-pdf-chat
$ python -m venv venv && source venv/bin/activate
$ pip install -r requirements.txt

# 2. Add your keys in .env file

# 3. Run 🏃
$ streamlit run app.py
```

---

## 🔍 Design Decisions & Key Parameters

| Layer                           | Library                                                                                                                                                                                                                                             | Parameters                                       | Why this choice?                                                                                                                                                             |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Chunking**                    | `SemanticChunker` (LangChain Experimental)                                                                                                                                                                                                          | `breakpoint_threshold_type="standard_deviation"` |                                                                                                                                                                              |
| `breakpoint_threshold_amount=3` | Semantic breaks respect topic shifts and sentence boundaries, preserving coherence better than naïve fixed‑length splits. The threshold of **3 σ** filters only strong semantic jumps, giving \~500‑1 000 token chunks—ideal for OpenAI embeddings. |                                                  |                                                                                                                                                                              |
| **Vector Store**                | **Chroma** (persistent)                                                                                                                                                                                                                             | `collection_name="sample"`                       | Chroma offers millisecond‑level similarity search, local persistence, and effortless versioning—perfect for desktop or small‑team deployments.                               |
| **Retrieval**                   | `.as_retriever(k=3)`                                                                                                                                                                                                                                | `k = 3` (top 3 docs)                             | Empirically, 1–3 highly relevant chunks outperform larger sets by avoiding distraction & keeping prompts < 8 K tokens. **3** hits the sweet spot between recall and latency. |
| **Compression**                 | `LLMChainExtractor` (GPT‑3.5‑turbo, T=0.0)                                                                                                                                                                                                          | N/A                                              | Uses a fast, deterministic model to prune fluff and keep only sentences that answer the query—reducing cost and context size by \~60 %.                                      |
| **Answer LLM**                  | `ChatOpenAI` (GPT‑4, T=0.2)                                                                                                                                                                                                                         | `temperature = 0.2`                              | Slight randomness encourages natural language while remaining factual.                                                                                                       |

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **Streamlit** UI
* **LangChain** orchestration

  * `SemanticChunker`
  * `ContextualCompressionRetriever`
* **OpenAI** embeddings & chat models
* **Chroma** vector DB (local)
* **BeautifulSoup4** (optional URL ingestion)
* **python‑dotenv** for secrets

---

## 🗺️ Data Flow

```
📄 PDF → PyPDFLoader
        ↓
   SemanticChunker  ─────▶  Chroma (vector store)
        ↓                      ▲            
        ▼                      │            
ContextualCompressionRetriever │            
        ↓                      │            
  top‑3 compressed chunks  ─────┘            
        ↓                                   
     Prompt Template (context + question)
        ↓
      GPT‑4 → ✨ Answer
```

---

## 🌟 Why is this approach effective?

1. **Coherent Chunks ⇢ Better Embeddings** – Semantic boundaries yield embeddings that truly represent the idea in each passage.
2. **Small k = Fast & Focused** – Retrieving only 3 chunks keeps prompts short, lowers latency & cost, and reduces hallucination by limiting noise.
3. **Compression Before Generation** – Stripping excess sentences lets GPT spend its context window on signal, not filler.
4. **Separation of Concerns** – Different models do what they do best: GPT‑3.5‑turbo for cheap extraction, GPT‑4 for high‑quality answers.

---

## 📑 Environment Variables

| Name                           | Purpose                            |
| ------------------------------ | ---------------------------------- |
| `OPENAI_API_KEY`               | Your OpenAI key                    |

Create a `.env` file:

```env
OPENAI_API_KEY=sk‑...
```

---

