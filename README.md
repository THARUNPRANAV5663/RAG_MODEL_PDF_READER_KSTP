# 📄 RAG-Based PDF Question Answering System

An intelligent document Q&A system built using Retrieval-Augmented Generation (RAG). Upload any PDF, ask questions in natural language, and get precise, document-grounded answers — with zero hallucination.

🔗 **Live Demo:** https://ragmodelpdfreaderkstp.streamlit.app/

---

## 📌 Problem Statement

Large documents are hard to navigate manually. This system lets users ask natural language questions from any PDF and get accurate answers strictly based on the document content — no guessing, no hallucination.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Embeddings:** `all-MiniLM-L6-v2` (Sentence Transformers)
- **Vector Search:** FAISS (`IndexFlatL2`)
- **LLM:** `google/flan-t5-small` (Hugging Face Transformers)
- **PDF Parsing:** `pypdf`
- **Deployment:** Streamlit

---

## 🧠 RAG Pipeline

```
PDF Upload
  → Text Extraction (pypdf)
  → Text Cleaning (regex whitespace normalization)
  → Chunking (400 chars, 50 char overlap)
  → Embedding (all-MiniLM-L6-v2)
  → FAISS IndexFlatL2 (vector store)
  → Top-3 Semantic Retrieval (L2 distance)
  → Answer Generation (flan-t5-small, strictly on retrieved context)
```

---

## ⚙️ Key Implementation Details

- **Chunking:** Character-level, chunk size = 400, overlap = 50 (preserves context across boundaries)
- **Embeddings:** `all-MiniLM-L6-v2` — lightweight, fast, high quality
- **FAISS:** `IndexFlatL2` — exact L2 similarity search
- **Retrieval:** Top-k=3 most relevant chunks combined as context
- **Generation:** `flan-t5-small` with strict prompt — answers only from retrieved context, responds "Not found in document." if answer is absent
- **Caching:** `@st.cache_resource` used for both embedding model and LLM to avoid reloading on every interaction

---

## ✨ App Features

- Upload any PDF file
- Ask unlimited questions from the document
- Strict anti-hallucination prompt — model will not infer or assume
- Fast responses via cached models

---

## 📁 Project Structure

```
RAG_MODEL_PDF_READER_KSTP/
├── app.py            # Full RAG pipeline + Streamlit UI
└── requirements.txt
```

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/THARUNPRANAV5663/RAG_MODEL_PDF_READER_KSTP.git
cd RAG_MODEL_PDF_READER_KSTP
pip install -r requirements.txt
streamlit run app.py
```

---

## 👤 Author

**Tharun Pranav K S**  
[LinkedIn](https://www.linkedin.com/in/tharunpranav-k-s-8a0608250/) | [GitHub](https://github.com/THARUNPRANAV5663)
