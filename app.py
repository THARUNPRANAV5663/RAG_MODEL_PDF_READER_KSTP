# ===============================
# RAG-based PDF Question Answering App
# ===============================

# ----------- IMPORT LIBRARIES -----------

import streamlit as st
import numpy as np
import faiss
import re

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ----------- PAGE CONFIG -----------

st.set_page_config(page_title="RAG PDF QA", layout="wide")
st.title("📄 RAG-based PDF Question Answering System")


# ----------- LOAD MODELS (Cached for performance) -----------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


embedder = load_embedding_model()
tokenizer, model = load_llm()


# ----------- PDF TEXT EXTRACTION FUNCTION -----------

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    return text


# ----------- TEXT CLEANING FUNCTION -----------

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ----------- CHUNKING FUNCTION -----------

def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# ----------- CREATE FAISS INDEX -----------

def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, embeddings


# ----------- RETRIEVE RELEVANT CHUNKS -----------

def retrieve_chunks(query, index, chunks, k=3):
    query_embedding = embedder.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k)

    retrieved = [chunks[i] for i in indices[0]]
    return retrieved


# ----------- GENERATE ANSWER -----------

def generate_answer(context, question):
    prompt = f"""
You are a strict information extraction system.

Answer the question using ONLY exact facts from the context.
If the exact information is not present in the context, respond exactly with:

Not found in document.

Do not infer.
Do not assume.
Do not use prior knowledge.

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=200)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


# ----------- STREAMLIT UI -----------

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:

    with st.spinner("Processing PDF..."):

        # Step 1: Extract text
        raw_text = extract_text_from_pdf(uploaded_file)

        # Step 2: Clean text
        cleaned_text = clean_text(raw_text)

        # Step 3: Chunk text
        chunks = chunk_text(cleaned_text)

        # Step 4: Create FAISS index
        index, embeddings = create_faiss_index(chunks)

    st.success("PDF processed successfully!")

    # ----------- QUESTION INPUT -----------

    user_query = st.text_input("Ask a question about the PDF")

    if user_query:

        with st.spinner("Generating answer..."):

            # Step 5: Retrieve relevant chunks
            retrieved_chunks = retrieve_chunks(user_query, index, chunks)

            # Combine retrieved context
            context = " ".join(retrieved_chunks)

            # Step 6: Generate answer
            answer = generate_answer(context, user_query)

        st.subheader("Answer:")
        st.write(answer)
