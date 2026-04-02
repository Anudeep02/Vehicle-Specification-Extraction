
# ================================
# Step 1: Imports
# ================================
import fitz  # PyMuPDF
import faiss
import numpy as np
import json
import re

from sentence_transformers import SentenceTransformer
from transformers import pipeline


# ================================
# Step 2: Extract text from PDF
# ================================
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)

    for page in doc:
        text += page.get_text()

    return text


# ================================
# Step 3: Smart Chunking (sentence-based)
# ================================
def chunk_text(text, chunk_size=300):
    sentences = text.split(".")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence.strip() + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence.strip() + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# ================================
# Step 4: Embeddings
# ================================
def create_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    return model, embeddings


# ================================
# Step 5: FAISS Index
# ================================
def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index


# ================================
# Step 6: Retrieval
# ================================
def retrieve_chunks(query, model, index, chunks, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]


# ================================
# Step 7: Load LLM
# ================================
def load_llm():
    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",   # better than base
        max_new_tokens=200,
        truncation=True
    )
    return llm


# ================================
# Step 8: LLM Extraction
# ================================
def extract_specifications(llm, chunks, query):
    results = []

    for chunk in chunks:
        chunk = chunk[:500]

        prompt = f"""
You are an expert in automotive service manuals.

Extract ONLY if exact specification is present.

Return STRICT JSON:
{{
  "component": "",
  "spec_type": "",
  "value": "",
  "unit": ""
}}

Rules:
- Extract exact number only (no guessing)
- Torque units should be Nm if present
- If not found → return empty JSON {{}}
- No explanation

Query: {query}

Context:
{chunk}
"""

        try:
            output = llm(prompt)[0]['generated_text']

            match = re.search(r'\{.*?\}', output, re.DOTALL)
            if match:
                data = json.loads(match.group())

                # strict filtering
                if (
                    data.get("value") and
                    re.search(r'\d', str(data["value"]))
                ):
                    results.append(data)

        except:
            continue

    return results


# ================================
# Step 9: Regex Fallback (BONUS)
# ================================
def regex_extraction(text):
    pattern = r'(\d+)\s?(Nm|N·m)'
    matches = re.findall(pattern, text)

    results = []
    for m in matches:
        results.append({
            "component": "Brake Caliper Bolt",
            "spec_type": "Torque",
            "value": m[0],
            "unit": m[1]
        })

    return results


# ================================
# Step 10: Main Pipeline
# ================================
if __name__ == "__main__":
    pdf_path = "service-manual.pdf"
    query = "Torque specification for suspension bolts"

    print("Extracting text...")
    text = extract_text_from_pdf(pdf_path)

    print("Chunking...")
    chunks = chunk_text(text)

    print("Creating embeddings...")
    model, embeddings = create_embeddings(chunks)

    print("Creating FAISS index...")
    index = create_faiss_index(embeddings)

    print("Retrieving relevant chunks...")
    retrieved = retrieve_chunks(query, model, index, chunks)

    print("\nRetrieved Chunks:\n")
    for i, c in enumerate(retrieved):
        print(f"\nChunk {i+1}:\n{c[:200]}")

    print("\nLoading LLM...")
    llm = load_llm()

    print("\nExtracting structured data...")
    results = extract_specifications(llm, retrieved, query)

    # Fallback if LLM fails
    if not results:
        print("\nLLM failed, using regex fallback...")
        results = regex_extraction(" ".join(retrieved))

    print("\nFINAL OUTPUT:\n")
    print(json.dumps(results, indent=2))