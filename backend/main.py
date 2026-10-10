from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import OrderedDict
import pymupdf
import os
import re
import math
import time
import uuid
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from fastembed import TextEmbedding

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"

# small ONNX embedding model, 1 thread keeps memory low on the 512MB free tier
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", threads=1)

# ---- limits (change these if your PDFs are bigger) ----
MAX_UPLOAD_MB = 25      # biggest file accepted
MAX_PAGES = 200         # most pages accepted
MAX_DOCS = 10           # documents kept in memory; the oldest is dropped first
EMBED_BATCH_SIZE = 8    # small batches keep memory low

# doc_id -> {"filename": ..., "chunks": [...], "embeddings": np.array}
# lives in memory, so it resets when the server restarts
document_store = OrderedDict()


def clean_math_notation(text):
    """Strip LaTeX syntax, leave plain readable math."""
    text = re.sub(r"\\begin\{cases\}", "", text)
    text = re.sub(r"\\end\{cases\}", "", text)
    text = re.sub(r"\\Rightarrow", "=>", text)
    text = re.sub(r"\\rightarrow", "->", text)
    text = re.sub(r"\\sum", "sum of", text)
    text = re.sub(r"\\times", "x", text)
    text = re.sub(r"\\cdot", "x", text)
    text = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1/\2)", text)
    text = re.sub(r"_\{([^}]*)\}", r"\1", text)   # x_{1} -> x1
    text = re.sub(r"\^\{([^}]*)\}", r"^\1", text) # x^{2} -> x^2
    text = text.replace("\\", "")
    text = text.replace("{", "").replace("}", "")
    return text


class AskRequest(BaseModel):
    doc_id: str
    question: str


@app.get("/")
def read_root():
    return {"message": "PDF Summarizer backend running"}


def extract_text_from_pdf(pdf_bytes):
    """Read the PDF straight from memory (nothing is saved to disk).
    Returns (text, page_count)."""
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        raise HTTPException(status_code=400, detail="This file could not be read as a PDF.")

    try:
        if doc.needs_pass:
            raise HTTPException(
                status_code=400,
                detail="This PDF is password protected. Remove the password and try again.",
            )
        if doc.page_count > MAX_PAGES:
            raise HTTPException(
                status_code=400,
                detail=f"This PDF has {doc.page_count} pages. The limit is {MAX_PAGES}. Try a shorter section.",
            )
        text = "".join(page.get_text() for page in doc)
        return text, doc.page_count
    finally:
        doc.close()


def ask_llm(system_prompt, user_prompt, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return clean_math_notation(response.choices[0].message.content)
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < retries - 1:
                time.sleep(15)  # wait for free-tier limit to refresh
                continue
            raise


def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks so no sentence context is lost at the edges."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def embed_chunks(chunks):
    """Turn each chunk into a semantic vector, a few at a time to keep memory low."""
    embeddings = list(embedder.passage_embed(chunks, batch_size=EMBED_BATCH_SIZE))
    return np.array(embeddings, dtype=np.float32)


def find_relevant_chunks(chunks, chunk_embeddings, question, top_k=6):
    """Embed the question, compare to chunk embeddings via cosine similarity."""
    question_embedding = list(embedder.query_embed([question]))[0]
    scores = np.dot(chunk_embeddings, question_embedding)
    top_indexes = np.argsort(scores)[::-1][:top_k]
    top_indexes = sorted(top_indexes.tolist())  # keep original document order
    return [chunks[i] for i in top_indexes]


def summarize_text(text):
    style_note = (
        "Do NOT use LaTeX syntax (no \\begin{cases}, no underscores for subscripts, no \\{ \\}). "
        "Write all math in plain readable text, e.g. 'y = 40 + 10x1 + 5x2' instead of LaTeX."
    )

    if len(text) <= 6000:
        return ask_llm(
            f"You are a helpful assistant that summarizes documents clearly and concisely. {style_note}",
            f"Summarize the following document:\n\n{text}",
        )

    max_sections = 3
    section_size = max(8000, math.ceil(len(text) / max_sections))
    sections = [text[i:i + section_size] for i in range(0, len(text), section_size)]

    partial_summaries = []
    for i, section in enumerate(sections):
        partial = ask_llm(
            f"You summarize one part of a longer document. Keep all key points. {style_note}",
            f"This is part {i + 1} of {len(sections)}. Summarize it:\n\n{section}",
        )
        partial_summaries.append(partial)
        if i < len(sections) - 1:
            time.sleep(5)

    combined = "\n\n".join(partial_summaries)
    return ask_llm(
        f"You combine partial summaries into one clear, well-structured summary. {style_note}",
        "These are summaries of consecutive parts of ONE document. "
        f"Combine them into a single summary:\n\n{combined}",
    )


def answer_question(chunks, chunk_embeddings, question):
    relevant = find_relevant_chunks(chunks, chunk_embeddings, question)
    context = "\n\n---\n\n".join(relevant)
    answer = ask_llm(
        "You answer questions using only the provided document excerpts. "
        "If the answer is not in the excerpts, say so clearly. "
        "Do NOT use LaTeX syntax (no \\begin{cases}, no underscores for subscripts). "
        "Write all math in plain readable text, e.g. 'y = 40 + 10x1 + 5x2'.",
        f"Document excerpts:\n{context}\n\nQuestion: {question}",
    )
    return answer, len(relevant)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024

    # check the size before loading the file into memory
    size = getattr(file, "size", None)
    if size is not None and size > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"This file is larger than {MAX_UPLOAD_MB} MB. Try a smaller PDF.",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"This file is larger than {MAX_UPLOAD_MB} MB. Try a smaller PDF.",
        )

    text, page_count = extract_text_from_pdf(content)

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text found in PDF (might be scanned/image-only)")

    chunks = chunk_text(text)

    try:
        chunk_embeddings = embed_chunks(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    try:
        summary = summarize_text(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

    # every upload gets its own id, so two people can upload the same filename safely
    doc_id = uuid.uuid4().hex
    document_store[doc_id] = {
        "filename": filename,
        "chunks": chunks,
        "embeddings": chunk_embeddings,
    }
    while len(document_store) > MAX_DOCS:
        document_store.popitem(last=False)  # drop the oldest

    return {
        "doc_id": doc_id,
        "filename": filename,
        "num_pages": page_count,
        "num_chars_extracted": len(text),
        "num_chunks": len(chunks),
        "summary": summary,
    }


@app.post("/ask")
async def ask_question(req: AskRequest):
    doc = document_store.get(req.doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found. Please upload it first.")

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        answer, chunks_used = answer_question(doc["chunks"], doc["embeddings"], req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to answer: {str(e)}")

    return {"question": req.question, "answer": answer, "chunks_used": chunks_used}