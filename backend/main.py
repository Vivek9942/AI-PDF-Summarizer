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

# ---- summary settings ----
FULL_SUMMARY_MAX_CHARS = 24000   # up to this size: read every part. Bigger: topic overview.
OVERVIEW_EXCERPTS = 12           # pieces shown to the AI for a long document (one per topic)

# doc_id -> {"filename": ..., "chunks": [...], "embeddings": np.array}
# lives in memory, so it resets when the server restarts
document_store = OrderedDict()

STYLE_NOTE = (
    "Do NOT use LaTeX syntax (no \\begin{cases}, no underscores for subscripts, no \\{ \\}). "
    "Write all math in plain readable text, e.g. 'y = 40 + 10x1 + 5x2' instead of LaTeX."
)


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


def ask_llm(system_prompt, user_prompt, max_tokens=2000, retries=4):
    """One AI call. max_tokens is set on purpose: it stops the answer being cut
    off by a hidden default, and keeps each call small for the free-tier limits."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=max_tokens,
                reasoning_effort="low",   # summaries do not need long hidden reasoning
            )
            choice = response.choices[0]
            text = clean_math_notation(choice.message.content or "")

            if not text.strip():
                # the model used its whole budget thinking: retry with more room
                if attempt < retries - 1:
                    max_tokens = int(max_tokens * 1.5)
                    continue
                raise RuntimeError("The AI returned an empty answer. Please try again.")

            if choice.finish_reason == "length":
                text += "\n\n*(This answer reached the length limit and was cut short.)*"
            return text
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < retries - 1:
                time.sleep(15)  # wait for the free-tier limit to refresh
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


def pick_representative_chunks(embeddings, k, restarts=3, iters=15):
    """Group the chunks by meaning (k-means) and return the chunk closest to the
    centre of each group, in document order. One chunk per topic means every
    topic in the document gets a place, however long the document is."""
    n = len(embeddings)
    if n <= k:
        return list(range(n))

    emb = embeddings.astype(np.float64)
    best_score, best_centers = -1.0, None

    for seed in range(restarts):
        rng = np.random.default_rng(seed)
        # k-means++ start: choose spread-out first centres
        centers = [emb[rng.integers(n)]]
        for _ in range(k - 1):
            dist = np.clip(1.0 - np.max(emb @ np.array(centers).T, axis=1), 0, None)
            total = dist.sum()
            probs = dist / total if total > 0 else np.full(n, 1.0 / n)
            centers.append(emb[rng.choice(n, p=probs)])
        centers = np.array(centers)

        for _ in range(iters):
            labels = np.argmax(emb @ centers.T, axis=1)
            for j in range(k):
                members = emb[labels == j]
                if len(members):
                    c = members.mean(axis=0)
                    norm = np.linalg.norm(c)
                    if norm > 0:
                        centers[j] = c / norm

        score = float(np.max(emb @ centers.T, axis=1).sum())  # how well the centres fit
        if score > best_score:
            best_score, best_centers = score, centers.copy()

    sims = emb @ best_centers.T
    labels = np.argmax(sims, axis=1)
    chosen = set()
    for j in range(k):
        members = np.where(labels == j)[0]
        if len(members):
            chosen.add(int(members[np.argmax(sims[members, j])]))
    return sorted(chosen)


def summarize_in_parts(text):
    """Short or medium document: read every part (a few AI calls)."""
    if len(text) <= 6000:
        return ask_llm(
            f"You are a helpful assistant that summarizes documents clearly and concisely. {STYLE_NOTE}",
            f"Summarize the following document:\n\n{text}",
            max_tokens=2500,
        )

    max_sections = 3
    section_size = max(8000, math.ceil(len(text) / max_sections))
    sections = [text[i:i + section_size] for i in range(0, len(text), section_size)]

    partial_summaries = []
    for i, section in enumerate(sections):
        partial = ask_llm(
            f"You summarize one part of a longer document. Keep all key points. {STYLE_NOTE}",
            f"This is part {i + 1} of {len(sections)}. Summarize it:\n\n{section}",
            max_tokens=1500,
        )
        partial_summaries.append(partial)
        if i < len(sections) - 1:
            time.sleep(5)

    combined = "\n\n".join(partial_summaries)
    return ask_llm(
        f"You combine partial summaries into one clear, well-structured summary. {STYLE_NOTE}",
        "These are summaries of consecutive parts of ONE document. "
        f"Combine them into a single summary:\n\n{combined}",
        max_tokens=2500,
    )


def summarize_overview(chunks, embeddings, page_count):
    """Long document: show the AI one representative piece per topic, in order."""
    picked = pick_representative_chunks(embeddings, OVERVIEW_EXCERPTS)
    parts = []
    for n, i in enumerate(picked, start=1):
        position = round(100 * i / max(len(chunks) - 1, 1))
        parts.append(f"[Excerpt {n}, about {position}% of the way through the document]\n{chunks[i]}")
    excerpts = "\n\n---\n\n".join(parts)

    return ask_llm(
        "You write clear study notes from excerpts of a long document. The excerpts were "
        f"chosen to represent every topic in the whole document, in document order. {STYLE_NOTE}",
        f"This document has about {page_count} pages. Below are {len(picked)} excerpts that "
        "represent its different topics, in document order.\n\n"
        "Write structured notes: one sentence on what the document is about, then a heading "
        "for each distinct topic the excerpts show (in order), with its key points as bullets. "
        "Do not invent details that are not in the excerpts.\n\n"
        f"{excerpts}",
        max_tokens=2000,
    )


def summarize_document(text, chunks, embeddings, page_count):
    """Returns (summary, mode). mode is 'full' or 'overview'."""
    if len(text) <= FULL_SUMMARY_MAX_CHARS:
        return summarize_in_parts(text), "full"
    return summarize_overview(chunks, embeddings, page_count), "overview"


def answer_question(chunks, chunk_embeddings, question):
    relevant = find_relevant_chunks(chunks, chunk_embeddings, question)
    context = "\n\n---\n\n".join(relevant)
    answer = ask_llm(
        "You answer questions using only the provided document excerpts. "
        "If the answer is not in the excerpts, say so clearly. "
        f"{STYLE_NOTE}",
        f"Document excerpts:\n{context}\n\nQuestion: {question}",
        max_tokens=1500,
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
        summary, summary_mode = summarize_document(text, chunks, chunk_embeddings, page_count)
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
        "summary_mode": summary_mode,
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