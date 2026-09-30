from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pymupdf
import os
import re
import math
import time
from collections import Counter
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"

# filename -> {"text": full text, "chunks": list of chunks} (resets when server restarts)
document_store = {}

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "in", "on",
    "at", "to", "for", "and", "or", "what", "which", "who", "how", "why", "when",
    "does", "do", "did", "this", "that", "it", "with", "about", "from", "by",
    "me", "tell", "explain", "give", "can", "you"
}

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
    filename: str
    question: str


@app.get("/")
def read_root():
    return {"message": "PDF Summarizer backend running"}


def extract_text_from_pdf(file_path):
    doc = pymupdf.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


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


def find_relevant_chunks(chunks, question, top_k=4):
    """Score each chunk by how many question words it contains, return the best ones."""
    words = set(re.findall(r"[a-z0-9]+", question.lower())) - STOPWORDS
    if not words:
        return chunks[:top_k]

    scored = []
    for i, chunk in enumerate(chunks):
        counts = Counter(re.findall(r"[a-z0-9]+", chunk.lower()))
        score = sum(counts[w] for w in words)
        if score > 0:
            scored.append((score, i))

    if not scored:
        return chunks[:top_k]

    scored.sort(reverse=True)
    best_indexes = sorted(i for _, i in scored[:top_k])  # keep original document order
    return [chunks[i] for i in best_indexes]


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

def answer_question(chunks, question):
    relevant = find_relevant_chunks(chunks, question)
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
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            f.write(content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        text = extract_text_from_pdf(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text found in PDF (might be scanned/image-only)")

    chunks = chunk_text(text)
    document_store[file.filename] = {"text": text, "chunks": chunks}

    try:
        summary = summarize_text(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

    return {
        "filename": file.filename,
        "num_chars_extracted": len(text),
        "num_chunks": len(chunks),
        "summary": summary,
    }


@app.post("/ask")
async def ask_question(req: AskRequest):
    if req.filename not in document_store:
        raise HTTPException(status_code=404, detail="Document not found. Please upload it first.")

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        answer, chunks_used = answer_question(document_store[req.filename]["chunks"], req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to answer: {str(e)}")

    return {"question": req.question, "answer": answer, "chunks_used": chunks_used}