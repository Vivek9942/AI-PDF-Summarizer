# 📄 AI PDF Summarizer

An AI-powered tool that reads any PDF, generates a clear structured summary, and lets you ask follow-up questions about its content — all in seconds.

🔗 **Live Demo:** [ai-pdf-summarizer-vp.netlify.app](https://ai-pdf-summarizer-vp.netlify.app/)

---

## ✨ Features

- **Instant PDF Summarization** — Upload any PDF and get a clean, well-structured summary powered by an LLM.
- **Ask Questions (RAG-based Q&A)** — Ask anything about the uploaded document and get answers grounded only in its actual content.
- **Handles Long Documents** — Uses chunking + map-reduce summarization so large PDFs (50+ pages) are fully covered, not just the first few pages.
- **Relevant Context Retrieval** — Questions are matched against the most relevant sections of the document before being sent to the AI, instead of dumping the whole file every time.
- **Robust Error Handling** — Gracefully handles non-PDF uploads, empty files, scanned/image-only PDFs, and API rate limits with automatic retries.
- **Clean, Readable Output** — Markdown-formatted summaries and answers with proper headings, bullet points, and tables — no raw symbols.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| PDF Parsing | PyMuPDF |
| AI / LLM | Groq API (`openai/gpt-oss-20b`) |
| Frontend | HTML, CSS, JavaScript (vanilla) |
| Markdown Rendering | marked.js |
| Backend Hosting | Render |
| Frontend Hosting | Netlify |

---

## 🧠 How It Works

1. **Upload** — User uploads a PDF through the browser.
2. **Extract** — Backend extracts all text using PyMuPDF.
3. **Chunk** — Text is split into overlapping chunks to preserve context across boundaries.
4. **Summarize** — Short documents are summarized in one call; long documents are summarized section-by-section (map-reduce) and then combined into one coherent summary.
5. **Ask** — When a question is asked, the most relevant chunks (keyword-scored) are selected and sent to the LLM along with the question, so answers stay grounded in the actual document — even for large files.

---

## 📂 Project Structure

```
AI-PDF-Summarizer/
├── backend/
│   ├── main.py              # FastAPI app — upload, summarize, ask endpoints
│   ├── requirements.txt     # Python dependencies
│   └── Procfile              # Start command for deployment
├── frontend/
│   └── index.html            # Single-page UI (upload, summary, Q&A)
└── README.md
```

---

## 🚀 Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/Vivek9942/AI-PDF-Summarizer.git
cd AI-PDF-Summarizer
```

**2. Backend setup**
```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file inside `backend/`:
```
GROQ_API_KEY=your_groq_api_key_here
```

Run the server:
```bash
uvicorn main:app --reload
```

**3. Frontend**
Open `frontend/index.html` directly in your browser (or use a local server).

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/upload` | Upload a PDF, get back a summary |
| POST | `/ask` | Ask a question about an uploaded PDF |

---

## 🔮 Future Improvements

- Replace keyword-based chunk retrieval with vector embeddings for more accurate context matching
- Persistent storage (database) instead of in-memory document store
- Multi-file comparison and cross-document Q&A
- User authentication and document history
- OCR support for scanned/image-only PDFs

---

## 👤 Author

**Vivek Pandey**
[GitHub](https://github.com/Vivek9942)

---

⭐ If you found this project useful, consider giving it a star!
