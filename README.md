# 📄 AI PDF Summarizer

Upload a PDF, get clear structured notes, then ask the document questions. The web app is called **Margin**.

🔗 **Live demo:** [ai-pdf-summarizer-vp.netlify.app](https://ai-pdf-summarizer-vp.netlify.app/)

> The backend runs on a free hosting plan. If nobody has used it for a while, the first request can take up to a minute while the server wakes up. The page shows a "Starting the server" step while it waits.

<!-- Add a screenshot here:  ![Margin screenshot](docs/screenshot.png) -->
<!-- Add a demo video or GIF link here -->

---

## ✨ Features

- **Structured summaries.** Short and medium PDFs are read part by part and combined into one set of notes.
- **Topic overview for long PDFs.** Long documents are grouped by topic, and one representative excerpt per topic goes to the AI. Every topic gets a place, including small ones.
- **Ask the document.** Questions are matched to the most relevant sections by meaning, not just keywords, so "why is this hard to crack for big numbers?" finds a section about computational hardness. Questions in other languages work too (tried with Hindi).
- **Clear errors, not crashes.** Wrong file type, empty file, password-protected PDF, unreadable PDF, scanned (image-only) PDF, too large, too many pages: each gives a readable message.
- **Safe with many users.** Every upload gets its own unique ID, so two people uploading `resume.pdf` never overwrite each other.
- **Privacy-friendly.** PDFs are read in memory and never saved to disk.
- **Readable output.** Markdown rendering (headings, bullets, tables) and math written in plain text instead of raw LaTeX.
- **Friendly with a sleeping free server.** The page wakes the backend as soon as it opens, shows progress stages, and explains what happened if the server restarts.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI |
| PDF parsing | PyMuPDF |
| Embeddings | fastembed (ONNX), `BAAI/bge-small-en-v1.5` |
| Search and grouping | NumPy (cosine similarity, k-means) |
| LLM | Groq API, `openai/gpt-oss-20b` |
| Frontend | HTML, CSS, JavaScript (no framework), marked.js |
| Hosting | Render (backend), Netlify (frontend) |
| Uptime | UptimeRobot (pings the backend every 5 minutes) |

---

## 🧠 How It Works

1. **Upload.** The browser sends the PDF to `POST /upload`.
2. **Validate.** The server checks file type, size, page count, password protection and that the PDF has text.
3. **Extract and chunk.** Text is read with PyMuPDF and split into overlapping chunks of about 1,000 characters.
4. **Embed.** Each chunk becomes a vector (a list of numbers that captures its meaning), computed a few at a time to keep memory low.
5. **Summarize.**
   - *Up to about 24,000 characters:* the text is summarized part by part (map-reduce) and combined.
   - *Longer:* the chunk vectors are grouped into 12 topics with k-means. The chunk nearest each group's centre is sent to the AI in one call, in document order. The page tells the reader this is an overview.
6. **Ask.** `POST /ask` embeds the question, finds the 6 most similar chunks by cosine similarity, and sends only those to the AI with the question. Answers stay grounded in the document.

---

## 📂 Project Structure

```
AI-PDF-Summarizer/
├── backend/
│   ├── main.py             # FastAPI app: upload, summarize, ask
│   ├── requirements.txt    # Python dependencies
│   └── Procfile            # Heroku-style start command (Render uses the command set in its dashboard)
├── frontend/
│   └── index.html          # Single-page UI: upload, progress, notes, Q&A
├── .gitignore
└── README.md
```

---

## 📡 API

Interactive docs are available at `/docs` on the backend.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/upload` | Send a PDF (form field `file`). Returns `doc_id`, `filename`, `num_pages`, `num_chars_extracted`, `num_chunks`, `summary`, `summary_mode` (`full` or `overview`) |
| POST | `/ask` | JSON body `{"doc_id": "...", "question": "..."}`. Returns `question`, `answer`, `chunks_used` |

```bash
curl -X POST https://ai-pdf-summarizer-cesv.onrender.com/upload -F "file=@notes.pdf"

curl -X POST https://ai-pdf-summarizer-cesv.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "<doc_id from the upload response>", "question": "What is a leaf node?"}'
```

---

## 🚀 Running Locally

**1. Clone**
```bash
git clone https://github.com/Vivek9942/AI-PDF-Summarizer.git
cd AI-PDF-Summarizer/backend
```

**2. Set up the backend** (Python 3.12 or newer)
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Create a file named `.env` inside `backend/`:
```
GROQ_API_KEY=your_groq_api_key_here
```
You can get a free key at [console.groq.com](https://console.groq.com).

**3. Run**
```bash
uvicorn main:app --reload
```
The first start downloads the small embedding model, so it takes a little longer.

**4. Open the frontend.** In `frontend/index.html`, change the `API` constant near the top of the script to `http://127.0.0.1:8000`, then open the file in your browser.

---

## ☁️ Deployment

- **Backend (Render, free plan):** root directory `backend`, build command `pip install -r requirements.txt`, start command `uvicorn main:app --host 0.0.0.0 --port $PORT`, environment variable `GROQ_API_KEY`.
- **Frontend (Netlify):** publish directory `frontend`, no build command.
- Both redeploy automatically on every push to `main`.

---

## ⚖️ Limits and Trade-offs

Built to run for free, so there are honest limits:

- **PDF size:** up to 25 MB and 200 pages. Both numbers are constants at the top of `backend/main.py`.
- **Long PDFs get an overview, not a line-by-line summary.** The AI sees 12 representative excerpts. A PDF with more than 12 distinct topics can still skip a small one. Ask a question to get the details of any topic.
- **Documents live in server memory.** The 10 most recent are kept. If the server restarts, press Summarize again.
- **Free AI plan limits.** The Groq free tier limits tokens per minute and per day, so very heavy use can be rate-limited. The server retries automatically.
- **No scanned PDFs yet.** PDFs that are only images have no text to read (OCR is on the roadmap).
- **Free hosting sleeps.** UptimeRobot keeps it awake, but a restart can still cause a short wait.

---

## 🧩 Engineering Notes

Problems met while building this, and what changed:

- **Ran out of memory on the free 512 MB server.** The first embedding setup (sentence-transformers with PyTorch) was too heavy. Switched to fastembed (ONNX), then cut embedding batches to 8 chunks with 1 thread.
- **Long PDFs lost whole topics.** The first summarizer split every big PDF into only 3 huge parts. Replaced it with topic grouping (k-means over chunk embeddings) so every topic reaches the AI.
- **Answers cut off mid-list.** Added an explicit length limit per AI call and a visible note when an answer still hits it.
- **Filename collisions.** Documents were stored by filename, so same-named uploads overwrote each other. Switched to a unique ID per upload.
- **Rate limits.** Added automatic retry with a wait when the AI service says "too many requests".
- **Sleeping free server.** The page now pings the backend on load and shows a "Starting the server" step.

---

## 🔮 Roadmap

- Download the summary as a file
- Saved history of past documents (database)
- Compare several PDFs
- OCR for scanned PDFs
- Automated tests
- User accounts

---

## 👤 Author

**Vivek Pandey**
[GitHub](https://github.com/Vivek9942)

---

⭐ If you found this project useful, consider giving it a star!
