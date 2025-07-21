# Batch Document Summarizer with OpenAI

This is a powerful Streamlit web application designed to process and summarize multiple PDF documents in batches. It intelligently extracts text, tables, and images (using OCR), chunks the content, and leverages OpenAI's GPT models to generate high-quality, detail-oriented summaries.

---

## Key Features

- **Batch Processing**: Upload and summarize multiple PDF documents at once.
- **Multi-Content Extraction**: Reliably extracts text, tables (as Markdown), and text from images (using OCR) in the correct reading order.
- **Intelligent Chunking**:
    - **Semantic Chunking**: For large or unstructured documents, uses sentence embeddings to keep related context together.
    - **Normal Chunking**: A faster fallback for smaller, simpler documents.
- **Advanced Summarization**:
    - Uses a **MapReduce** pattern to handle documents of any size.
    - Utilizes a faster model (`gpt-3.5-turbo`) for initial chunk summaries and a more powerful model (`gpt-4`) for the final, cohesive summary.
- **Performance Optimization**:
    - **Caching**: Caches all chunk and final summaries in a local SQLite database to dramatically speed up reprocessing and reduce API costs.
    - **Robust Error Handling**: Gracefully handles errors during extraction, chunking, and summarization without crashing.
- **Interactive UI**:
    - Built with Streamlit for a clean and user-friendly interface.
    - Provides real-time progress bars and status updates.
    - Displays detailed results, timing metrics, and warnings for each document.
    - Allows downloading a full report as an Excel file.
- **Debugging Tools**: Includes a utility script (`explore_chunks_db.py`) to inspect the generated chunks using SQL queries.

---

## Tech Stack

- **Frontend**: Streamlit
- **PDF Processing**: `pdfplumber`
- **Image Processing & OCR**: `Pillow`, `pytesseract`
- **NLP & Machine Learning**: `sentence-transformers`, `nltk`, `langchain`
- **LLM Integration**: `openai`
- **Data Handling**: `pandas`, `openpyxl`
- **Database**: `sqlite3`

---

## Setup and Installation

### 1. Prerequisites

- Python 3.9+
- [Tesseract OCR Engine](https://github.com/tesseract-ocr/tesseract#installing-tesseract) installed and accessible in your system's PATH.

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Set Up OpenAI API Key

Create a `.streamlit/secrets.toml` file with:

```toml
OPENAI_API_KEY = "sk-your-openai-api-key-here"
```

### 6. Run the Application

```bash
streamlit run summarizer_app.py
```

---

## How to Use

1. **Launch the app** using the command above.
2. **Upload Files**: Drag and drop one or more PDF files into the file uploader.
3. **Run Summarization**: Click the "Run Summarization" button to start the process.
4. **View Results**: Expand each file's section to see the summary, chunking type, and performance metrics.
5. **Download Report**: Click the "Download Excel Report" button to save all results to a spreadsheet.

---

## Project Structure

- `summarizer_app.py`: The main application file containing the Streamlit UI and high-level workflow.
- `utils_pdf.py`: Handles all PDF extraction logic.
- `utils_chunking.py`: Contains functions for token counting and text chunking.
- `utils_db.py`: Manages all SQLite database interactions (caching, debugging).
- `utils_summarize.py`: Contains the core summarization logic and LLM prompts.
- `utils_logging.py`: Configures application-wide logging.
- `explore_chunks_db.py`: A command-line tool for inspecting the chunk database.
- `requirements.txt`: A list of all Python dependencies.

---

## License

MIT License

---

## Acknowledgements

- [OpenAI](https://openai.com/)
- [Streamlit](https://streamlit.io/)
- [pdfplumber](https://github.com/jsvine/pdfplumber)
- [Pillow](https://python-pillow.org/)
- [pytesseract](https://github.com/madmaze/pytesseract)
- [sentence-transformers](https://www.sbert.net/)
