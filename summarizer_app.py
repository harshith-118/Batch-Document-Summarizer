# summarizer_app.py

import streamlit as st
import os
import tempfile
import time
import pandas as pd
import tiktoken
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Local utility imports
from utils_logging import setup_logging
from utils_pdf import extract_text_tables_images_from_pdf
from utils_chunking import count_tokens, semantic_chunking, normal_chunking, batch_chunks
from utils_db import ensure_summary_cache_table, get_cached_summary, cache_summary, save_chunks_to_db
from utils_summarize import safe_summarize_batch, SYSTEM_PROMPT_DETAILED, sha256_hash

# Setup logging
setup_logging()

# Initialize API clients and models
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
st.set_page_config(page_title="Document Summarizer", layout="wide")
st.title("📄 Batch Document Summarizer with OpenAI")

# Constants
MODEL_NAME = "gpt-4"
TOKEN_LIMIT = 10000
SUMMARY_TOKEN_TARGET = 500
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
DISPERSION_THRESHOLD = 60

# Initialize tokenizer and embedding model
tokenizer = tiktoken.encoding_for_model(MODEL_NAME)

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = get_embedding_model()

def summarize_map_reduce(chunks):
    """
    Summarize a list of text chunks using batch MapReduce, cache, and model selection.
    """
    ensure_summary_cache_table()
    batch_token_limit = 3500
    map_model = "gpt-3.5-turbo"
    reduce_model = MODEL_NAME
    
    batches = batch_chunks(chunks, tokenizer, batch_token_limit=batch_token_limit)
    batch_summaries = []
    
    for batch in batches:
        batch_text = "\n\n".join(batch)
        summary = safe_summarize_batch(
            batch_text=batch_text,
            model=map_model,
            max_tokens=300,
            system_prompt=SYSTEM_PROMPT_DETAILED,
            user_prompt=batch_text,
            cache_type="batch",
            count_tokens=lambda text: count_tokens(text, tokenizer),
            sha256_hash=sha256_hash,
            get_cached_summary=get_cached_summary,
            cache_summary=cache_summary,
            client=client
        )
        batch_summaries.append(summary)
        
    # Reduce step
    reduce_text = "\n\n".join(batch_summaries)
    final_summary = safe_summarize_batch(
        batch_text=reduce_text,
        model=reduce_model,
        max_tokens=SUMMARY_TOKEN_TARGET,
        system_prompt=SYSTEM_PROMPT_DETAILED,
        user_prompt=reduce_text,
        cache_type="final",
        count_tokens=lambda text: count_tokens(text, tokenizer),
        sha256_hash=sha256_hash,
        get_cached_summary=get_cached_summary,
        cache_summary=cache_summary,
        client=client
    )
    return final_summary

def process_documents(files, progress_callback=None, status_callback=None):
    """
    Process uploaded files: extract text, chunk, summarize, and collect timing info.
    """
    summaries = []
    errors = []
    total_files = len(files)

    for idx, file in enumerate(files):
        try:
            if status_callback:
                status_callback(f"Processing: `{file.name}` ({idx+1}/{total_files})")
            if progress_callback:
                progress_callback((idx) / total_files)
            
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(file.read())
                tmp_path = tmp_file.name

            start_time = time.time()
            
            # Text Extraction
            extract_start = time.time()
            text = extract_text_tables_images_from_pdf(tmp_path)
            extract_end = time.time()
            
            # Token counting and chunking decision
            token_count_val = count_tokens(text, tokenizer)
            chunk_start = time.time()
            dispersion = token_count_val / max(1, len(text.split(". ")))
            if token_count_val > TOKEN_LIMIT or dispersion > DISPERSION_THRESHOLD:
                chunks = semantic_chunking(text, embedding_model, tokenizer, chunk_size=CHUNK_SIZE)
                chunking_type = "Semantic"
            else:
                chunks = normal_chunking(text, tokenizer)
                chunking_type = "Normal"
            chunk_end = time.time()
            
            # Save chunks to DB for debugging
            save_chunks_to_db(chunks, file.name)
            
            # Warn if a chunk exceeds model context window
            chunk_warnings = [f"Chunk {i} exceeds model context window." for i, chunk in enumerate(chunks) if count_tokens(chunk, tokenizer) > TOKEN_LIMIT]
            
            # Summarization
            summary_start = time.time()
            summary = summarize_map_reduce(chunks)
            summary_end = time.time()
            
            summaries.append({
                "Filename": file.name,
                "Token Count": token_count_val,
                "Chunking Type": chunking_type,
                "Summary": summary,
                "Extraction Time (s)": round(extract_end - extract_start, 5),
                "Chunking Time (s)": round(chunk_end - chunk_start, 5),
                "Summarization Time (s)": round(summary_end - summary_start, 5),
                "Total Time (s)": round(time.time() - start_time, 5),
                "Chunk Warnings": "\n".join(chunk_warnings)
            })
            
            if progress_callback:
                progress_callback((idx+1) / total_files)
            
            try:
                os.remove(tmp_path)
            except Exception as e:
                st.error(f"❌ Error deleting temp file: {e}")

        except Exception as e:
            errors.append(f"{file.name}: {e}")
            st.error(f"❌ Error processing file {file.name}: {e}")

    try:
        df = pd.DataFrame(summaries)
        output_path = os.path.join(tempfile.gettempdir(), "document_summaries.xlsx")
        df.to_excel(output_path, index=False)
        return df, output_path, errors
    except Exception as e:
        st.error(f"❌ Error creating output DataFrame or Excel: {e}")
        return pd.DataFrame(), None, errors

# Streamlit UI
uploaded_files = st.file_uploader("Upload multiple PDF documents", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Run Summarization"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(val):
            progress_bar.progress(val)
            
        def status_callback(msg):
            status_text.info(msg)
            
        with st.spinner("⏳ Processing documents..."):
            df_result, excel_path, errors = process_documents(uploaded_files, progress_callback, status_callback)
            
        progress_bar.empty()
        status_text.empty()
        st.success("✅ Summarization complete!")
        
        for idx, row in df_result.iterrows():
            with st.expander(f"📄 {row['Filename']}"):
                st.markdown(f"**Chunking Type:** {row['Chunking Type']}")
                st.markdown("#### ⏱️ Time Taken (seconds)")
                st.markdown(f"- **Extraction:** {row['Extraction Time (s)']}")
                st.markdown(f"- **Chunking:** {row['Chunking Time (s)']}")
                st.markdown(f"- **Summarization:** {row['Summarization Time (s)']}")
                st.markdown(f"- **Total:** {row['Total Time (s)']}")
                if row['Chunk Warnings']:
                    st.warning(row['Chunk Warnings'])
                st.markdown("#### 📝 Summary")
                st.write(row['Summary'])
                
        if errors:
            st.error("Some files had errors:\n" + "\n".join(errors))
            
        if excel_path:
            with open(excel_path, "rb") as f:
                st.download_button("📥 Download Excel Report", data=f, file_name="summaries.xlsx") 