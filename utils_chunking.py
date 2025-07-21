import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nltk.tokenize import sent_tokenize
import streamlit as st
import logging

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# You must set the tokenizer externally or pass it in

def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))

def semantic_chunking(text, embedding_model, tokenizer, chunk_size=CHUNK_SIZE):
    try:
        sentences = sent_tokenize(text)
        embeddings = embedding_model.encode(sentences)
        chunks = []
        current_chunk = []
        current_tokens = 0
        for sentence in sentences:
            tokens = count_tokens(sentence, tokenizer)
            if current_tokens + tokens > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = tokens
            else:
                current_chunk.append(sentence)
                current_tokens += tokens
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
    except Exception as e:
        st.error(f"❌ Error during semantic chunking: {e}")
        logging.error(f"Error during semantic chunking: {e}")
        return []

def normal_chunking(text, tokenizer):
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        return splitter.split_text(text)
    except Exception as e:
        st.error(f"❌ Error during normal chunking: {e}")
        logging.error(f"Error during normal chunking: {e}")
        return []

def batch_chunks(chunks, tokenizer, batch_token_limit=3500):
    batches = []
    current_batch = []
    current_tokens = 0
    for chunk in chunks:
        tokens = count_tokens(chunk, tokenizer)
        if current_tokens + tokens > batch_token_limit and current_batch:
            batches.append(current_batch)
            current_batch = [chunk]
            current_tokens = tokens
        else:
            current_batch.append(chunk)
            current_tokens += tokens
    if current_batch:
        batches.append(current_batch)
    return batches 