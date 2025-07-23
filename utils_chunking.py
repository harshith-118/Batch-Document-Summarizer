import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nltk.tokenize import sent_tokenize
import streamlit as st
import logging
from sklearn.cluster import AgglomerativeClustering
import numpy as np

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))

def get_openai_embeddings(sentences, client, model="text-embedding-ada-002"):
    try:
        response = client.embeddings.create(
            input=sentences,
            model=model
        )
        return [d.embedding for d in response.data]
    except Exception as e:
        st.error(f"❌ Error getting OpenAI embeddings: {e}")
        logging.error(f"Error getting OpenAI embeddings: {e}")
        return []

def semantic_chunking(text, client, tokenizer, chunk_size=CHUNK_SIZE, distance_threshold=1.0):
    """
    Semantic chunking using sentence embeddings and Agglomerative Clustering.
    Each cluster forms a chunk of semantically similar sentences.
    """
    try:
        # 1. Sentence tokenization
        sentences = sent_tokenize(text)
        if not sentences:
            return []
        
        # 2. Get embeddings
        embeddings = get_openai_embeddings(sentences, client)
        if not embeddings or len(embeddings) != len(sentences):
            st.error("❌ Embedding count mismatch or empty.")
            return []

        # 3. Clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            affinity='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(np.array(embeddings))

        # 4. Group sentences by cluster
        clusters = {}
        for label, sentence in zip(labels, sentences):
            clusters.setdefault(label, []).append(sentence)

        # 5. Assemble chunks, check token limits
        chunks = []
        for cluster_sentences in clusters.values():
            chunk = " ".join(cluster_sentences)
            tokens = count_tokens(chunk, tokenizer)
            # If chunk exceeds chunk_size, split further by sentence
            if tokens > chunk_size:
                temp_chunk = []
                temp_tokens = 0
                for sentence in cluster_sentences:
                    stokens = count_tokens(sentence, tokenizer)
                    if temp_tokens + stokens > chunk_size:
                        chunks.append(" ".join(temp_chunk))
                        temp_chunk = [sentence]
                        temp_tokens = stokens
                    else:
                        temp_chunk.append(sentence)
                        temp_tokens += stokens
                if temp_chunk:
                    chunks.append(" ".join(temp_chunk))
            else:
                chunks.append(chunk)
        return chunks
    except Exception as e:
        st.error(f"❌ Error during semantic chunking: {e}")
        logging.error(f"Error during semantic chunking: {e}")
        return []

def normal_chunking(text, tokenizer):
    try:
        tokenizer=tokenizer
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