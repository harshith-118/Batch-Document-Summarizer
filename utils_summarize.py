import logging
import streamlit as st
from nltk.tokenize import sent_tokenize

SYSTEM_PROMPT_DETAILED = (
    "You are a precise summarizer. For the following document chunk, extract and list all key facts, numbers, names, and important details. "
    "If there are tables (in markdown), summarize their main findings or trends. "
    "If there is OCR image text, include any numbers, names, or key phrases. "
    "Use bullet points if possible. Be concise, but do not omit important information."
)

def safe_summarize_batch(batch_text, model, max_tokens, system_prompt, user_prompt, cache_type, count_tokens, sha256_hash, get_cached_summary, cache_summary):
    context_limit = 16385 if model in ["gpt-3.5-turbo-16k", "gpt-4-32k"] else 4096 if model == "gpt-3.5-turbo" else 8192
    tokens = count_tokens(batch_text)
    if tokens > context_limit:
        # Try splitting by paragraphs
        parts = batch_text.split('\n\n')
        if len(parts) > 1:
            summaries = []
            for part in parts:
                if count_tokens(part) > context_limit:
                    summaries.append(safe_summarize_batch(part, model, max_tokens, system_prompt, user_prompt, cache_type, count_tokens, sha256_hash, get_cached_summary, cache_summary))
                else:
                    summaries.append(safe_summarize_batch(part, model, max_tokens, system_prompt, user_prompt, cache_type, count_tokens, sha256_hash, get_cached_summary, cache_summary))
            combined = '\n\n'.join(summaries)
            return safe_summarize_batch(combined, model, max_tokens, system_prompt, user_prompt, cache_type, count_tokens, sha256_hash, get_cached_summary, cache_summary)
        else:
            # Split by sentences
            sentences = sent_tokenize(batch_text)
            if len(sentences) > 1:
                summaries = []
                for sentence in sentences:
                    if count_tokens(sentence) > context_limit:
                        # Split by words
                        words = sentence.split()
                        if len(words) > 1:
                            for word in words:
                                if count_tokens(word) > context_limit:
                                    # As a last resort, truncate
                                    word = word[:context_limit]
                                return safe_summarize_batch(word, model, max_tokens, system_prompt, user_prompt, cache_type, count_tokens, sha256_hash, get_cached_summary, cache_summary)
                        else:
                            # Truncate the sentence
                            sentence = sentence[:context_limit]
                        return safe_summarize_batch(sentence, model, max_tokens, system_prompt, user_prompt, cache_type, count_tokens, sha256_hash, get_cached_summary, cache_summary)
                    else:
                        summaries.append(safe_summarize_batch(sentence, model, max_tokens, system_prompt, user_prompt, cache_type, count_tokens, sha256_hash, get_cached_summary, cache_summary))
                combined = ' '.join(summaries)
                return safe_summarize_batch(combined, model, max_tokens, system_prompt, user_prompt, cache_type, count_tokens, sha256_hash, get_cached_summary, cache_summary)
            else:
                # Truncate the text to fit the context window
                truncated = batch_text[:context_limit]
                st.warning("A chunk was truncated to fit the model's context window.")
                return safe_summarize_batch(truncated, model, max_tokens, system_prompt, user_prompt, cache_type, count_tokens, sha256_hash, get_cached_summary, cache_summary)
    # Normal summarization with cache
    batch_hash = sha256_hash(batch_text + model)
    cached = get_cached_summary(batch_hash)
    if cached:
        return cached
    try:
        import openai
        response = openai.OpenAI().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": batch_text}
            ],
            max_tokens=max_tokens,
            temperature=0.5
        )
        summary = response.choices[0].message.content.strip()
        cache_summary(batch_hash, summary, cache_type)
        return summary
    except Exception as e:
        st.error(f"❌ Error summarizing batch: {e}")
        logging.error(f"Error summarizing batch: {e}")
        return f"Error summarizing batch: {e}"

import hashlib
def sha256_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# summarize_map_reduce should be imported and used in the main app, passing all dependencies as needed. 