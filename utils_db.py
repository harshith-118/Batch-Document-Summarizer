import sqlite3
import datetime
import streamlit as st
import logging

def get_db_connection(db_path="chunks_debug.db"):
    return sqlite3.connect(db_path)

def ensure_summary_cache_table(db_path="chunks_debug.db"):
    conn = get_db_connection(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chunk_summaries (
            hash TEXT PRIMARY KEY,
            summary TEXT,
            type TEXT,
            created_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

def get_cached_summary(hash_val, db_path="chunks_debug.db"):
    conn = get_db_connection(db_path)
    c = conn.cursor()
    c.execute('SELECT summary FROM chunk_summaries WHERE hash=?', (hash_val,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def cache_summary(hash_val, summary, type_val, db_path="chunks_debug.db"):
    conn = get_db_connection(db_path)
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO chunk_summaries (hash, summary, type, created_at)
        VALUES (?, ?, ?, ?)
    ''', (hash_val, summary, type_val, datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()

def save_chunks_to_db(chunks, doc_name, db_path="chunks_debug.db"):
    try:
        conn = get_db_connection(db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS chunks_debug (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_name TEXT,
                chunk_index INTEGER,
                chunk_text TEXT
            )
        ''')
        for idx, chunk in enumerate(chunks):
            c.execute('''
                INSERT INTO chunks_debug (doc_name, chunk_index, chunk_text)
                VALUES (?, ?, ?)
            ''', (doc_name, idx, chunk))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"❌ Error saving chunks to DB: {e}")
        logging.error(f"Error saving chunks to DB: {e}") 