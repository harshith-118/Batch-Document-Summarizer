import sqlite3
import datetime
import streamlit as st
import logging
import pandas as pd

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

def get_db_connection(db_path="document_info.db"):
    return sqlite3.connect(db_path)

def ensure_document_tables(db_path="document_info.db"):
    conn = get_db_connection(db_path)
    c = conn.cursor()
    # Document-level table
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            size_kb REAL,
            upload_time REAL,
            extract_time REAL,
            chunking_time REAL,
            summary_time REAL,
            total_tokens INTEGER,
            timestamp TEXT
        )
    ''')
    # Page-level table
    c.execute('''
        CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_name TEXT,
            page_number INTEGER,
            num_tokens INTEGER,
            num_tables INTEGER,
            num_images INTEGER,
            num_paragraphs INTEGER,
            extract_time REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def store_document_info(
    name, size_kb, upload_time, extract_time, chunking_time,summary_time, total_tokens, timestamp, db_path="document_info.db"
):
    ensure_document_tables(db_path)
    conn = get_db_connection(db_path)
    c = conn.cursor()
    c.execute('''
        INSERT INTO documents (name, size_kb, upload_time, extract_time, chunking_time, summary_time, total_tokens, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (name, size_kb, upload_time, extract_time, chunking_time, summary_time, total_tokens, timestamp))
    conn.commit()
    conn.close()

def store_page_info(
    document_name, page_number, num_tokens, num_paragraphs, num_tables, num_images, extract_time, timestamp, db_path="document_info.db"
):
    ensure_document_tables(db_path)
    conn = get_db_connection(db_path)
    c = conn.cursor()
    c.execute('''
        INSERT INTO pages (document_name, page_number, num_tokens, num_paragraphs, num_tables, num_images, extract_time, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (document_name, page_number, num_tokens, num_paragraphs, num_tables, num_images, extract_time, timestamp))
    conn.commit()
    conn.close()

def export_db_to_excel(db_path, excel_path):
    """
    Export document and page info from the database to an Excel file.
    Assumes tables: documents, pages.
    """
    conn = sqlite3.connect(db_path)
    try:
        # Read tables into DataFrames
        doc_df = pd.read_sql_query("SELECT * FROM documents", conn)
        page_df = pd.read_sql_query("SELECT * FROM pages", conn)
        # Write to Excel with two sheets
        with pd.ExcelWriter(excel_path) as writer:
            doc_df.to_excel(writer, sheet_name="Documents", index=False)
            page_df.to_excel(writer, sheet_name="Pages", index=False)
    finally:
        conn.close()

if __name__ == "__main__":
    export_db_to_excel("document_info.db","exported_info.xlsx")