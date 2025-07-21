import sqlite3
import pandas as pd

DB_PATH = "chunks_debug.db"

def main():
    print("\nSQLite Chunk Explorer for chunks_debug.db")
    print("Type your SQL query below. Type 'exit' or 'quit' to leave.\n")
    conn = sqlite3.connect(DB_PATH)
    while True:
        try:
            query = input("SQL> ").strip()
            if query.lower() in ("exit", "quit"):
                print("Exiting.")
                break
            if not query:
                continue
            df = pd.read_sql_query(query, conn)
            if df.empty:
                print("(No results)")
            else:
                print(df.to_markdown(index=False))
                save = input("Save results? (csv/md/skip): ").strip().lower()
                if save == "csv":
                    fname = input("Enter filename (e.g. results.csv): ").strip()
                    df.to_csv(fname, index=False)
                    print(f"Saved to {fname}")
                elif save == "md":
                    fname = input("Enter filename (e.g. results.md): ").strip()
                    with open(fname, "w") as f:
                        f.write(df.to_markdown(index=False))
                    print(f"Saved to {fname}")
        except Exception as e:
            print(f"Error: {e}")
    conn.close()

if __name__ == "__main__":
    main() 