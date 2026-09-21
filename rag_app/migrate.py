"""
Migration: Add user_id column to conversations table
"""
import psycopg2

conn = psycopg2.connect('postgresql://postgres:admin%40123@localhost:5433/RAG')
cur = conn.cursor()

# Add user_id column to conversations table (nullable for backward compatibility)
try:
    cur.execute("""
        ALTER TABLE conversations 
        ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id) ON DELETE SET NULL
    """)
    conn.commit()
    print("SUCCESS: Added user_id column to conversations table")
except Exception as e:
    conn.rollback()
    print(f"ERROR: {e}")

cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='conversations' ORDER BY ordinal_position")
print('conversations columns now:', cur.fetchall())

conn.close()
