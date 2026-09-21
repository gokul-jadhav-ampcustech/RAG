import psycopg2

conn = psycopg2.connect('postgresql://postgres:admin%40123@localhost:5433/RAG')
cur = conn.cursor()

cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='conversations' ORDER BY ordinal_position")
print('conversations columns:', cur.fetchall())

cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
print('tables:', cur.fetchall())

conn.close()
