import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="dw",
    user="postgres",
    password="123456"
)

cur = conn.cursor()
cur.execute("SELECT version();")
print(cur.fetchone())

cur.close()
conn.close()
