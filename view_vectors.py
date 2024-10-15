import sqlite3

conn = sqlite3.connect('chroma/chroma.sqlite3')
cursor = conn.cursor()

# Replace 'embeddings_table' with the correct table name
cursor.execute("SELECT * FROM embeddings;")
vectors = cursor.fetchall()

for vector in vectors:
    print(vector)
