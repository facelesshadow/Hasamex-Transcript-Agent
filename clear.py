import sqlite3

conn = sqlite3.connect("checkpoints.db")

tables = conn.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type='table'
""").fetchall()



conn.execute("DELETE FROM checkpoints")
conn.execute("DELETE FROM writes")
conn.execute("DELETE FROM streamlit_chats")
conn.execute("DELETE FROM streamlit_messages")
conn.execute("DELETE FROM sqlite_sequence")

conn.commit()
conn.close()

print(tables)