import hashlib
import json
from datetime import datetime
import sqlite3

DB_PATH = "audit.db"  # Encrypt with hospital tools

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS decisions (id INTEGER PRIMARY KEY, timestamp TEXT, mrn TEXT, doctor TEXT, question TEXT, answer TEXT, hash TEXT)")
    conn.commit()
    conn.close()

def log_decision(mrn, visit_id, question, answer, doctor):
    init_db()
    timestamp = datetime.utcnow().isoformat()
    entry = f"{timestamp}{mrn}{doctor}{question}{answer}"
    entry_hash = hashlib.sha256(entry.encode()).hexdigest()
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO decisions VALUES (NULL, ?, ?, ?, ?, ?, ?)", (timestamp, mrn, doctor, question[:500], answer[:2000], entry_hash))
    conn.commit()
    conn.close()
    
    with open("merkle_chain.txt", "a") as f:
        f.write(entry_hash + "\n")
