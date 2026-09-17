import sqlite3
from pathlib import Path

db_path = Path("pipeline/dashboard/jobs.db")
if db_path.exists():
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        UPDATE jobs 
        SET youtube_url = NULL 
        WHERE youtube_url LIKE '%mock_%' 
           OR youtube_url LIKE '%DRY_RUN_%' 
           OR youtube_url LIKE '%ORLc4VHZARk%' 
           OR youtube_url LIKE '%_KESV02vwRc%'
    """)
    cur.execute("""
        UPDATE telemetry 
        SET youtube_url = NULL 
        WHERE youtube_url LIKE '%mock_%' 
           OR youtube_url LIKE '%DRY_RUN_%' 
           OR youtube_url LIKE '%ORLc4VHZARk%' 
           OR youtube_url LIKE '%_KESV02vwRc%'
    """)
    conn.commit()
    print("Cleaned mock URLs. Total changes:", conn.total_changes)
    conn.close()
else:
    print("Database not found")
