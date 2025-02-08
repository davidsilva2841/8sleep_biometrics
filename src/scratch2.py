import sqlite3

# Path to the SQLite database file
db_path = "/persistent/free-sleep-data/free-sleep.db"

def fetch_sleep_records():
    try:
        # Connect to the SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Query to select all rows from the sleep_records table
cursor.execute("SELECT * FROM sleep_records")

# Fetch all rows
rows = cursor.fetchall()

# Get column names for better output formatting
column_names = [description[0] for description in cursor.description]

import sqlite3

# Path to the SQLite database
db_path = "/persistent/free-sleep-data/free-sleep.db"

# Connect to the SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Query to get all table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

if not tables:
    print("No tables found in the database.")
    return

# Print schema for each table
for table_name, in tables:
    print(f"\n🔍 Schema for table: {table_name}")
    cursor.execute(f"PRAGMA table_info({table_name});")
    schema = cursor.fetchall()

    # Display column info
    if schema:
        print("------------------------------------------------------")
        print(f"{'CID':<5} {'Name':<20} {'Type':<10} {'Not Null':<10} {'Default':<15} {'PK':<5}")
        print("------------------------------------------------------")
        for col in schema:
            cid, name, col_type, notnull, dflt_value, pk = col
            print(f"{cid:<5} {name:<20} {col_type:<10} {notnull:<10} {str(dflt_value):<15} {pk:<5}")
    else:
        print("No schema information available.")

