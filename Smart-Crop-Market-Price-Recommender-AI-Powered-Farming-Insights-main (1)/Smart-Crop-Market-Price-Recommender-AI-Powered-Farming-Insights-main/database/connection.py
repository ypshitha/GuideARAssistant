# database/connection.py
import os
import sqlite3
import pandas as pd
from typing import Optional, Dict, Any
import streamlit as st

DB_FILE = os.path.join(os.path.dirname(__file__), "agri.db")

class DatabaseConnection:
    def __init__(self, db_file=DB_FILE):
        self.db_file = db_file
        self.connection = None
        self._ensure_db()

    def _ensure_db(self):
        """Create sqlite DB and tables (if not exist)."""
        con = sqlite3.connect(self.db_file)
        cur = con.cursor()
        # farmer table (simple)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS farmers (
            id TEXT PRIMARY KEY,
            name TEXT,
            email TEXT,
            phone TEXT,
            location TEXT,
            state TEXT,
            farm_size_acres REAL,
            created_at TEXT
        );
        """)
        con.commit()
        con.close()

    def get_connection(self):
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_file, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
        return self.connection

    def execute_query(self, query: str, params: tuple = ()):
        con = self.get_connection()
        cur = con.cursor()
        cur.execute(query, params)
        con.commit()
        return cur

    def insert_farmer(self, farmer: Dict[str, Any]):
        cur = self.execute_query("""
            INSERT OR REPLACE INTO farmers (id, name, email, phone, location, state, farm_size_acres, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            farmer.get("id"),
            farmer.get("name"),
            farmer.get("email"),
            farmer.get("phone"),
            farmer.get("location"),
            farmer.get("state"),
            farmer.get("farm_size_acres"),
            farmer.get("created_at")
        ))
        return True

    def fetch_farmer_by_id(self, farmer_id: str) -> Optional[Dict]:
        cur = self.get_connection().cursor()
        cur.execute("SELECT * FROM farmers WHERE id = ?", (farmer_id,))
        row = cur.fetchone()
        if row:
            return dict(row)
        return None

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None

@st.cache_resource
def get_database():
    return DatabaseConnection()
