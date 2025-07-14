# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import sqlite3
import os
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Process Mining CRM Backend",
    description="A simple custom CRM backend for storing and retrieving customer interaction logs, designed for integration with n8n and AI analysis."
)

# Define the database file path
DATABASE_FILE = "crm_data.db"

# --- Database Initialization ---
def init_db():
    """Initializes the SQLite database and creates the interactions table if it doesn't exist."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            agent_name TEXT,
            customer_id TEXT NOT NULL,
            task_type TEXT,
            task_description TEXT NOT NULL,
            start_time TEXT,
            end_time TEXT,
            status TEXT,
            ai_classification TEXT,
            ai_reason TEXT,
            ai_suggestion TEXT,
            processed_timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database '{DATABASE_FILE}' initialized successfully.")

# Run database initialization when the application starts
# This will create the database file and table if they don't exist
init_db()

# --- Pydantic Models for Data Validation ---
class InteractionCreate(BaseModel):
    """Model for creating a new interaction log."""
    agent_name: Optional[str] = None
    customer_id: str
    task_type: Optional[str] = None
    task_description: str
    start_time: Optional[str] = None # ISO format string
    end_time: Optional[str] = None   # ISO format string
    status: Optional[str] = "Pending"

class InteractionUpdate(BaseModel):
    """Model for updating an existing interaction log."""
    agent_name: Optional[str] = None
    customer_id: Optional[str] = None
    task_type: Optional[str] = None
    task_description: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    status: Optional[str] = None
    ai_classification: Optional[str] = None
    ai_reason: Optional[str] = None
    ai_suggestion: Optional[str] = None
    processed_timestamp: Optional[str] = None

class InteractionInDB(InteractionCreate):
    """Model representing an interaction as stored in the database, including its ID."""
    id: int
    timestamp: str # Auto-generated creation timestamp

# --- API Endpoints ---

@app.post("/interactions/", response_model=InteractionInDB, status_code=201)
async def create_interaction(interaction: InteractionCreate):
    """
    Creates a new customer interaction log in the CRM.
    Automatically adds a timestamp for creation.
    """
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    current_timestamp = datetime.now().isoformat()

    try:
        cursor.execute(
            """
            INSERT INTO interactions (
                timestamp, agent_name, customer_id, task_type, task_description,
                start_time, end_time, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                current_timestamp,
                interaction.agent_name,
                interaction.customer_id,
                interaction.task_type,
                interaction.task_description,
                interaction.start_time,
                interaction.end_time,
                interaction.status,
            ),
        )
        conn.commit()
        new_id = cursor.lastrowid
        # Retrieve the newly created record to return it
        cursor.execute("SELECT * FROM interactions WHERE id = ?", (new_id,))
        new_interaction_data = cursor.fetchone()
        if new_interaction_data:
            # Map the fetched tuple to a dictionary for Pydantic model
            columns = [description[0] for description in cursor.description]
            new_interaction_dict = dict(zip(columns, new_interaction_data))
            return InteractionInDB(**new_interaction_dict)
        else:
            raise HTTPException(status_code=500, detail="Failed to retrieve created interaction.")
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        conn.close()

@app.get("/interactions/", response_model=List[InteractionInDB])
async def get_all_interactions():
    """
    Retrieves all customer interaction logs from the CRM.
    """
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM interactions")
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        return [InteractionInDB(**dict(zip(columns, row))) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        conn.close()

@app.get("/interactions/{interaction_id}", response_model=InteractionInDB)
async def get_interaction_by_id(interaction_id: int):
    """
    Retrieves a single customer interaction log by its ID.
    """
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM interactions WHERE id = ?", (interaction_id,))
        row = cursor.fetchone()
        if row:
            columns = [description[0] for description in cursor.description]
            return InteractionInDB(**dict(zip(columns, row)))
        raise HTTPException(status_code=404, detail="Interaction not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        conn.close()

@app.put("/interactions/{interaction_id}", response_model=InteractionInDB)
async def update_interaction(interaction_id: int, interaction: InteractionUpdate):
    """
    Updates an existing customer interaction log by its ID.
    This is where AI analysis results can be added.
    """
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    # Build the update query dynamically based on provided fields
    set_clauses = []
    values = []
    update_data = interaction.dict(exclude_unset=True) # Only include fields that are set

    if not update_data:
        raise HTTPException(status_code=400, detail="No fields provided for update.")

    for key, value in update_data.items():
        set_clauses.append(f"{key} = ?")
        values.append(value)

    values.append(interaction_id) # Add the ID for the WHERE clause

    try:
        cursor.execute(
            f"UPDATE interactions SET {', '.join(set_clauses)} WHERE id = ?",
            values
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Interaction not found")
        conn.commit()

        # Retrieve the updated record to return it
        cursor.execute("SELECT * FROM interactions WHERE id = ?", (interaction_id,))
        updated_interaction_data = cursor.fetchone()
        if updated_interaction_data:
            columns = [description[0] for description in cursor.description]
            updated_interaction_dict = dict(zip(columns, updated_interaction_data))
            return InteractionInDB(**updated_interaction_dict)
        else:
            raise HTTPException(status_code=500, detail="Failed to retrieve updated interaction.")
    except HTTPException as e:
        raise e # Re-raise HTTPExceptions to maintain their status code
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        conn.close()

@app.delete("/interactions/{interaction_id}", status_code=204)
async def delete_interaction(interaction_id: int):
    """
    Deletes a customer interaction log by its ID.
    """
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM interactions WHERE id = ?", (interaction_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Interaction not found")
        conn.commit()
        return {} # Return empty dict for 204 No Content
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        conn.close()

# Example of how to run this locally:
# 1. Save the code above as `main.py`
# 2. Open your terminal in the same directory
# 3. Install dependencies: `pip install "fastapi[all]" uvicorn`
# 4. Run the application: `uvicorn main:app --reload`
# 5. Access the API documentation at: http://127.0.0.1:8000/docs
