# main.py
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timedelta
import os
import logging
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Database Configuration ---
# Get the database URL from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

# SQLAlchemy setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Database Model ---
class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now, nullable=False)
    agent_name = Column(String)
    customer_id = Column(String, nullable=False, index=True)
    task_type = Column(String)
    task_description = Column(Text, nullable=False)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    status = Column(String, default="Pending", index=True)
    ai_classification = Column(String)
    ai_reason = Column(Text)
    ai_suggestion = Column(Text)
    processed_timestamp = Column(DateTime)
    priority = Column(String, default="Medium")
    tags = Column(String) # Storing as comma-separated string
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Process Mining CRM Backend",
    description="A comprehensive CRM backend for storing and retrieving customer interaction logs, designed for integration with n8n and AI analysis.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Initialization ---
def init_db():
    """Creates database tables if they don't exist."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("PostgreSQL database tables initialized successfully.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise HTTPException(status_code=500, detail="Database initialization failed")

# Dependency to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Run database initialization on startup
@app.on_event("startup")
async def startup_event():
    init_db()

# --- Database Initialization ---
def init_db():
    """Initializes the SQLite database and creates the interactions table if it doesn't exist."""
    try:
        with get_db_connection() as conn:
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
                    status TEXT DEFAULT 'Pending',
                    ai_classification TEXT,
                    ai_reason TEXT,
                    ai_suggestion TEXT,
                    processed_timestamp TEXT,
                    priority TEXT DEFAULT 'Medium',
                    tags TEXT,
                    notes TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_customer_id ON interactions(customer_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON interactions(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON interactions(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_name ON interactions(agent_name)')
            
            conn.commit()
            logger.info(f"Database '{DATABASE_FILE}' initialized successfully with indexes.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise HTTPException(status_code=500, detail="Database initialization failed")

# Run database initialization when the application starts
init_db()

# --- Pydantic Models for Data Validation ---
class InteractionCreate(BaseModel):
    """Model for creating a new interaction log."""
    agent_name: Optional[str] = Field(None, description="Name of the agent handling the interaction")
    customer_id: str = Field(..., description="Unique identifier for the customer")
    task_type: Optional[str] = Field(None, description="Type of task (e.g., Support, Sales, Billing)")
    task_description: str = Field(..., min_length=1, description="Description of the task")
    start_time: Optional[datetime] = Field(None, description="Start time in ISO format (e.g., '2023-07-15T10:00:00')")
    end_time: Optional[datetime] = Field(None, description="End time in ISO format (e.g., '2023-07-15T10:30:00')")
    status: Optional[str] = Field("Pending", description="Current status of the interaction")
    priority: Optional[str] = Field("Medium", description="Priority level (Low, Medium, High, Urgent)")
    tags: Optional[str] = Field(None, description="Comma-separated tags (e.g., 'urgent,escalated')")
    notes: Optional[str] = Field(None, description="Additional notes")

    @validator('priority')
    def validate_priority(cls, v):
        if v and v not in ['Low', 'Medium', 'High', 'Urgent']:
            raise ValueError('Priority must be one of: Low, Medium, High, Urgent')
        return v

    @validator('status')
    def validate_status(cls, v):
        if v and v not in ['Pending', 'In Progress', 'Completed', 'Cancelled', 'On Hold', 'Analyzed - Action Required', 'Analyzed - Efficient']:
            raise ValueError('Status must be one of: Pending, In Progress, Completed, Cancelled, On Hold, Analyzed - Action Required, Analyzed - Efficient')
        return v

class InteractionUpdate(BaseModel):
    """Model for updating an existing interaction log."""
    agent_name: Optional[str] = None
    customer_id: Optional[str] = None
    task_type: Optional[str] = None
    task_description: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    tags: Optional[str] = None
    notes: Optional[str] = None
    ai_classification: Optional[str] = None
    ai_reason: Optional[str] = None
    ai_suggestion: Optional[str] = None
    processed_timestamp: Optional[datetime] = None

    @validator('priority')
    def validate_priority(cls, v):
        if v and v not in ['Low', 'Medium', 'High', 'Urgent']:
            raise ValueError('Priority must be one of: Low, Medium, High, Urgent')
        return v

    @validator('status')
    def validate_status(cls, v):
        if v and v not in ['Pending', 'In Progress', 'Completed', 'Cancelled', 'On Hold', 'Analyzed - Action Required', 'Analyzed - Efficient']:
            raise ValueError('Status must be one of: Pending, In Progress, Completed, Cancelled, On Hold, Analyzed - Action Required, Analyzed - Efficient')
        return v

class InteractionInDB(InteractionCreate):
    """Model representing an interaction as stored in the database, including its ID."""
    id: int
    timestamp: datetime
    created_at: datetime
    updated_at: datetime

class InteractionResponse(BaseModel):
    """Response model for interactions with computed fields."""
    id: int
    timestamp: datetime
    agent_name: Optional[str]
    customer_id: str
    task_type: Optional[str]
    task_description: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    status: str
    priority: str
    tags: Optional[str]
    notes: Optional[str]
    ai_classification: Optional[str]
    ai_reason: Optional[str]
    ai_suggestion: Optional[str]
    processed_timestamp: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    duration_minutes: Optional[float] = None

    @validator('duration_minutes', always=True)
    def calculate_duration(cls, v, values):
        start = values.get('start_time')
        end = values.get('end_time')
        if start and end:
            try:
                # Ensure start and end are datetime objects
                if isinstance(start, str):
                    start = datetime.fromisoformat(start)
                if isinstance(end, str):
                    end = datetime.fromisoformat(end)
                return (end - start).total_seconds() / 60
            except (ValueError, TypeError):
                logger.warning(f"Could not calculate duration for interaction {values.get('id')}. Start: {start}, End: {end}")
                return None
        return None

class StatsResponse(BaseModel):
    """Response model for statistics."""
    total_interactions: int
    interactions_by_status: Dict[str, int]
    interactions_by_priority: Dict[str, int]
    average_duration_minutes: float
    top_agents: List[Dict[str, Union[str, int]]]
    recent_activity: List[Dict[str, Union[str, int]]] # Daily counts for last 7 days

# --- API Endpoints ---

@app.post("/interactions/", response_model=InteractionResponse, status_code=201)
async def create_interaction(interaction: InteractionCreate, db: Session = Depends(get_db)):
    """
    Creates a new customer interaction log in the CRM.
    Automatically adds a timestamp for creation.
    """
    try:
        # Create new interaction instance
        db_interaction = Interaction(
            agent_name=interaction.agent_name,
            customer_id=interaction.customer_id,
            task_type=interaction.task_type,
            task_description=interaction.task_description,
            start_time=interaction.start_time,
            end_time=interaction.end_time,
            status=interaction.status,
            priority=interaction.priority,
            tags=interaction.tags,
            notes=interaction.notes
        )
        
        db.add(db_interaction)
        db.commit()
        db.refresh(db_interaction)
        
        # Calculate duration
        duration_minutes = None
        if db_interaction.start_time and db_interaction.end_time:
            duration_minutes = (db_interaction.end_time - db_interaction.start_time).total_seconds() / 60
        
        # Create response with duration
        response_data = {
            "id": db_interaction.id,
            "timestamp": db_interaction.timestamp,
            "agent_name": db_interaction.agent_name,
            "customer_id": db_interaction.customer_id,
            "task_type": db_interaction.task_type,
            "task_description": db_interaction.task_description,
            "start_time": db_interaction.start_time,
            "end_time": db_interaction.end_time,
            "status": db_interaction.status,
            "priority": db_interaction.priority,
            "tags": db_interaction.tags,
            "notes": db_interaction.notes,
            "ai_classification": db_interaction.ai_classification,
            "ai_reason": db_interaction.ai_reason,
            "ai_suggestion": db_interaction.ai_suggestion,
            "processed_timestamp": db_interaction.processed_timestamp,
            "created_at": db_interaction.created_at,
            "updated_at": db_interaction.updated_at,
            "duration_minutes": duration_minutes
        }
        
        return InteractionResponse(**response_data)
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating interaction: {e}")
        raise HTTPException(status_code=500, detail="Failed to create interaction")

@app.get("/interactions/", response_model=List[InteractionResponse])
async def get_all_interactions(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    status: Optional[str] = Query(None, description="Filter by status"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    agent_name: Optional[str] = Query(None, description="Filter by agent name"),
    customer_id: Optional[str] = Query(None, description="Filter by customer ID"),
    search: Optional[str] = Query(None, description="Search in task description and notes")
):
    """
    Retrieves customer interaction logs from the CRM with filtering and pagination.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Build query with filters
            query = "SELECT * FROM interactions WHERE 1=1"
            params = []
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            if priority:
                query += " AND priority = ?"
                params.append(priority)
            
            if agent_name:
                query += " AND agent_name LIKE ?"
                params.append(f"%{agent_name}%")
            
            if customer_id:
                query += " AND customer_id = ?"
                params.append(customer_id)
            
            if search:
                query += " AND (task_description LIKE ? OR notes LIKE ?)"
                search_term = f"%{search}%"
                params.extend([search_term, search_term])
            
            # Add ordering and pagination
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, skip])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to response models with duration calculation
            interactions = []
            for row in rows:
                interaction_dict = dict(row)
                duration_minutes = None
                
                if interaction_dict.get('start_time') and interaction_dict.get('end_time'):
                    try:
                        start = datetime.fromisoformat(interaction_dict['start_time'])
                        end = datetime.fromisoformat(interaction_dict['end_time'])
                        duration_minutes = (end - start).total_seconds() / 60
                    except ValueError:
                        pass
                
                interaction_dict['duration_minutes'] = duration_minutes
                interactions.append(InteractionResponse(**interaction_dict))
            
            return interactions
    except Exception as e:
        logger.error(f"Error retrieving interactions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve interactions")

@app.get("/interactions/{interaction_id}", response_model=InteractionResponse)
async def get_interaction_by_id(interaction_id: int):
    """
    Retrieves a single customer interaction log by its ID.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM interactions WHERE id = ?", (interaction_id,))
            row = cursor.fetchone()
            
            if row:
                interaction_dict = dict(row)
                duration_minutes = None
                
                if interaction_dict.get('start_time') and interaction_dict.get('end_time'):
                    try:
                        start = datetime.fromisoformat(interaction_dict['start_time'])
                        end = datetime.fromisoformat(interaction_dict['end_time'])
                        duration_minutes = (end - start).total_seconds() / 60
                    except ValueError:
                        pass
                
                interaction_dict['duration_minutes'] = duration_minutes
                return InteractionResponse(**interaction_dict)
            
            raise HTTPException(status_code=404, detail="Interaction not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving interaction {interaction_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve interaction")

@app.put("/interactions/{interaction_id}", response_model=InteractionResponse)
async def update_interaction(interaction_id: int, interaction: InteractionUpdate):
    """
    Updates an existing customer interaction log by its ID.
    This is where AI analysis results can be added.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Build the update query dynamically based on provided fields
            set_clauses = []
            values = []
            update_data = interaction.dict(exclude_unset=True) # Only include fields that are set

            if not update_data:
                raise HTTPException(status_code=400, detail="No fields provided for update.")

            # Add updated_at timestamp
            set_clauses.append("updated_at = datetime('now')")
            
            for key, value in update_data.items():
                set_clauses.append(f"{key} = ?")
                values.append(value)

            values.append(interaction_id) # Add the ID for the WHERE clause

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
                interaction_dict = dict(updated_interaction_data)
                duration_minutes = None
                
                if interaction_dict.get('start_time') and interaction_dict.get('end_time'):
                    try:
                        start = datetime.fromisoformat(interaction_dict['start_time'])
                        end = datetime.fromisoformat(interaction_dict['end_time'])
                        duration_minutes = (end - start).total_seconds() / 60
                    except ValueError:
                        pass
                
                interaction_dict['duration_minutes'] = duration_minutes
                return InteractionResponse(**interaction_dict)
            else:
                raise HTTPException(status_code=500, detail="Failed to retrieve updated interaction.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating interaction {interaction_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update interaction")

@app.delete("/interactions/{interaction_id}", status_code=204)
async def delete_interaction(interaction_id: int):
    """
    Deletes a customer interaction log by its ID.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM interactions WHERE id = ?", (interaction_id,))
            
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Interaction not found")
            
            conn.commit()
            return {} # Return empty dict for 204 No Content
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting interaction {interaction_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete interaction")

# --- Advanced Endpoints ---

@app.get("/stats/", response_model=StatsResponse)
async def get_statistics():
    """
    Retrieves comprehensive statistics about interactions.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Total interactions
            cursor.execute("SELECT COUNT(*) FROM interactions")
            total_interactions = cursor.fetchone()[0]
            
            # Interactions by status
            cursor.execute("""
                SELECT status, COUNT(*) as count 
                FROM interactions 
                GROUP BY status
            """)
            status_counts = {row['status']: row['count'] for row in cursor.fetchall()}
            
            # Interactions by priority
            cursor.execute("""
                SELECT priority, COUNT(*) as count 
                FROM interactions 
                GROUP BY priority
            """)
            priority_counts = {row['priority']: row['count'] for row in cursor.fetchall()}
            
            # Average duration
            cursor.execute("""
                SELECT AVG(
                    (julianday(end_time) - julianday(start_time)) * 24 * 60
                ) as avg_duration
                FROM interactions 
                WHERE start_time IS NOT NULL AND end_time IS NOT NULL
            """)
            avg_duration_result = cursor.fetchone()
            average_duration_minutes = avg_duration_result[0] if avg_duration_result[0] else 0.0
            
            # Top agents
            cursor.execute("""
                SELECT agent_name, COUNT(*) as count 
                FROM interactions 
                WHERE agent_name IS NOT NULL
                GROUP BY agent_name 
                ORDER BY count DESC 
                LIMIT 5
            """)
            top_agents = [{"agent_name": row['agent_name'], "count": row['count']} for row in cursor.fetchall()]
            
            # Recent activity (last 7 days)
            cursor.execute("""
                SELECT DATE(created_at) as date, COUNT(*) as count 
                FROM interactions 
                WHERE created_at >= datetime('now', '-7 days')
                GROUP BY DATE(created_at)
                ORDER BY date DESC
            """)
            recent_activity = [{"date": row['date'], "count": row['count']} for row in cursor.fetchall()]
            
            return StatsResponse(
                total_interactions=total_interactions,
                interactions_by_status=status_counts,
                interactions_by_priority=priority_counts,
                average_duration_minutes=average_duration_minutes,
                top_agents=top_agents,
                recent_activity=recent_activity
            )
    except Exception as e:
        logger.error(f"Error retrieving statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")

@app.post("/interactions/bulk/", response_model=List[InteractionResponse])
async def create_bulk_interactions(interactions: List[InteractionCreate]):
    """
    Creates multiple customer interactions in a single request.
    """
    if not interactions:
        raise HTTPException(status_code=400, detail="No interactions provided")
    
    if len(interactions) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 interactions per bulk request")
    
    created_interactions = []
    current_timestamp = datetime.now().isoformat()
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            for interaction in interactions:
                cursor.execute(
                    """
                    INSERT INTO interactions (
                        timestamp, agent_name, customer_id, task_type, task_description,
                        start_time, end_time, status, priority, tags, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        interaction.priority,
                        interaction.tags,
                        interaction.notes,
                    ),
                )
                new_id = cursor.lastrowid
                
                # Retrieve the created interaction
                cursor.execute("SELECT * FROM interactions WHERE id = ?", (new_id,))
                new_interaction_data = cursor.fetchone()
                
                if new_interaction_data:
                    interaction_dict = dict(new_interaction_data)
                    duration_minutes = None
                    
                    if interaction_dict.get('start_time') and interaction_dict.get('end_time'):
                        try:
                            start = datetime.fromisoformat(interaction_dict['start_time'])
                            end = datetime.fromisoformat(interaction_dict['end_time'])
                            duration_minutes = (end - start).total_seconds() / 60
                        except ValueError:
                            pass
                    
                    interaction_dict['duration_minutes'] = duration_minutes
                    created_interactions.append(InteractionResponse(**interaction_dict))
            
            conn.commit()
            return created_interactions
    except Exception as e:
        logger.error(f"Error creating bulk interactions: {e}")
        raise HTTPException(status_code=500, detail="Failed to create bulk interactions")

@app.get("/health/")
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM interactions")
            interaction_count = cursor.fetchone()[0]
            
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "database": "connected",
                "total_interactions": interaction_count,
                "version": "1.0.0"
            }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/search/", response_model=List[InteractionResponse])
async def search_interactions(
    q: str = Query(..., description="Search query"),
    limit: int = Query(50, ge=1, le=100, description="Maximum results to return")
):
    """
    Full-text search across interactions.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Search in multiple fields
            query = """
                SELECT * FROM interactions 
                WHERE task_description LIKE ? 
                   OR notes LIKE ? 
                   OR customer_id LIKE ? 
                   OR agent_name LIKE ?
                ORDER BY created_at DESC 
                LIMIT ?
            """
            search_term = f"%{q}%"
            cursor.execute(query, [search_term, search_term, search_term, search_term, limit])
            rows = cursor.fetchall()
            
            interactions = []
            for row in rows:
                interaction_dict = dict(row)
                duration_minutes = None
                
                if interaction_dict.get('start_time') and interaction_dict.get('end_time'):
                    try:
                        start = datetime.fromisoformat(interaction_dict['start_time'])
                        end = datetime.fromisoformat(interaction_dict['end_time'])
                        duration_minutes = (end - start).total_seconds() / 60
                    except ValueError:
                        pass
                
                interaction_dict['duration_minutes'] = duration_minutes
                interactions.append(InteractionResponse(**interaction_dict))
            
            return interactions
    except Exception as e:
        logger.error(f"Error searching interactions: {e}")
        raise HTTPException(status_code=500, detail="Failed to search interactions")

# --- Root Endpoint ---
@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "AI-Powered Process Mining CRM Backend",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/health",
        "endpoints": {
            "interactions": "/interactions/",
            "statistics": "/stats/",
            "search": "/search/",
            "bulk_create": "/interactions/bulk/"
        }
    }

# Example of how to run this locally:
# 1. Save the code above as `main.py`
# 2. Open your terminal in the same directory
# 3. Install dependencies: `pip install -r requirements.txt`
# 4. Run the application: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`
# 5. Access the API documentation at: http://127.0.0.1:8000/docs 