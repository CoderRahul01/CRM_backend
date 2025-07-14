# main.py (Updated for PostgreSQL with Supabase in mind, incorporating all new features)
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Database Configuration ---
# Get the database URL from environment variables
# IMPORTANT: When deploying to Render, you will set this as an environment variable
# This DATABASE_URL will be provided by Supabase.
# Example: postgresql://postgres:[YOUR-PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local_crm_data.db") # Fallback to SQLite for local dev

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


# --- FastAPI App Initialization ---
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
    allow_origins=["*"],  # Configure appropriately for production (e.g., ["https://your-dashboard.com"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Initialization (Create tables) ---
def init_db_postgres():
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
    init_db_postgres()

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
    db_interaction = Interaction(**interaction.dict(exclude_unset=True))
    db.add(db_interaction)
    db.commit()
    db.refresh(db_interaction)
    return db_interaction

@app.get("/interactions/", response_model=List[InteractionResponse])
async def get_all_interactions(
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    status: Optional[str] = Query(None, description="Filter by status"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    agent_name: Optional[str] = Query(None, description="Filter by agent name"),
    customer_id: Optional[str] = Query(None, description="Filter by customer ID"),
    search: Optional[str] = Query(None, description="Search in task description, notes, customer ID, and agent name")
):
    """
    Retrieves customer interaction logs from the CRM with filtering and pagination.
    """
    query = db.query(Interaction)

    if status:
        query = query.filter(Interaction.status == status)
    if priority:
        query = query.filter(Interaction.priority == priority)
    if agent_name:
        query = query.filter(Interaction.agent_name.ilike(f"%{agent_name}%"))
    if customer_id:
        query = query.filter(Interaction.customer_id == customer_id)
    if search:
        search_pattern = f"%{search}%"
        query = query.filter(
            (Interaction.task_description.ilike(search_pattern)) |
            (Interaction.notes.ilike(search_pattern)) |
            (Interaction.customer_id.ilike(search_pattern)) |
            (Interaction.agent_name.ilike(search_pattern))
        )

    interactions = query.order_by(Interaction.created_at.desc()).offset(skip).limit(limit).all()
    return interactions

@app.get("/interactions/{interaction_id}", response_model=InteractionResponse)
async def get_interaction_by_id(interaction_id: int, db: Session = Depends(get_db)):
    """
    Retrieves a single customer interaction log by its ID.
    """
    interaction = db.query(Interaction).filter(Interaction.id == interaction_id).first()
    if not interaction:
        raise HTTPException(status_code=404, detail="Interaction not found")
    return interaction

@app.put("/interactions/{interaction_id}", response_model=InteractionResponse)
async def update_interaction(interaction_id: int, interaction: InteractionUpdate, db: Session = Depends(get_db)):
    """
    Updates an existing customer interaction log by its ID.
    This is where AI analysis results can be added.
    """
    db_interaction = db.query(Interaction).filter(Interaction.id == interaction_id).first()
    if not db_interaction:
        raise HTTPException(status_code=404, detail="Interaction not found")

    update_data = interaction.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_interaction, key, value)

    db_interaction.updated_at = datetime.now() # Explicitly update timestamp
    db.add(db_interaction)
    db.commit()
    db.refresh(db_interaction)
    return db_interaction

@app.delete("/interactions/{interaction_id}", status_code=204)
async def delete_interaction(interaction_id: int, db: Session = Depends(get_db)):
    """
    Deletes a customer interaction log by its ID.
    """
    db_interaction = db.query(Interaction).filter(Interaction.id == interaction_id).first()
    if not db_interaction:
        raise HTTPException(status_code=404, detail="Interaction not found")

    db.delete(db_interaction)
    db.commit()
    return {}

# --- Advanced Endpoints ---

@app.get("/stats/", response_model=StatsResponse)
async def get_statistics(db: Session = Depends(get_db)):
    """
    Retrieves comprehensive statistics about interactions.
    """
    # Total interactions
    total_interactions = db.query(Interaction).count()

    # Interactions by status
    status_counts_raw = db.query(Interaction.status, func.count(Interaction.status)).group_by(Interaction.status).all()
    interactions_by_status = {status: count for status, count in status_counts_raw}

    # Interactions by priority
    priority_counts_raw = db.query(Interaction.priority, func.count(Interaction.priority)).group_by(Interaction.priority).all()
    interactions_by_priority = {priority: count for priority, count in priority_counts_raw}

    # Average duration
    # Note: func.avg is used for SQLAlchemy's AVG function.
    # The calculation for duration needs to be done in Python if not directly stored in DB
    # For simplicity, we'll fetch all and calculate in Python or rely on the InteractionResponse's validator
    all_interactions_with_duration = db.query(Interaction).filter(
        Interaction.start_time.isnot(None),
        Interaction.end_time.isnot(None)
    ).all()

    total_duration = 0
    valid_durations_count = 0
    for interaction in all_interactions_with_duration:
        if interaction.start_time and interaction.end_time:
            try:
                duration = (interaction.end_time - interaction.start_time).total_seconds() / 60
                total_duration += duration
                valid_durations_count += 1
            except Exception as e:
                logger.warning(f"Error calculating duration for interaction {interaction.id}: {e}")

    average_duration_minutes = total_duration / valid_durations_count if valid_durations_count > 0 else 0.0

    # Top agents (by interaction count)
    top_agents_raw = db.query(Interaction.agent_name, func.count(Interaction.id)).filter(
        Interaction.agent_name.isnot(None)
    ).group_by(Interaction.agent_name).order_by(func.count(Interaction.id).desc()).limit(5).all()
    top_agents = [{"agent_name": agent, "count": count} for agent, count in top_agents_raw]

    # Recent activity (daily counts for last 7 days)
    seven_days_ago = datetime.now() - timedelta(days=7)
    recent_activity_raw = db.query(
        func.date(Interaction.created_at), # Use func.date for PostgreSQL date extraction
        func.count(Interaction.id)
    ).filter(
        Interaction.created_at >= seven_days_ago
    ).group_by(
        func.date(Interaction.created_at)
    ).order_by(func.date(Interaction.created_at).desc()).all()

    recent_activity = [{"date": date.isoformat(), "count": count} for date, count in recent_activity_raw]

    return StatsResponse(
        total_interactions=total_interactions,
        interactions_by_status=interactions_by_status,
        interactions_by_priority=interactions_by_priority,
        average_duration_minutes=average_duration_minutes,
        top_agents=top_agents,
        recent_activity=recent_activity
    )

@app.post("/interactions/bulk/", response_model=List[InteractionResponse])
async def create_bulk_interactions(interactions: List[InteractionCreate], db: Session = Depends(get_db)):
    """
    Creates multiple customer interactions in a single request.
    """
    if not interactions:
        raise HTTPException(status_code=400, detail="No interactions provided")

    if len(interactions) > 100: # Limit bulk creation for performance/abuse prevention
        raise HTTPException(status_code=400, detail="Maximum 100 interactions per bulk request")

    db_interactions = []
    for interaction in interactions:
        db_interactions.append(Interaction(**interaction.dict(exclude_unset=True)))

    db.add_all(db_interactions)
    db.commit()
    for db_interaction in db_interactions:
        db.refresh(db_interaction) # Refresh each object to get its ID and generated timestamps

    return db_interactions

@app.get("/health/")
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint for monitoring.
    """
    try:
        # Attempt a simple query to check database connectivity
        interaction_count = db.query(Interaction).count()

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
    limit: int = Query(50, ge=1, le=100, description="Maximum results to return"),
    db: Session = Depends(get_db)
):
    """
    Full-text search across interactions.
    """
    search_pattern = f"%{q}%"
    interactions = db.query(Interaction).filter(
        (Interaction.task_description.ilike(search_pattern)) |
        (Interaction.notes.ilike(search_pattern)) |
        (Interaction.customer_id.ilike(search_pattern)) |
        (Interaction.agent_name.ilike(search_pattern))
    ).order_by(Interaction.created_at.desc()).limit(limit).all()

    return interactions

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
