# main.py - Supabase PostgreSQL CRM Backend
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

# --- Pydantic Models for Data Validation ---
class InteractionCreate(BaseModel):
    """Model for creating a new interaction log."""
    agent_name: Optional[str] = Field(None, description="Name of the agent handling the interaction")
    customer_id: str = Field(..., description="Unique identifier for the customer")
    task_type: Optional[str] = Field(None, description="Type of task (e.g., Support, Sales, Billing)")
    task_description: str = Field(..., min_length=1, description="Description of the task")
    start_time: Optional[datetime] = Field(None, description="Start time in ISO format")
    end_time: Optional[datetime] = Field(None, description="End time in ISO format")
    status: Optional[str] = Field("Pending", description="Current status of the interaction")
    priority: Optional[str] = Field("Medium", description="Priority level (Low, Medium, High, Urgent)")
    tags: Optional[str] = Field(None, description="Comma-separated tags")
    notes: Optional[str] = Field(None, description="Additional notes")

    @validator('priority')
    def validate_priority(cls, v):
        if v and v not in ['Low', 'Medium', 'High', 'Urgent']:
            raise ValueError('Priority must be one of: Low, Medium, High, Urgent')
        return v

    @validator('status')
    def validate_status(cls, v):
        if v and v not in ['Pending', 'In Progress', 'Completed', 'Cancelled', 'On Hold']:
            raise ValueError('Status must be one of: Pending, In Progress, Completed, Cancelled, On Hold')
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
        if v and v not in ['Pending', 'In Progress', 'Completed', 'Cancelled', 'On Hold']:
            raise ValueError('Status must be one of: Pending, In Progress, Completed, Cancelled, On Hold')
        return v

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
                return (end - start).total_seconds() / 60
            except (ValueError, TypeError):
                logger.warning(f"Could not calculate duration for interaction {values.get('id')}")
                return None
        return None

class StatsResponse(BaseModel):
    """Response model for statistics."""
    total_interactions: int
    interactions_by_status: Dict[str, int]
    interactions_by_priority: Dict[str, int]
    average_duration_minutes: float
    top_agents: List[Dict[str, Union[str, int]]]
    recent_activity: List[Dict[str, Union[str, int]]]

# --- API Endpoints ---

@app.post("/interactions/", response_model=InteractionResponse, status_code=201)
async def create_interaction(interaction: InteractionCreate, db: Session = Depends(get_db)):
    """
    Creates a new customer interaction log in the CRM.
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
    db: Session = Depends(get_db),
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
        query = db.query(Interaction)
        
        # Apply filters
        if status:
            query = query.filter(Interaction.status == status)
        
        if priority:
            query = query.filter(Interaction.priority == priority)
        
        if agent_name:
            query = query.filter(Interaction.agent_name.ilike(f"%{agent_name}%"))
        
        if customer_id:
            query = query.filter(Interaction.customer_id == customer_id)
        
        if search:
            query = query.filter(
                (Interaction.task_description.ilike(f"%{search}%")) |
                (Interaction.notes.ilike(f"%{search}%"))
            )
        
        # Add ordering and pagination
        query = query.order_by(Interaction.created_at.desc()).offset(skip).limit(limit)
        
        interactions = query.all()
        
        # Convert to response models with duration calculation
        response_interactions = []
        for interaction in interactions:
            duration_minutes = None
            if interaction.start_time and interaction.end_time:
                duration_minutes = (interaction.end_time - interaction.start_time).total_seconds() / 60
            
            response_data = {
                "id": interaction.id,
                "timestamp": interaction.timestamp,
                "agent_name": interaction.agent_name,
                "customer_id": interaction.customer_id,
                "task_type": interaction.task_type,
                "task_description": interaction.task_description,
                "start_time": interaction.start_time,
                "end_time": interaction.end_time,
                "status": interaction.status,
                "priority": interaction.priority,
                "tags": interaction.tags,
                "notes": interaction.notes,
                "ai_classification": interaction.ai_classification,
                "ai_reason": interaction.ai_reason,
                "ai_suggestion": interaction.ai_suggestion,
                "processed_timestamp": interaction.processed_timestamp,
                "created_at": interaction.created_at,
                "updated_at": interaction.updated_at,
                "duration_minutes": duration_minutes
            }
            
            response_interactions.append(InteractionResponse(**response_data))
        
        return response_interactions
    except Exception as e:
        logger.error(f"Error retrieving interactions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve interactions")

@app.get("/interactions/{interaction_id}", response_model=InteractionResponse)
async def get_interaction_by_id(interaction_id: int, db: Session = Depends(get_db)):
    """
    Retrieves a single customer interaction log by its ID.
    """
    try:
        interaction = db.query(Interaction).filter(Interaction.id == interaction_id).first()
        
        if not interaction:
            raise HTTPException(status_code=404, detail="Interaction not found")
        
        # Calculate duration
        duration_minutes = None
        if interaction.start_time and interaction.end_time:
            duration_minutes = (interaction.end_time - interaction.start_time).total_seconds() / 60
        
        response_data = {
            "id": interaction.id,
            "timestamp": interaction.timestamp,
            "agent_name": interaction.agent_name,
            "customer_id": interaction.customer_id,
            "task_type": interaction.task_type,
            "task_description": interaction.task_description,
            "start_time": interaction.start_time,
            "end_time": interaction.end_time,
            "status": interaction.status,
            "priority": interaction.priority,
            "tags": interaction.tags,
            "notes": interaction.notes,
            "ai_classification": interaction.ai_classification,
            "ai_reason": interaction.ai_reason,
            "ai_suggestion": interaction.ai_suggestion,
            "processed_timestamp": interaction.processed_timestamp,
            "created_at": interaction.created_at,
            "updated_at": interaction.updated_at,
            "duration_minutes": duration_minutes
        }
        
        return InteractionResponse(**response_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving interaction {interaction_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve interaction")

@app.put("/interactions/{interaction_id}", response_model=InteractionResponse)
async def update_interaction(interaction_id: int, interaction: InteractionUpdate, db: Session = Depends(get_db)):
    """
    Updates an existing customer interaction log by its ID.
    """
    try:
        db_interaction = db.query(Interaction).filter(Interaction.id == interaction_id).first()
        
        if not db_interaction:
            raise HTTPException(status_code=404, detail="Interaction not found")
        
        # Update fields if provided
        update_data = interaction.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_interaction, field, value)
        
        db_interaction.updated_at = datetime.now()
        db.commit()
        db.refresh(db_interaction)
        
        # Calculate duration
        duration_minutes = None
        if db_interaction.start_time and db_interaction.end_time:
            duration_minutes = (db_interaction.end_time - db_interaction.start_time).total_seconds() / 60
        
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
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating interaction {interaction_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update interaction")

@app.delete("/interactions/{interaction_id}", status_code=204)
async def delete_interaction(interaction_id: int, db: Session = Depends(get_db)):
    """
    Deletes a customer interaction log by its ID.
    """
    try:
        interaction = db.query(Interaction).filter(Interaction.id == interaction_id).first()
        
        if not interaction:
            raise HTTPException(status_code=404, detail="Interaction not found")
        
        db.delete(interaction)
        db.commit()
        return {}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting interaction {interaction_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete interaction")

@app.get("/stats/", response_model=StatsResponse)
async def get_statistics(db: Session = Depends(get_db)):
    """
    Retrieves comprehensive statistics about interactions.
    """
    try:
        # Total interactions
        total_interactions = db.query(Interaction).count()
        
        # Interactions by status
        status_counts = db.query(Interaction.status, func.count(Interaction.id)).group_by(Interaction.status).all()
        interactions_by_status = {status: count for status, count in status_counts}
        
        # Interactions by priority
        priority_counts = db.query(Interaction.priority, func.count(Interaction.id)).group_by(Interaction.priority).all()
        interactions_by_priority = {priority: count for priority, count in priority_counts}
        
        # Average duration
        avg_duration = db.query(func.avg(
            func.extract('epoch', Interaction.end_time - Interaction.start_time) / 60
        )).filter(
            Interaction.start_time.isnot(None),
            Interaction.end_time.isnot(None)
        ).scalar()
        average_duration_minutes = float(avg_duration) if avg_duration else 0.0
        
        # Top agents
        top_agents = db.query(
            Interaction.agent_name,
            func.count(Interaction.id).label('count')
        ).filter(
            Interaction.agent_name.isnot(None)
        ).group_by(Interaction.agent_name).order_by(
            func.count(Interaction.id).desc()
        ).limit(5).all()
        
        top_agents_list = [{"agent_name": agent, "count": count} for agent, count in top_agents]
        
        # Recent activity (last 7 days)
        seven_days_ago = datetime.now() - timedelta(days=7)
        recent_activity = db.query(
            func.date(Interaction.created_at).label('date'),
            func.count(Interaction.id).label('count')
        ).filter(
            Interaction.created_at >= seven_days_ago
        ).group_by(
            func.date(Interaction.created_at)
        ).order_by(
            func.date(Interaction.created_at).desc()
        ).all()
        
        recent_activity_list = [{"date": str(date), "count": count} for date, count in recent_activity]
        
        return StatsResponse(
            total_interactions=total_interactions,
            interactions_by_status=interactions_by_status,
            interactions_by_priority=interactions_by_priority,
            average_duration_minutes=average_duration_minutes,
            top_agents=top_agents_list,
            recent_activity=recent_activity_list
        )
    except Exception as e:
        logger.error(f"Error retrieving statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")

@app.post("/interactions/bulk/", response_model=List[InteractionResponse])
async def create_bulk_interactions(interactions: List[InteractionCreate], db: Session = Depends(get_db)):
    """
    Creates multiple customer interactions in a single request.
    """
    if not interactions:
        raise HTTPException(status_code=400, detail="No interactions provided")
    
    if len(interactions) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 interactions per bulk request")
    
    try:
        created_interactions = []
        
        for interaction_data in interactions:
            db_interaction = Interaction(
                agent_name=interaction_data.agent_name,
                customer_id=interaction_data.customer_id,
                task_type=interaction_data.task_type,
                task_description=interaction_data.task_description,
                start_time=interaction_data.start_time,
                end_time=interaction_data.end_time,
                status=interaction_data.status,
                priority=interaction_data.priority,
                tags=interaction_data.tags,
                notes=interaction_data.notes
            )
            
            db.add(db_interaction)
            db.flush()  # Get the ID without committing
            
            # Calculate duration
            duration_minutes = None
            if db_interaction.start_time and db_interaction.end_time:
                duration_minutes = (db_interaction.end_time - db_interaction.start_time).total_seconds() / 60
            
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
            
            created_interactions.append(InteractionResponse(**response_data))
        
        db.commit()
        return created_interactions
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating bulk interactions: {e}")
        raise HTTPException(status_code=500, detail="Failed to create bulk interactions")

@app.get("/health/")
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint for monitoring.
    """
    try:
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
    try:
        query = db.query(Interaction).filter(
            (Interaction.task_description.ilike(f"%{q}%")) |
            (Interaction.notes.ilike(f"%{q}%")) |
            (Interaction.customer_id.ilike(f"%{q}%")) |
            (Interaction.agent_name.ilike(f"%{q}%"))
        ).order_by(Interaction.created_at.desc()).limit(limit)
        
        interactions = query.all()
        
        response_interactions = []
        for interaction in interactions:
            duration_minutes = None
            if interaction.start_time and interaction.end_time:
                duration_minutes = (interaction.end_time - interaction.start_time).total_seconds() / 60
            
            response_data = {
                "id": interaction.id,
                "timestamp": interaction.timestamp,
                "agent_name": interaction.agent_name,
                "customer_id": interaction.customer_id,
                "task_type": interaction.task_type,
                "task_description": interaction.task_description,
                "start_time": interaction.start_time,
                "end_time": interaction.end_time,
                "status": interaction.status,
                "priority": interaction.priority,
                "tags": interaction.tags,
                "notes": interaction.notes,
                "ai_classification": interaction.ai_classification,
                "ai_reason": interaction.ai_reason,
                "ai_suggestion": interaction.ai_suggestion,
                "processed_timestamp": interaction.processed_timestamp,
                "created_at": interaction.created_at,
                "updated_at": interaction.updated_at,
                "duration_minutes": duration_minutes
            }
            
            response_interactions.append(InteractionResponse(**response_data))
        
        return response_interactions
    except Exception as e:
        logger.error(f"Error searching interactions: {e}")
        raise HTTPException(status_code=500, detail="Failed to search interactions")

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