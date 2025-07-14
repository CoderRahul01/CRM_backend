# AI-Powered Process Mining CRM Backend

A simple custom CRM backend for storing and retrieving customer interaction logs, designed for integration with n8n and AI analysis.

## Features

- FastAPI-based REST API
- SQLite database for data persistence
- Customer interaction logging
- AI analysis integration support
- Full CRUD operations for interactions

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation & Setup

### Option 1: Using requirements.txt (Recommended)

1. **Navigate to the project directory:**
   ```bash
   cd my-crm-backend
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using setup.py

1. **Navigate to the project directory:**
   ```bash
   cd my-crm-backend
   ```

2. **Install the package:**
   ```bash
   pip install -e .
   ```

## Running the Application

### Development Mode (with auto-reload)

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Documentation

Once the server is running, you can access:

- **Interactive API Documentation (Swagger UI):** http://127.0.0.1:8000/docs
- **Alternative API Documentation (ReDoc):** http://127.0.0.1:8000/redoc

## API Endpoints

- `POST /interactions/` - Create a new interaction
- `GET /interactions/` - Get all interactions
- `GET /interactions/{id}` - Get interaction by ID
- `PUT /interactions/{id}` - Update interaction
- `DELETE /interactions/{id}` - Delete interaction

## Database

The application uses SQLite with a database file named `crm_data.db` that will be created automatically when you first run the application.

## Example Usage

### Create an interaction:
```bash
curl -X POST "http://127.0.0.1:8000/interactions/" \
     -H "Content-Type: application/json" \
     -d '{
       "agent_name": "John Doe",
       "customer_id": "CUST001",
       "task_type": "Support",
       "task_description": "Customer inquiry about billing",
       "status": "Pending"
     }'
```

### Get all interactions:
```bash
curl -X GET "http://127.0.0.1:8000/interactions/"
```

## Troubleshooting

1. **Port already in use:** Change the port number in the uvicorn command
2. **Database errors:** Delete `crm_data.db` and restart the application
3. **Import errors:** Ensure all dependencies are installed with `pip install -r requirements.txt`

## Development

To add new features or modify the API:

1. Edit `main.py` to add new endpoints or modify existing ones
2. Restart the server (or use `--reload` flag for auto-reload)
3. Test your changes using the interactive documentation at `/docs` 