import os
import uvicorn
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import the actual FastAPI application instance
from api.main import app

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    logging.info(f"Starting AsterScope API on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True)
