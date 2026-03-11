import os
import logging
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    """
    Validates the API key from the request header against the configured key.
    """
    valid_api_key = os.getenv("API_KEY")
    
    if not valid_api_key:
        logger.warning("No API_KEY environment variable set. Defaulting to open access for dev.")
        return api_key_header # Open access if not configured
        
    if api_key_header == valid_api_key:
        return api_key_header
        
    logger.warning("Invalid or missing API Key provided.")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing X-API-KEY header",
    )
