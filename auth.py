import os
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable is required")
API_KEY_HEADER = APIKeyHeader(name="x-api-key", auto_error=False)


async def validate_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> str:
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "status": "error",
                "message": "Invalid API key or malformed request"
            }
        )
    
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "status": "error",
                "message": "Invalid API key or malformed request"
            }
        )
    
    return api_key
