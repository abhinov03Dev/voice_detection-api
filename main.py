from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
from contextlib import asynccontextmanager
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from auth import validate_api_key
from detector import get_detector, preload_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for CPU-bound audio processing
executor = ThreadPoolExecutor(max_workers=4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload model at startup for low latency first request."""
    logger.info("Preloading voice detection model...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, preload_model)
    logger.info("Model preloaded successfully!")
    yield
    executor.shutdown(wait=True)


# Initialize FastAPI app
app = FastAPI(
    title="Voice Detection API",
    description="""
    A secure REST API that analyzes voice audio and determines whether 
    it is AI-generated or human speech.
    
    ## Features
    - Accepts Base64-encoded MP3 audio
    - Supports Tamil, English, Hindi, Malayalam, and Telugu
    - Returns classification with confidence score
    - Protected with API key authentication
    
    ## Authentication
    All requests must include the `x-api-key` header with a valid API key.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class AnalyzeRequest(BaseModel):
    language: Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"] = Field(
        ...,
        description="Language of the audio (Tamil / English / Hindi / Malayalam / Telugu)",
        examples=["Tamil", "English"]
    )
    audioFormat: Literal["mp3"] = Field(
        default="mp3",
        description="Audio format (always mp3)"
    )
    audioBase64: str = Field(
        ...,
        description="Base64-encoded MP3 audio data",
        min_length=100,
        examples=["SGVsbG8gV29ybGQ=..."]
    )


class AnalyzeResponse(BaseModel):
    status: Literal["success"] = Field(
        default="success",
        description="Status of the request"
    )
    language: str = Field(
        ...,
        description="Language of the audio"
    )
    classification: Literal["AI_GENERATED", "HUMAN"] = Field(
        ...,
        description="Classification result"
    )
    confidenceScore: float = Field(
        ...,
        description="Confidence score between 0.0 and 1.0",
        ge=0.0,
        le=1.0
    )
    explanation: str = Field(
        ...,
        description="Short reason for the decision"
    )


class ErrorResponse(BaseModel):
    status: Literal["error"] = Field(
        default="error",
        description="Status indicating error"
    )
    message: str = Field(..., description="Error message")


# Health check endpoint
@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "healthy",
        "service": "Voice Detection API",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": "Voice Detection API",
        "version": "1.0.0",
        "supported_languages": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    }


def _run_analysis(audio_base64: str):
    """Run analysis in thread pool to avoid blocking."""
    detector = get_detector()
    return detector.analyze(audio_base64)


# Main endpoint
@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={
        200: {"description": "Successful classification"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        422: {"model": ErrorResponse, "description": "Validation error"}
    },
    tags=["Voice Analysis"],
    summary="Analyze voice audio",
    description="Accepts Base64-encoded MP3 audio and returns whether the voice is AI-generated or human."
)
async def analyze_voice(
    request: AnalyzeRequest,
    api_key: str = Depends(validate_api_key)
):
    try:
        logger.info(f"Received voice analysis request for language: {request.language}")
        
        # Run CPU-bound analysis in thread pool for non-blocking
        loop = asyncio.get_event_loop()
        classification, confidence, explanation = await loop.run_in_executor(
            executor,
            _run_analysis,
            request.audioBase64
        )
        
        logger.info(f"Analysis complete: {classification} (confidence: {confidence:.2f})")
        
        return AnalyzeResponse(
            status="success",
            language=request.language,
            classification=classification,
            confidenceScore=round(confidence, 2),
            explanation=explanation
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "message": str(e)
            }
        )
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "message": f"Invalid API key or malformed request"
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
