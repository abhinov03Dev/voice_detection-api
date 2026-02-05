from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field
from typing import Literal
from contextlib import asynccontextmanager
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from auth import validate_api_key
from detector import get_detector, preload_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


executor = ThreadPoolExecutor(max_workers=4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Preloading model...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, preload_model)
    logger.info("Model ready!")
    yield
    executor.shutdown(wait=True)


app = FastAPI(
    title="Voice Detection API",
    description="AI vs Human voice classification API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    default_response_class=ORJSONResponse  
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    language: Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"] = Field(...)
    audioFormat: Literal["mp3"] = Field(default="mp3")
    audioBase64: str = Field(..., min_length=100)
    
    model_config = {"extra": "ignore"} 


class AnalyzeResponse(BaseModel):
    status: Literal["success"] = "success"
    language: str
    classification: Literal["AI_GENERATED", "HUMAN"]
    confidenceScore: float = Field(ge=0.0, le=1.0)
    explanation: str


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    message: str


@app.get("/", tags=["Health"])
async def root():
    return {"status": "healthy", "service": "Voice Detection API", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": "Voice Detection API",
        "version": "1.0.0",
        "supported_languages": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    }


def _run_analysis(audio_base64: str):
    detector = get_detector()
    return detector.analyze(audio_base64)


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        422: {"model": ErrorResponse}
    },
    tags=["Voice Analysis"]
)
async def analyze_voice(
    request: AnalyzeRequest,
    api_key: str = Depends(validate_api_key)
):
    try:
        loop = asyncio.get_event_loop()
        classification, confidence, explanation = await loop.run_in_executor(
            executor, _run_analysis, request.audioBase64
        )
        
        return AnalyzeResponse(
            language=request.language,
            classification=classification,
            confidenceScore=round(confidence, 2),
            explanation=explanation
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"status": "error", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"status": "error", "message": "Invalid request"}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
