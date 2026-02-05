# Voice Detection API

A secure REST API that accepts Base64-encoded MP3 audio and classifies whether the voice is AI-generated or human. Supports Tamil, English, Hindi, Malayalam, and Telugu languages.

## Features

- 🎵 **Audio Analysis**: Accepts Base64-encoded MP3 audio files
- 🤖 **AI Detection**: Classifies voices as `AI_GENERATED` or `HUMAN`
- 🔐 **API Key Authentication**: Secure access via `x-api-key` header
- 📊 **Confidence Scores**: Returns classification confidence (0-1)
- 🌐 **Multi-language Support**: Tamil, English, Hindi, Malayalam, Telugu

## Quick Start

### 1. Setup Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Edit `.env` file to set your API key:

```env
API_KEY=your_secret_api_key_here
```

### 4. Run the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Documentation

Interactive docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "service": "Voice Detection API",
  "version": "1.0.0",
  "supported_languages": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
}
```

### Analyze Voice

```bash
POST /analyze
```

**Headers:**
```
x-api-key: YOUR_SECRET_API_KEY
Content-Type: application/json
```

**Request Body:**
```json
{
  "audio": "BASE64_ENCODED_MP3_DATA"
}
```

**Success Response (200):**
```json
{
  "result": "AI_GENERATED",
  "confidence": 0.87
}
```
or
```json
{
  "result": "HUMAN",
  "confidence": 0.92
}
```

**Error Responses:**
- `401 Unauthorized` - Missing or invalid API key
- `400 Bad Request` - Invalid Base64 or audio format
- `422 Unprocessable Entity` - Validation error

## Example Usage

### Python

```python
import requests
import base64

# Read and encode audio file
with open("sample.mp3", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()

# Make request
response = requests.post(
    "http://localhost:8000/analyze",
    json={"audio": audio_base64},
    headers={"x-api-key": "your_api_key"}
)

print(response.json())
# {'result': 'HUMAN', 'confidence': 0.92}
```

### cURL

```bash
# Encode audio file
AUDIO_BASE64=$(base64 -w 0 sample.mp3)

# Make request
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -H "x-api-key: your_api_key" \
  -d "{\"audio\": \"$AUDIO_BASE64\"}"
```

## Running Tests

```bash
python test_api.py
```

## Project Structure

```
voice_detection-api/
├── main.py           # FastAPI application
├── auth.py           # API key authentication
├── detector.py       # Voice detection logic
├── requirements.txt  # Python dependencies
├── test_api.py       # Test suite
├── .env              # Environment variables
├── .env.example      # Environment template
├── model/            # ML model files (auto-generated)
│   ├── voice_classifier.pkl
│   └── scaler.pkl
└── README.md         # This file
```

## Technical Details

### Feature Extraction

The API extracts the following audio features for classification:

- **MFCC** (Mel-frequency cepstral coefficients): 13 coefficients with mean and standard deviation
- **Spectral Centroid**: Center of mass of the spectrum
- **Spectral Rolloff**: Frequency below which 85% of the energy is concentrated
- **Zero Crossing Rate**: Rate of sign changes in the signal
- **RMS Energy**: Root mean square energy

### Classification

Uses a Random Forest classifier trained to distinguish between AI-generated and human voices based on subtle differences in audio characteristics.

## License

MIT License
