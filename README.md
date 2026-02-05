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
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# .env file
API_KEY=your_secret_api_key
```

### 3. Run Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Usage

### Endpoint: `POST /analyze`

**Headers:**
```
x-api-key: your_api_key
Content-Type: application/json
```

**Request:**
```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "BASE64_ENCODED_MP3"
}
```

**Success Response:**
```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.91,
  "explanation": "Unnatural pitch consistency and robotic speech patterns detected"
}
```

**Error Response:**
```json
{
  "status": "error",
  "message": "Invalid API key or malformed request"
}
```

## Request Fields

| Field | Type | Description |
|-------|------|-------------|
| `language` | string | Tamil / English / Hindi / Malayalam / Telugu |
| `audioFormat` | string | Always `mp3` |
| `audioBase64` | string | Base64-encoded MP3 audio |

## Response Fields

| Field | Description |
|-------|-------------|
| `status` | `success` or `error` |
| `language` | Language of the audio |
| `classification` | `AI_GENERATED` or `HUMAN` |
| `confidenceScore` | Value between 0.0 and 1.0 |
| `explanation` | Short reason for decision |

## Testing with Postman

1. Set method to **POST**, URL: `http://localhost:8000/analyze`
2. Add header: `x-api-key: hackathon_secret_key_2024`
3. Body → raw → JSON:
```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "your_base64_audio"
}
```

### Convert MP3 to Base64 (PowerShell)
```powershell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("audio.mp3")) | Set-Clipboard
```

## Project Structure

```
voice_detection-api/
├── main.py          # FastAPI application
├── auth.py          # API key validation
├── detector.py      # Wav2Vec2 deepfake detection
├── requirements.txt # Dependencies
├── .env             # API key config
└── test_api.py      # Test suite
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

## Run Tests

```bash
python test_api.py
```

## API Docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## License

MIT License
