"""
Test script for Voice Detection API
"""
import requests
import base64
import os

# API Configuration
API_URL = "http://localhost:8000"
API_KEY = "hackathon_secret_key_2024"

def test_health():
    """Test health endpoint."""
    response = requests.get(f"{API_URL}/health")
    print(f"Health Check: {response.status_code}")
    print(f"Response: {response.json()}\n")
    return response.status_code == 200

def test_missing_api_key():
    """Test request without API key."""
    response = requests.post(
        f"{API_URL}/analyze",
        json={
            "language": "Tamil",
            "audioFormat": "mp3",
            "audioBase64": "a" * 200
        }
    )
    print(f"Missing API Key Test: {response.status_code}")
    result = response.json()
    print(f"Response: {result}\n")
    # Check for correct error format
    detail = result.get("detail", {})
    return response.status_code == 401 and detail.get("status") == "error"

def test_invalid_api_key():
    """Test request with invalid API key."""
    response = requests.post(
        f"{API_URL}/analyze",
        json={
            "language": "English",
            "audioFormat": "mp3",
            "audioBase64": "a" * 200
        },
        headers={"x-api-key": "invalid_key"}
    )
    print(f"Invalid API Key Test: {response.status_code}")
    result = response.json()
    print(f"Response: {result}\n")
    detail = result.get("detail", {})
    return response.status_code == 401 and detail.get("status") == "error"

def test_request_format():
    """Test that request format is validated correctly."""
    # Test missing language field
    response = requests.post(
        f"{API_URL}/analyze",
        json={
            "audioFormat": "mp3",
            "audioBase64": "a" * 200
        },
        headers={"x-api-key": API_KEY}
    )
    print(f"Missing Language Field Test: {response.status_code}")
    print(f"Response: {response.json()}\n")
    return response.status_code == 422

def test_with_audio_file(file_path: str):
    """Test with actual audio file."""
    if not os.path.exists(file_path):
        print(f"Audio file not found: {file_path}")
        return False
    
    # Read and encode audio file
    with open(file_path, 'rb') as f:
        audio_bytes = f.read()
    
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    response = requests.post(
        f"{API_URL}/analyze",
        json={
            "language": "English",
            "audioFormat": "mp3",
            "audioBase64": audio_base64
        },
        headers={"x-api-key": API_KEY}
    )
    
    print(f"Audio Analysis Test: {response.status_code}")
    result = response.json()
    print(f"Response: {result}\n")
    
    # Verify response format
    if response.status_code == 200:
        required_fields = ["status", "language", "classification", "confidenceScore", "explanation"]
        has_all_fields = all(field in result for field in required_fields)
        print(f"Has all required fields: {has_all_fields}")
        return has_all_fields
    return False

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Voice Detection API - Test Suite (Updated Format)")
    print("=" * 60 + "\n")
    
    print("Expected Request Format:")
    print('  {"language": "Tamil", "audioFormat": "mp3", "audioBase64": "..."}')
    print()
    print("Expected Response Format:")
    print('  {"status": "success", "language": "Tamil", "classification": "AI_GENERATED",')
    print('   "confidenceScore": 0.91, "explanation": "..."}')
    print()
    print("-" * 60 + "\n")
    
    tests = [
        ("Health Check", test_health),
        ("Missing API Key", test_missing_api_key),
        ("Invalid API Key", test_invalid_api_key),
        ("Request Format Validation", test_request_format),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"Error in {name}: {e}\n")
            results.append((name, False))
    
    print("=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for name, passed in results:
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"  {status}: {name}")
    
    print("\n" + "=" * 60)
    total_passed = sum(1 for _, p in results if p)
    print(f"Total: {total_passed}/{len(results)} tests passed")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
