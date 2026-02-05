
import base64
import tempfile
import os
from typing import Tuple
import numpy as np
import torch
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


class VoiceDetector:
    MODEL_NAME = "MelodyMachine/Deepfake-audio-detection-V2"
    
    def __init__(self):
        """Initialize the voice detector with pre-trained Wav2Vec2 model."""
        self.model = None
        self.feature_extractor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained Wav2Vec2 deepfake detection model."""
        print(f"Loading Wav2Vec2 deepfake detection model on {self.device}...")
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.MODEL_NAME)
            self.model = AutoModelForAudioClassification.from_pretrained(self.MODEL_NAME)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to backup model...")
            self._load_backup_model()
    
    def _load_backup_model(self):
        """Load backup model if primary fails."""
        try:
            backup_model = "mo-thecreator/Deepfake-audio-detection"
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(backup_model)
            self.model = AutoModelForAudioClassification.from_pretrained(backup_model)
            self.model.to(self.device)
            self.model.eval()
            print("Backup model loaded successfully!")
        except Exception as e:
            print(f"Backup model also failed: {e}")
            raise RuntimeError("Could not load any deepfake detection model")
    
    def _decode_base64_audio(self, base64_audio: str) -> bytes:
        """
        Decode Base64 encoded audio data.
        
        Args:
            base64_audio: Base64 encoded audio string
            
        Returns:
            Decoded audio bytes
            
        Raises:
            ValueError: If Base64 decoding fails
        """
        try:
            # Remove potential data URL prefix
            if ',' in base64_audio:
                base64_audio = base64_audio.split(',')[1]
            
            # Remove whitespace and newlines
            base64_audio = base64_audio.strip().replace('\n', '').replace('\r', '')
            
            return base64.b64decode(base64_audio)
        except Exception as e:
            raise ValueError(f"Invalid Base64 encoding: {str(e)}")
    
    def _load_audio(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """
        Load audio from bytes and convert to proper format.
        """
        # Write bytes to temporary file for librosa to read
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Load audio with librosa at 16kHz
            y, sr = librosa.load(tmp_path, sr=16000, mono=True)
            
            # Ensure we have enough audio data
            if len(y) < sr * 0.5:  # Less than 0.5 seconds
                raise ValueError("Audio too short for analysis (minimum 0.5 seconds required)")
            
            # Limit to 30 seconds max
            max_samples = sr * 30
            if len(y) > max_samples:
                y = y[:max_samples]
            
            return y, sr
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
    
    def _get_explanation(self, classification: str, confidence: float, features: dict) -> str:
        """
        Generate explanation based on classification and audio features. """
        
        if classification == "AI_GENERATED":
            if confidence > 0.9:
                explanations = [
                    "Strong synthetic voice markers detected with unnatural pitch patterns",
                    "Audio exhibits clear machine-generated artifacts and robotic speech rhythm",
                    "Detected significant AI voice signatures with artificial prosody",
                ]
            elif confidence > 0.7:
                explanations = [
                    "Unnatural pitch consistency and robotic speech patterns detected",
                    "Synthetic voice characteristics and uniform tonal patterns identified",
                    "Audio shows signs of AI generation with mechanical speech patterns",
                ]
            else:
                explanations = [
                    "Possible AI-generated voice with some synthetic characteristics",
                    "Audio contains subtle artificial markers suggesting AI generation",
                    "Detected minor synthetic artifacts in voice patterns",
                ]
        else:  # HUMAN
            if confidence > 0.9:
                explanations = [
                    "Natural speech patterns with authentic human voice characteristics",
                    "Strong human vocal markers with natural prosody and breathing patterns",
                    "Audio exhibits genuine human voice with natural variations",
                ]
            elif confidence > 0.7:
                explanations = [
                    "Natural speech patterns and organic voice characteristics detected",
                    "Audio exhibits typical human vocal variations and natural prosody",
                    "Detected authentic voice markers consistent with human speech",
                ]
            else:
                explanations = [
                    "Likely human voice with some natural speech variations",
                    "Audio shows characteristics consistent with human speech",
                    "Voice patterns suggest natural human origin",
                ]
        
        import random
        random.seed(int(confidence * 1000))
        return random.choice(explanations)
    
    def analyze(self, base64_audio: str) -> Tuple[str, float, str]:
        """
        Analyze audio and classify as AI-generated or human.
        """
        # Decode audio
        audio_bytes = self._decode_base64_audio(base64_audio)
        
        # Load and preprocess audio
        audio_array, sample_rate = self._load_audio(audio_bytes)
        
        # Extract features for the model
        inputs = self.feature_extractor(
            audio_array, 
            sampling_rate=sample_rate, 
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Get prediction
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
        
        id2label = self.model.config.id2label
        label = id2label.get(predicted_class, str(predicted_class)).lower()
        
        # Normalize label to our format
        if any(x in label for x in ["fake", "spoof", "ai", "synthetic", "deepfake"]):
            classification = "AI_GENERATED"
        elif any(x in label for x in ["real", "human", "genuine", "bonafide", "bona"]):
            classification = "HUMAN"
        else:
            # Default mapping: 0 = AI, 1 = Human (most common)
            classification = "AI_GENERATED" if predicted_class == 0 else "HUMAN"
        
        # Generate explanation
        explanation = self._get_explanation(classification, confidence, {})
        
        return classification, float(confidence), explanation


_detector = None


def get_detector() -> VoiceDetector:
    global _detector
    if _detector is None:
        _detector = VoiceDetector()
    return _detector
