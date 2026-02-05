
import base64
import tempfile
import os
from typing import Tuple
import numpy as np
import torch
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import warnings
import threading

warnings.filterwarnings("ignore")


class VoiceDetector:
    MODEL_NAME = "MelodyMachine/Deepfake-audio-detection-V2"
    
    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lock = threading.Lock()  # Thread-safe inference
        self._load_model()
    
    def _load_model(self):
        print(f"Loading Wav2Vec2 deepfake detection model on {self.device}...")
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.MODEL_NAME,
                cache_dir=".model_cache"  # Cache locally for faster loading
            )
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.MODEL_NAME,
                cache_dir=".model_cache"
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Optimize for inference
            if self.device == "cpu":
                self.model = torch.jit.optimize_for_inference(
                    torch.jit.script(self.model)
                ) if hasattr(torch.jit, 'optimize_for_inference') else self.model
            
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to backup model...")
            self._load_backup_model()
    
    def _load_backup_model(self):
        try:
            backup_model = "mo-thecreator/Deepfake-audio-detection"
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                backup_model,
                cache_dir=".model_cache"
            )
            self.model = AutoModelForAudioClassification.from_pretrained(
                backup_model,
                cache_dir=".model_cache"
            )
            self.model.to(self.device)
            self.model.eval()
            print("Backup model loaded successfully!")
        except Exception as e:
            print(f"Backup model also failed: {e}")
            raise RuntimeError("Could not load any deepfake detection model")
    
    def _decode_base64_audio(self, base64_audio: str) -> bytes:
        try:
            if ',' in base64_audio:
                base64_audio = base64_audio.split(',')[1]
            base64_audio = base64_audio.strip().replace('\n', '').replace('\r', '')
            return base64.b64decode(base64_audio)
        except Exception as e:
            raise ValueError(f"Invalid Base64 encoding: {str(e)}")
    
    def _load_audio(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Load audio at 16kHz for Wav2Vec2
            y, sr = librosa.load(tmp_path, sr=16000, mono=True)
            
            if len(y) < sr * 0.5:
                raise ValueError("Audio too short for analysis (minimum 0.5 seconds required)")
            
            # Limit to 30 seconds for faster processing
            max_samples = sr * 30
            if len(y) > max_samples:
                y = y[:max_samples]
            
            return y, sr
        finally:
            os.unlink(tmp_path)
    
    def _get_explanation(self, classification: str, confidence: float) -> str:
        if classification == "AI_GENERATED":
            if confidence > 0.9:
                return "Strong synthetic voice markers detected with unnatural pitch patterns"
            elif confidence > 0.7:
                return "Unnatural pitch consistency and robotic speech patterns detected"
            else:
                return "Possible AI-generated voice with some synthetic characteristics"
        else:
            if confidence > 0.9:
                return "Natural speech patterns with authentic human voice characteristics"
            elif confidence > 0.7:
                return "Natural speech patterns and organic voice characteristics detected"
            else:
                return "Likely human voice with some natural speech variations"
    
    def analyze(self, base64_audio: str) -> Tuple[str, float, str]:
        # Decode audio
        audio_bytes = self._decode_base64_audio(base64_audio)
        
        # Load and preprocess audio
        audio_array, sample_rate = self._load_audio(audio_bytes)
        
        # Extract features
        inputs = self.feature_extractor(
            audio_array, 
            sampling_rate=sample_rate, 
            return_tensors="pt",
            padding=True
        )
        
        # Thread-safe model inference
        with self._lock:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        id2label = self.model.config.id2label
        label = id2label.get(predicted_class, str(predicted_class)).lower()
        
        if any(x in label for x in ["fake", "spoof", "ai", "synthetic", "deepfake"]):
            classification = "AI_GENERATED"
        elif any(x in label for x in ["real", "human", "genuine", "bonafide", "bona"]):
            classification = "HUMAN"
        else:
            classification = "AI_GENERATED" if predicted_class == 0 else "HUMAN"
        
        explanation = self._get_explanation(classification, confidence)
        
        return classification, float(confidence), explanation


# Singleton detector with eager loading
_detector = None
_detector_lock = threading.Lock()


def get_detector() -> VoiceDetector:
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = VoiceDetector()
    return _detector


def preload_model():
    """Preload model at startup for low latency."""
    get_detector()
