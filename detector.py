
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
import hashlib
from functools import lru_cache

warnings.filterwarnings("ignore")


torch.set_num_threads(4)  


class VoiceDetector:
    MODEL_NAME = "MelodyMachine/Deepfake-audio-detection-V2"
    
    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.device = "cpu" 
        self._lock = threading.Lock()
        self._result_cache = {} 
        self._cache_max_size = 100
        self._load_model()
    
    def _load_model(self):
        print(f"Loading deepfake detection model (optimized for CPU)...")
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.MODEL_NAME,
                cache_dir=".model_cache"
            )
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.MODEL_NAME,
                cache_dir=".model_cache",
                torch_dtype=torch.float32  
            )
            self.model.eval()
            

            for param in self.model.parameters():
                param.requires_grad = False
            
            print("Model loaded and optimized!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self._load_backup_model()
    
    def _load_backup_model(self):
        try:
            backup_model = "mo-thecreator/Deepfake-audio-detection"
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                backup_model, cache_dir=".model_cache"
            )
            self.model = AutoModelForAudioClassification.from_pretrained(
                backup_model, cache_dir=".model_cache"
            )
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            print("Backup model loaded!")
        except Exception as e:
            raise RuntimeError(f"Could not load model: {e}")
    
    def _get_audio_hash(self, audio_data: str) -> str:
        return hashlib.md5(audio_data[:1000].encode()).hexdigest()
    
    def _decode_base64_audio(self, base64_audio: str) -> bytes:
        try:
            if ',' in base64_audio:
                base64_audio = base64_audio.split(',')[1]
            base64_audio = base64_audio.strip().replace('\n', '').replace('\r', '')
            return base64.b64decode(base64_audio)
        except Exception as e:
            raise ValueError(f"Invalid Base64 encoding: {str(e)}")
    
    def _load_audio_fast(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            y, sr = librosa.load(tmp_path, sr=16000, mono=True)
            
            if len(y) < sr * 0.5:
                raise ValueError("Audio too short (minimum 0.5 seconds)")
            
            max_samples = sr * 10
            if len(y) > max_samples:
                y = y[:max_samples]
            
            return y, sr
        except Exception as e:
            raise ValueError(f"Failed to process audio: {str(e)}")
        finally:
            os.unlink(tmp_path)
    
    def _get_explanation(self, classification: str, confidence: float) -> str:
        if classification == "AI_GENERATED":
            if confidence > 0.9:
                return "Strong synthetic voice markers detected"
            elif confidence > 0.7:
                return "Unnatural pitch patterns detected"
            return "Possible AI-generated voice"
        else:
            if confidence > 0.9:
                return "Natural human voice characteristics"
            elif confidence > 0.7:
                return "Natural speech patterns detected"
            return "Likely human voice"
    
    def analyze(self, base64_audio: str) -> Tuple[str, float, str]:
        # Check cache first
        audio_hash = self._get_audio_hash(base64_audio)
        if audio_hash in self._result_cache:
            return self._result_cache[audio_hash]
        
        # Decode and load audio
        audio_bytes = self._decode_base64_audio(base64_audio)
        audio_array, sample_rate = self._load_audio_fast(audio_bytes)
        
        # Extract features
        inputs = self.feature_extractor(
            audio_array, 
            sampling_rate=sample_rate, 
            return_tensors="pt",
            padding=True
        )
        
        # Thread-safe inference
        with self._lock:
            with torch.inference_mode():  
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
            
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Map to classification
        id2label = self.model.config.id2label
        label = id2label.get(predicted_class, str(predicted_class)).lower()
        
        if any(x in label for x in ["fake", "spoof", "ai", "synthetic", "deepfake"]):
            classification = "AI_GENERATED"
        elif any(x in label for x in ["real", "human", "genuine", "bonafide", "bona"]):
            classification = "HUMAN"
        else:
            classification = "AI_GENERATED" if predicted_class == 0 else "HUMAN"
        
        explanation = self._get_explanation(classification, confidence)
        result = (classification, float(confidence), explanation)
        
        # Cache result
        if len(self._result_cache) >= self._cache_max_size:
            self._result_cache.pop(next(iter(self._result_cache)))
        self._result_cache[audio_hash] = result
        
        return result



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
    get_detector()
