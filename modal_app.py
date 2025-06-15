import modal
import os
import io
import base64
import json
import uuid
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Header, BackgroundTasks, Response
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import asyncio
from functools import lru_cache
from torch.nn.attention import sdpa_kernel
# Handle imports that are only needed in Modal environment
import aiofiles
import torch
from joblib import Parallel, delayed
import gc
import spacy
from concurrent.futures import ThreadPoolExecutor
import psutil
from typing import Dict, Any

# API Configuration
API_KEY = "789"
CLEANUP_INTERVAL = 3600  # 1 hour in seconds
FILE_RETENTION_HOURS = 24

def validate_api_key(api_key: str) -> bool:
    return api_key == API_KEY

# Event management functions
async def save_event(event_id: str, event: dict):
    try:
        event_path = f"/events/{event_id}.json"
        async with aiofiles.open(event_path, "w") as f:
            await f.write(json.dumps(event, indent=2))
        events_volume.commit()
        print(f"📁 Saved and committed event {event_id}")
    except Exception as e:
        print(f"❌ Error saving event {event_id}: {e}")

async def load_event(event_id: str) -> Optional[dict]:
    try:
        events_volume.reload()
        async with aiofiles.open(f"/events/{event_id}.json", "r") as f:
            return json.loads(await f.read())
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"❌ Error loading event {event_id}: {e}")
        return None

async def update_event(event_id: str, updates: dict):
    event = await load_event(event_id)
    if event:
        event.update(updates)
        if updates.get("status") in ["completed", "failed"]:
            event["completed_at"] = datetime.utcnow().isoformat()
        await save_event(event_id, event)
        print(f"🔄 Updated event {event_id}: {updates}")
    else:
        print(f"⚠️ Event {event_id} not found for update")

# Modal image with FIXED dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        "ffmpeg", 
        "libsndfile1", 
        "espeak-ng",  # For text preprocessing
        "sox",        # Audio processing utilities
        "libsox-fmt-all"
    ])
    .pip_install([
        "aiofiles==23.2.0",  # Move aiofiles to the top
        "chatterbox-tts==0.1.1",
        "fastapi==0.104.1",
        "pydantic==2.5.0",
        "requests==2.31.0",
        # Text processing optimizations
        "nltk==3.8.1",           # Advanced text tokenization
        "spacy==3.7.2",          # NLP for smart text splitting
        # Audio processing optimizations - FIXED VERSIONS
        "librosa==0.10.0",       # Match chatterbox-tts requirement
        "soundfile==0.12.1",     # Efficient audio I/O
        "pydub==0.25.1",         # Audio manipulation
        "resampy==0.4.3",        # Match chatterbox-tts requirement
        # Performance optimizations
        "numba==0.58.1",         # JIT compilation for speed
        "joblib==1.3.2",         # Parallel processing utilities
        "psutil==5.9.6",         # System resource monitoring
        # Memory optimizations
        "memory-profiler==0.61.0", # Memory usage tracking
        "pympler==0.9",          # Memory analysis
        "uvicorn",
        "python-multipart",
        "torch",
        "torchaudio",
        "numpy",
        "scipy",
        "python-dotenv",
        "tqdm",
        "transformers",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "einops",
        "xformers",
        "bitsandbytes",
        "safetensors",
        "huggingface_hub",
        "gradio",
        "pydub",
        "pytest",
        "pytest-asyncio",
        "httpx",
        "websockets",
        "python-jose[cryptography]",
        "passlib[bcrypt]",
        "email-validator"
    ])
    .run_commands([
        # Download spaCy model for text processing
        "python -m spacy download en_core_web_sm",
        # Download NLTK data
        "python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords')\"",
    ])
)

# Create Modal app
app = modal.App("vclar-api-v2", image=image)

# Persistent volumes
models_volume = modal.Volume.from_name("vclar-models", create_if_missing=True)
audio_volume = modal.Volume.from_name("vclar-audio-files", create_if_missing=True)
events_volume = modal.Volume.from_name("vclar-events", create_if_missing=True)

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.start_times = {}
        self.metrics = {}
    
    def start_monitoring(self, key: str):
        self.start_times[key] = time.time()
    
    def end_monitoring(self, key: str) -> dict:
        if key in self.start_times:
            duration = time.time() - self.start_times[key]
            memory_used = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            self.metrics[key] = {
                'duration': duration,
                'memory_used': memory_used
            }
            return self.metrics[key]
        return {}

# Memory optimization
def optimize_memory_usage():
    """Optimize memory usage during processing"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Smart text chunking
@lru_cache(maxsize=128)
def smart_text_chunking(text: str, max_length: int = 250) -> List[str]:
    """Intelligent text chunking based on semantic boundaries"""
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        
        chunks = []
        current_chunk = ""
        
        for sent in doc.sents:
            if len(current_chunk + sent.text) <= max_length:
                current_chunk += " " + sent.text if current_chunk else sent.text
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sent.text
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    except Exception as e:
        print(f"⚠️ Smart chunking failed: {e}, using simple split")
        return [text[i:i + max_length] for i in range(0, len(text), max_length)]

# Dynamic batch sizing
def dynamic_batch_size(text_length: int) -> int:
    """Dynamically adjust batch size based on text length"""
    if text_length < 1000:
        return 1
    elif text_length < 5000:
        return 5
    else:
        return 10

# Audio processing optimization
def optimize_audio_processing(audio_segments: List[torch.Tensor]) -> torch.Tensor:
    """Optimize audio processing with efficient concatenation"""
    try:
        # Convert to numpy for faster processing
        np_segments = [seg.cpu().numpy() for seg in audio_segments]
        
        # Use efficient numpy concatenation
        combined = np.concatenate(np_segments, axis=-1)
        
        # Convert back to torch tensor
        return torch.from_numpy(combined)
    except Exception as e:
        print(f"⚠️ Audio optimization failed: {e}, using simple concatenation")
        return torch.cat(audio_segments, dim=-1)

# Early stopping check
def early_stopping_check(progress: float, threshold: float = 0.95) -> bool:
    """Implement early stopping for processing"""
    return progress >= threshold

# Parallel processing
def parallel_process_chunks(chunks: List[str], process_func, n_jobs: int = -1) -> List[Any]:
    """Process chunks in parallel using all available CPU cores"""
    return Parallel(n_jobs=n_jobs)(
        delayed(process_func)(chunk, i) for i, chunk in enumerate(chunks)
    )

# Progressive loading
async def progressive_loading(chunks: List[str], process_func, batch_size: int = 5) -> List[Any]:
    """Load and process chunks progressively"""
    all_results = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_jobs = []
        
        # Create jobs for the batch
        for idx, chunk in enumerate(batch):
            job = process_func.spawn(chunk, i + idx)
            batch_jobs.append(job)
        
        # Wait for all jobs in the batch to complete
        batch_results = []
        for job in batch_jobs:
            try:
                result = await job.get(timeout=180)  # 3-minute timeout per chunk
                if result['success']:
                    batch_results.append(result)
                else:
                    print(f"❌ Chunk processing failed: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"❌ Chunk processing error: {str(e)}")
        
        all_results.extend(batch_results)
        
        # Early stopping check
        if early_stopping_check(len(all_results) / len(chunks)):
            print(f"🎯 Early stopping triggered at {len(all_results)}/{len(chunks)} chunks")
            break
    
    return all_results

# Optimized text processing utilities
class OptimizedTextProcessor:
    def __init__(self):
        self.nlp = None
        self.sentence_tokenizer = None
        self._initialize_processors()
    
    def _initialize_processors(self):
        """Initialize NLP processors with caching"""
        try:
            import spacy
            import nltk
            from nltk.tokenize import sent_tokenize
            
            # Load spaCy model (cached after first load)
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # Disable unnecessary components
            self.sentence_tokenizer = sent_tokenize
            print("📚 Initialized optimized text processors")
        except Exception as e:
            print(f"⚠️ Text processors not available: {e}")
            self.nlp = None
            self.sentence_tokenizer = None
    
    async def smart_split_text(self, text: str, max_length: int, target_chunks: int = None):
        """Optimized text splitting with semantic awareness"""
        if len(text) <= max_length:
            return [text]
        
        try:
            if self.sentence_tokenizer:
                # Use NLTK for better sentence detection
                sentences = self.sentence_tokenizer(text)
            else:
                # Fallback to regex
                import re
                sentences = re.split(r'(?<=[.!?])\s+', text)
            
            # Group sentences into semantically coherent chunks
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk + sentence) <= max_length:
                    current_chunk += " " + sentence if current_chunk else sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        # Sentence too long, split by phrases
                        chunks.extend(self._split_long_sentence(sentence, max_length))
                        current_chunk = ""
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
            
        except Exception as e:
            print(f"⚠️ Advanced text splitting failed: {e}, using simple split")
            return self._simple_split(text, max_length)
    
    def _split_long_sentence(self, sentence: str, max_length: int):
        """Split long sentences at natural break points"""
        import re
        # Split by commas, semicolons, and other natural breaks
        phrases = re.split(r'(?<=[,;:])\s+', sentence)
        
        chunks = []
        current_chunk = ""
        
        for phrase in phrases:
            if len(current_chunk + phrase) <= max_length:
                current_chunk += " " + phrase if current_chunk else phrase
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if len(phrase) > max_length:
                    # Force split by words
                    words = phrase.split()
                    word_chunk = ""
                    for word in words:
                        if len(word_chunk + " " + word) <= max_length:
                            word_chunk += " " + word if word_chunk else word
                        else:
                            if word_chunk:
                                chunks.append(word_chunk.strip())
                            word_chunk = word
                    if word_chunk:
                        current_chunk = word_chunk
                    else:
                        current_chunk = ""
                else:
                    current_chunk = phrase
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _simple_split(self, text: str, max_length: int):
        """Fallback simple text splitting"""
        chunks = []
        for i in range(0, len(text), max_length):
            chunks.append(text[i:i + max_length])
        return chunks

# Audio processing utilities
class OptimizedAudioProcessor:
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self._setup_audio_processing()
    
    def _setup_audio_processing(self):
        """Initialize audio processing tools"""
        try:
            import librosa
            import soundfile as sf
            from pydub import AudioSegment
            self.librosa = librosa
            self.sf = sf
            self.AudioSegment = AudioSegment
            print("🎵 Initialized optimized audio processors")
        except Exception as e:
            print(f"⚠️ Audio processors not available: {e}")
    
    async def optimize_audio_concatenation(self, audio_segments, silence_duration=0.15):
        """Efficiently concatenate audio with optimized silence"""
        import torch
        
        if not audio_segments:
            return None
        
        try:
            # Use librosa for better audio processing if available
            if hasattr(self, 'librosa'):
                # Convert to numpy for librosa processing
                np_segments = []
                for segment in audio_segments:
                    if isinstance(segment, torch.Tensor):
                        np_segment = segment.cpu().numpy()
                        if np_segment.ndim > 1:
                            np_segment = np_segment[0]  # Take first channel
                        np_segments.append(np_segment)
                
                # Add optimized silence between segments
                silence_samples = int(silence_duration * self.sample_rate)
                silence = np.zeros(silence_samples)
                
                final_segments = []
                for i, segment in enumerate(np_segments):
                    final_segments.append(segment)
                    if i < len(np_segments) - 1:  # Not the last segment
                        final_segments.append(silence)
                
                # Concatenate efficiently
                final_audio = np.concatenate(final_segments)
                
                # Convert back to torch tensor
                return torch.from_numpy(final_audio).unsqueeze(0)
            
        except Exception as e:
            print(f"⚠️ Advanced audio concatenation failed: {e}, using simple method")
        
        # Fallback to simple concatenation
        return torch.cat(audio_segments, dim=-1)
    
    def preprocess_reference_audio(self, audio_path: str):
        """Optimize reference audio for better voice cloning"""
        try:
            if hasattr(self, 'librosa'):
                # Load with librosa for better quality
                audio, sr = self.librosa.load(audio_path, sr=self.sample_rate)
                
                # Apply audio enhancement
                audio = self.librosa.effects.preemphasis(audio)  # Enhance speech clarity
                audio = self.librosa.util.normalize(audio)       # Normalize volume
                
                # Save optimized version
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    self.sf.write(f.name, audio, self.sample_rate)
                    return f.name
            
        except Exception as e:
            print(f"⚠️ Audio preprocessing failed: {e}")
        
        return audio_path  # Return original if preprocessing fails

# Pydantic models
class TTSRequest(BaseModel):
    text: str
    reference_audio_url: Optional[str] = None
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    webhook_url: Optional[str] = None

class TTSResponse(BaseModel):
    success: bool
    event_id: str
    message: str
    estimated_completion: str

class EventStatusResponse(BaseModel):
    event_id: str
    status: str
    progress: int
    created_at: str
    completed_at: Optional[str] = None
    audio_url: Optional[str] = None
    download_url: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

# Optimized chunk processor
@app.function(
    image=image,
    volumes={
        "/cache": models_volume,
        "/audio_files": audio_volume,
        "/events": events_volume
    },
    gpu="A10G",
    timeout=300,
    max_containers=10,
    enable_memory_snapshot=True

)


async def process_single_chunk(chunk_text: str, chunk_index: int, event_id: str, audio_params: Dict[str, Any]):
    """Process a single text chunk with optimizations"""
    monitor = PerformanceMonitor()
    monitor.start_monitoring(f"chunk_{chunk_index}")
    
    try:
        print(f"🎤 [Chunk {chunk_index}] Starting optimized processing...")
        
        # Initialize model with caching
        os.environ["HF_HOME"] = "/cache"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        
        import torch
        from chatterbox.tts import ChatterboxTTS
        
        # Memory optimization
        optimize_memory_usage()
        
        model = ChatterboxTTS.from_pretrained(device="cuda")
        
        # Process reference audio if provided (only for first chunk)
        audio_processor = OptimizedAudioProcessor(model.sr)
        reference_audio_path = None
        
        if audio_params.get('reference_audio_path') and chunk_index == 0:
            reference_audio_path = audio_processor.preprocess_reference_audio(
                audio_params['reference_audio_path']
            )
        
        # Generate audio with optimizations
        with torch.cuda.amp.autocast():
            wav = model.generate(
                chunk_text,
                audio_prompt_path=reference_audio_path,
                exaggeration=audio_params.get('exaggeration', 0.5),
                cfg_weight=audio_params.get('cfg_weight', 0.5)
            )
        
        # Save chunk audio
        chunk_filename = f"{event_id}_chunk_{chunk_index:03d}.wav"
        chunk_filepath = f"/audio_files/{chunk_filename}"
        
        import torchaudio as ta
        
        # Use optimized audio saving
        if hasattr(audio_processor, 'sf'):
            audio_np = wav.cpu().numpy()
            if audio_np.ndim > 1:
                audio_np = audio_np[0]
            audio_processor.sf.write(chunk_filepath, audio_np, model.sr)
        else:
            ta.save(chunk_filepath, wav, model.sr)
        
        # Clean up reference audio if it was preprocessed
        if reference_audio_path and reference_audio_path != audio_params.get('reference_audio_path'):
            try:
                os.unlink(reference_audio_path)
            except:
                pass
        
        # Get performance stats
        stats = monitor.end_monitoring(f"chunk_{chunk_index}")
        
        print(f"✅ [Chunk {chunk_index}] Completed: {wav.shape[-1]} samples ({stats.get('duration', 0):.2f}s)")
        
        return {
            "success": True,
            "chunk_index": chunk_index,
            "chunk_filename": chunk_filename,
            "samples": wav.shape[-1],
            "duration": wav.shape[-1] / model.sr,
            "processing_time": stats.get('duration', 0),
            "memory_used": stats.get('memory_used', 0),
            "audio": wav
        }
        
    except Exception as e:
        stats = monitor.end_monitoring(f"chunk_{chunk_index}")
        print(f"❌ [Chunk {chunk_index}] Failed: {str(e)}")
        return {
            "success": False,
            "chunk_index": chunk_index,
            "error": str(e),
            "processing_time": stats.get('duration', 0)
        }

# Background TTS processor - (keeping the rest of the functions as they were)
@app.function(
    image=image,
    volumes={
        "/cache": models_volume,
        "/audio_files": audio_volume,
        "/events": events_volume
    },
    gpu="A10G",
    timeout=900
)
async def process_tts_job(event_id: str, request_data: Dict[str, Any]):
    """Enhanced background function with optimizations"""
    start_time = time.time()
    all_completed_chunks = []
    
    
    try:
        print(f"🎤 Starting optimized TTS processing for event {event_id}")
        
        # Initialize optimized components
        os.environ["HF_HOME"] = "/cache"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        
        import torch
        import tempfile
        import requests
        from chatterbox.tts import ChatterboxTTS
        
        # GPU optimization
        optimize_memory_usage()
        print(f"🧹 Optimized GPU memory management")
        
        print(f"🤖 Loading Chatterbox model...")
        model = ChatterboxTTS.from_pretrained(device="cuda")
        print(f"✅ Model loaded successfully")
        
        # Ensure directories exist
        os.makedirs("/audio_files", exist_ok=True)
        os.makedirs("/events", exist_ok=True)
        
        # Initialize processors
        text_processor = OptimizedTextProcessor()
        audio_processor = OptimizedAudioProcessor()
        monitor = PerformanceMonitor()
        
        monitor.start_monitoring("text_processing")
        
        input_text = request_data['text'].strip()
        if not input_text:
            raise ValueError("Text cannot be empty")
        
        # Optimized text processing with smart chunking
        MAX_CHUNK_LENGTH = 1000
        MIN_CHUNK_COUNT = 3
        
        if len(input_text) > MAX_CHUNK_LENGTH:
            print(f"📝 Long text detected ({len(input_text)} chars), using optimized chunking...")
            
            # Use smart chunking with caching
            text_chunks = smart_text_chunking(input_text, MAX_CHUNK_LENGTH)
            
            # Determine optimal batch size
            batch_size = dynamic_batch_size(len(input_text))
            processing_method = "parallel_chunks" if len(text_chunks) >= MIN_CHUNK_COUNT else "sequential_chunks"
            
            text_stats = monitor.end_monitoring("text_processing")
            print(f"📝 Text processed in {text_stats.get('duration', 0):.2f}s: {len(text_chunks)} chunks")
            
            await update_event(event_id, {
                "progress": 25,
                "metadata": {
                    "original_text_length": len(input_text),
                    "chunked": True,
                    "chunks_total": len(text_chunks),
                    "processing_method": processing_method,
                    "max_chunk_length": MAX_CHUNK_LENGTH,
                    "text_processing_time": text_stats.get('duration', 0)
                }
            })
        else:
            text_chunks = [input_text]
            processing_method = "single_chunk"
            text_stats = monitor.end_monitoring("text_processing")
            print(f"📝 Single text processed: {len(input_text)} characters")
            await update_event(event_id, {"progress": 25})
        
        print(f"📝 Total chunks to process: {len(text_chunks)}")
        
        # Handle reference audio
        audio_prompt_path = None
        if request_data.get('reference_audio_url'):
            print(f"📥 Downloading and optimizing reference audio...")
            try:
                response = requests.get(request_data['reference_audio_url'], timeout=30)
                response.raise_for_status()
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(response.content)
                    audio_prompt_path = f.name
                
                audio_prompt_path = audio_processor.preprocess_reference_audio(audio_prompt_path)
                
                try:
                    import torchaudio as ta
                    audio_test, sr_test = ta.load(audio_prompt_path)
                    print(f"📊 Optimized reference audio: {audio_test.shape}, sample_rate: {sr_test}")
                except Exception as audio_error:
                    print(f"⚠️ Invalid reference audio after optimization: {audio_error}")
                    if os.path.exists(audio_prompt_path):
                        os.unlink(audio_prompt_path)
                    audio_prompt_path = None
                
                await update_event(event_id, {"progress": 35})
                
            except Exception as download_error:
                print(f"⚠️ Failed to download/optimize reference audio: {download_error}")
                audio_prompt_path = None
                await update_event(event_id, {"progress": 30})
        else:
            await update_event(event_id, {"progress": 30})
        
        # Generate speech
        print(f"🎵 Generating speech for {len(text_chunks)} chunk(s)...")
        await update_event(event_id, {
            "status": "processing", 
            "progress": 50
        })
        
        if len(text_chunks) == 1:
            # Single chunk processing
            chunk_text = text_chunks[0]
            print(f"🎵 Processing single chunk directly...")
            
            try:
                optimize_memory_usage()
                
                wav = model.generate(
                    chunk_text,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=float(request_data.get('exaggeration', 0.5)),
                    cfg_weight=float(request_data.get('cfg_weight', 0.5))
                )
                
                if wav is None or wav.numel() == 0:
                    raise RuntimeError("Generated audio is empty")
                    
                print(f"✅ Single chunk completed: {wav.shape[-1]} samples")
                
            except Exception as single_error:
                print(f"❌ Single chunk failed: {str(single_error)}")
                try:
                    optimize_memory_usage()
                    short_text = chunk_text[:300] + "." if len(chunk_text) > 300 else chunk_text
                    wav = model.generate(short_text, audio_prompt_path=None, exaggeration=0.5, cfg_weight=0.5)
                    print(f"✅ Fallback successful")
                except Exception as fallback_error:
                    raise RuntimeError(f"Single chunk processing failed: {str(single_error)} | Fallback: {str(fallback_error)}")
        
        else:
            # Multiple chunks with optimized processing
            print(f"🚀 Starting optimized parallel processing for {len(text_chunks)} chunks...")
            
            audio_params = {
                'reference_audio_path': audio_prompt_path,
                'exaggeration': float(request_data.get('exaggeration', 0.5)),
                'cfg_weight': float(request_data.get('cfg_weight', 0.5))
            }
            
            # Process chunks in batches
            for i in range(0, len(text_chunks), batch_size):
                batch = text_chunks[i:i + batch_size]
                batch_jobs = []
                
                # Create jobs for the batch
                for idx, chunk in enumerate(batch):
                    job = process_single_chunk.spawn(chunk, i + idx, event_id, audio_params)
                    batch_jobs.append(job)
                
                # Wait for all jobs in the batch to complete
                for job in batch_jobs:
                    try:
                        result = await job.get(timeout=180)  # 3-minute timeout per chunk
                        if result['success']:
                            all_completed_chunks.append(result)
                        else:
                            print(f"❌ Chunk processing failed: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        print(f"❌ Chunk processing error: {str(e)}")
                
                # Update progress
                progress = 50 + (40 * (i + len(batch)) / len(text_chunks))
                await update_event(event_id, {"progress": int(progress)})
            
            if not all_completed_chunks:
                raise RuntimeError(f"All {len(text_chunks)} chunks failed to process")
            
            print(f"🎉 Parallel processing completed: {len(all_completed_chunks)}/{len(text_chunks)} chunks successful")
            
            all_completed_chunks.sort(key=lambda x: x['chunk_index'])
            
            # Concatenate audio with optimization
            print(f"🔗 Concatenating {len(all_completed_chunks)} audio chunks with optimization...")
            
            audio_segments = [chunk['audio'] for chunk in all_completed_chunks if chunk['success']]
            
            if audio_segments:
                wav = optimize_audio_processing(audio_segments)
                print(f"🎵 Final optimized audio: {wav.shape[-1]} samples ({wav.shape[-1] / model.sr:.2f}s)")
            else:
                raise RuntimeError("No audio segments were successfully loaded")
        
        await update_event(event_id, {"progress": 80})
        
        # Save audio file
        print(f"💾 Saving audio file...")
        audio_filename = f"{event_id}.wav"
        audio_filepath = f"/audio_files/{audio_filename}"
        
        try:
            import torchaudio as ta
            ta.save(audio_filepath, wav, model.sr)
            
            file_size = os.path.getsize(audio_filepath)
            if file_size == 0:
                raise RuntimeError("Saved audio file is empty")
            
            print(f"💾 Audio file saved: {audio_filename} ({file_size} bytes)")
            audio_volume.commit()
            print(f"💾 Audio file committed to volume")
            
        except Exception as save_error:
            print(f"❌ Error saving audio file: {save_error}")
            raise RuntimeError(f"Failed to save audio file: {save_error}")
        
        # Clean up
        if audio_prompt_path and os.path.exists(audio_prompt_path):
            os.unlink(audio_prompt_path)
        
        # Enhanced metadata with performance stats
        processing_stats = {
            "total_chunks": len(text_chunks),
            "successful_chunks": len(all_completed_chunks) if len(text_chunks) > 1 else 1,
            "failed_chunks": len(text_chunks) - len(all_completed_chunks) if len(text_chunks) > 1 else 0,
            "average_chunk_time": sum(chunk.get('processing_time', 0) for chunk in all_completed_chunks) / len(all_completed_chunks) if all_completed_chunks else 0,
            "total_processing_time": sum(chunk.get('processing_time', 0) for chunk in all_completed_chunks) if all_completed_chunks else 0,
            "memory_efficiency": sum(chunk.get('memory_used', 0) for chunk in all_completed_chunks) if all_completed_chunks else 0
        }
        
        final_metadata = {
            "duration": wav.shape[-1] / model.sr,
            "sample_rate": model.sr,
            "text_length": len(request_data['text']),
            "chunks_processed": processing_stats["successful_chunks"],
            "processing_method": processing_method,
            "optimization_enabled": True,
            "performance_stats": processing_stats
        }
        
        # Add chunking details if applicable
        if len(text_chunks) > 1:
            final_metadata.update({
                "chunks_total": len(text_chunks),
                "chunk_lengths": [len(chunk) for chunk in text_chunks],
                "average_chunk_length": sum(len(chunk) for chunk in text_chunks) / len(text_chunks),
                "processing_efficiency": processing_stats["successful_chunks"] / len(text_chunks) * 100
            })
        
        await update_event(event_id, {
            "status": "completed",
            "progress": 100,
            "audio_file": audio_filename,
            "metadata": final_metadata
        })
        
        total_time = time.time() - start_time
        print(f"✅ Event {event_id} completed successfully in {total_time:.2f}s!")
        print(f"📊 Performance: {processing_stats['successful_chunks']}/{processing_stats['total_chunks']} chunks, {processing_stats['average_chunk_time']:.2f}s avg/chunk")
        
    except Exception as e:
        # Clear CUDA cache on error
        optimize_memory_usage()
        
        # Update event as failed
        try:
            await update_event(event_id, {
                "status": "failed",
                "error": str(e)
            })
        except Exception as save_error:
            print(f"❌ Error updating failed event: {save_error}")
        
        # Clean up temp file if it exists
        if 'audio_prompt_path' in locals() and audio_prompt_path and os.path.exists(audio_prompt_path):
            os.unlink(audio_prompt_path)
        
        print(f"❌ Event {event_id} failed: {str(e)}")
        raise

# Cleanup task
@app.function(
    image=image,
    volumes={
        "/audio_files": audio_volume,
        "/events": events_volume
    },
    schedule=modal.Cron("0 */1 * * *")
)
def cleanup_old_files():
    """Cleanup old audio files and events"""
    print("🧹 Starting cleanup of old files...")
    
    events_dir = "/events"
    audio_dir = "/audio_files"
    
    if os.path.exists(events_dir):
        cutoff_time = datetime.utcnow() - timedelta(hours=FILE_RETENTION_HOURS)
        
        for filename in os.listdir(events_dir):
            if filename.endswith('.json'):
                filepath = f"{events_dir}/{filename}"
                try:
                    with open(filepath, 'r') as f:
                        event = json.load(f)
                    
                    created_at = datetime.fromisoformat(event['created_at'])
                    if created_at < cutoff_time:
                        os.remove(filepath)
                        
                        if event.get('audio_file'):
                            audio_path = f"{audio_dir}/{event['audio_file']}"
                            if os.path.exists(audio_path):
                                os.remove(audio_path)
                        
                        print(f"🗑️ Cleaned up old event: {event['event_id']}")
                        
                except Exception as e:
                    print(f"Error cleaning event {filename}: {e}")
    
    print("✅ Cleanup completed!")

# Create FastAPI app
@app.function(
    image=image,
    volumes={
        "/audio_files": audio_volume,
        "/events": events_volume
    }
)
@modal.asgi_app(label="api")
def create_app():
    web_app = FastAPI(
        title="vClar API",
        description="Text-to-Speech API with optimizations",
        version="2.0.0"
    )
    
    def require_auth(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
        if not x_api_key or not validate_api_key(x_api_key):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key"
            )
    
    @web_app.post("/tts", response_model=TTSResponse)
    async def create_tts_job(
        request_data: TTSRequest,
        x_api_key: Optional[str] = Header(None, alias="X-API-Key")
    ):
        require_auth(x_api_key)
        
        event_id = str(uuid.uuid4())
        created_at = datetime.utcnow()
        
        # Create initial event file
        event = {
            "event_id": event_id,
            "status": "pending",
            "progress": 0,
            "created_at": created_at.isoformat(),
            "request_data": request_data.dict()
        }
        
        await save_event(event_id, event)
        print(f"📁 Created and committed initial event file for {event_id}")
        
        # Start background processing
       # process_tts_job.spawn(event_id, request_data.dict())
        process_tts_job.spawn(event_id, request_data.model_dump())
        return TTSResponse(
            success=True,
            event_id=event_id,
            message="TTS job created successfully",
            estimated_completion=(created_at + timedelta(minutes=5)).isoformat()
        )
    
    @web_app.get("/status/{event_id}", response_model=EventStatusResponse)
    async def get_event_status(
        event_id: str,
        x_api_key: Optional[str] = Header(None, alias="X-API-Key")
    ):
        require_auth(x_api_key)
        
        event = await load_event(event_id)
        if not event:
            raise HTTPException(
                status_code=404,
                detail="Event not found"
            )
        
        print(f"📊 Status check for {event_id}: {event['status']}")
        
        return EventStatusResponse(
            event_id=event_id,
            status=event["status"],
            progress=event.get("progress", 0),
            created_at=event["created_at"],
            completed_at=event.get("completed_at"),
            audio_url=event.get("audio_url"),
            download_url=event.get("download_url"),
            error=event.get("error"),
            metadata=event.get("metadata")
        )
    
    @web_app.get("/download/{event_id}")
    async def download_audio(event_id: str):
        """Download generated audio file"""
        try:
            # Reload volume to ensure we have latest files
            audio_volume.reload()
            
            # Check if event exists and is completed
            events_volume.reload()
            async with aiofiles.open(f"/events/{event_id}.json", "r") as f:
                event = json.loads(await f.read())
            
            if event["status"] != "completed":
                raise HTTPException(status_code=400, detail="Audio not ready for download")
            
            audio_filename = event.get("audio_file")
            if not audio_filename:
                raise HTTPException(status_code=404, detail="Audio file not found")
            
            audio_path = f"/audio_files/{audio_filename}"
            if not os.path.exists(audio_path):
                raise HTTPException(status_code=404, detail="Audio file not found")
            
            # Return the file using FileResponse
            return FileResponse(
                path=audio_path,
                filename=audio_filename,
                media_type="audio/wav",
                headers={
                    "Cache-Control": "no-cache, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
            
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Audio file not found")
        except Exception as e:
            print(f"❌ Error downloading audio: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to download audio")
    
    @web_app.get("/health")
    async def health_check(
        x_api_key: Optional[str] = Header(None, alias="X-API-Key")
    ):
        require_auth(x_api_key)
        return {"status": "healthy"}
    
    @web_app.get("/info")
    async def api_info(
        x_api_key: Optional[str] = Header(None, alias="X-API-Key")
    ):
        require_auth(x_api_key)
        return {
            "name": "vClar API",
            "version": "2.0.0",
            "description": "Text-to-Speech API with optimizations",
            "endpoints": [
                {
                    "path": "/tts",
                    "method": "POST",
                    "description": "Create a new TTS job"
                },
                {
                    "path": "/status/{event_id}",
                    "method": "GET",
                    "description": "Get the status of a TTS job"
                },
                {
                    "path": "/download/{event_id}",
                    "method": "GET",
                    "description": "Download the generated audio file"
                }
            ]
        }
    
    @web_app.get("/")
    async def root():
        return {
            "message": "Welcome to vClar API",
            "version": "2.0.0",
            "documentation": "/docs"
        }
    
    return web_app

# Test function
@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": models_volume}
)
def test_generation(text: str = "Hello, this is vClar API with optimizations!"):
    """Test function"""
    os.environ["HF_HOME"] = "/cache"
    
    from chatterbox.tts import ChatterboxTTS
    
    print(f"🎤 Testing optimized vClar API with: {text}")
    model = ChatterboxTTS.from_pretrained(device="cuda")
    wav = model.generate(text)
    
    print(f"✅ Generated {wav.shape[-1]} samples")
    return f"vClar API optimization test successful! Generated {wav.shape[-1]} audio samples"