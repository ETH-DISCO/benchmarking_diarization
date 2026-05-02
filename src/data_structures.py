"""
Shared dataclasses for the diarization pipeline.

CRITICAL: These are the standardized formats that all components use.
All code must convert to/from these formats.

When adding a new model: convert model output → DiarizationResponse
When adding a new dataset: return AudioDataBatch and GroundTruthAnnotation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


# ============================================================================
# AUDIO DATA STRUCTURES
# ============================================================================

@dataclass

class AudioDataItem:
    """
    Single audio file ready for processing.
    
    This is the standardized format returned by DataLoader.load_audio().
    All models receive AudioDataItem instances and must convert to their own format internally.
    
    Attributes:
        audio_id (str): Unique identifier (e.g., "eng_0001", "aepyx", "EN2001a")
        audio_path (str): Absolute path to audio file
        split (str): Split/subset name (e.g., "eng" for CallHome, "dev" for VoxConverse, "dev" for AMI)
        language (str): Language code (e.g., "eng", "deu", "jpn", "spa", "zho")
                        Even monolingual datasets have this (e.g., "eng" for VoxConverse/AMI/ALI)
        duration (float): Audio duration in seconds (pre-computed for efficiency)
        dataset (str): Dataset name (e.g., "callhome", "voxconverse", "ami", "ali")
    """
    audio_id: str
    audio_path: str
    split: str
    language: str
    duration: float
    dataset: str


@dataclass
class AudioDataBatch:
    """
    Collection of audio items from a single dataset, organized by split.
    
    Preserves the dataset's inherent structure (language for CallHome, split for others).
    
    Attributes:
        dataset (str): Dataset name (e.g., "callhome", "voxconverse", "ami", "ali")
        items_by_split (Dict[str, List[AudioDataItem]]): Items organized hierarchically
                                                         (e.g., {"eng": [...], "deu": [...]})
        num_files (int): Total number of files (auto-computed)
        total_duration_seconds (float): Sum of all durations (auto-computed)
        splits (Dict[str, int]): Count per split (auto-computed)
        languages (List[str]): Unique languages (auto-computed)
    """
    dataset: str
    items_by_split: Dict[str, List[AudioDataItem]]
    num_files: int = field(init=False)
    total_duration_seconds: float = field(init=False)
    splits: Dict[str, int] = field(init=False)
    languages: List[str] = field(init=False)
    
    def __post_init__(self):
        """Auto-compute statistics."""
        self.num_files = sum(len(items) for items in self.items_by_split.values())
        self.total_duration_seconds = sum(
            item.duration 
            for items in self.items_by_split.values() 
            for item in items
        )
        
        # Count per split
        self.splits = {split: len(items) for split, items in self.items_by_split.items()}
        
        # Get unique languages
        all_items = [item for items in self.items_by_split.values() for item in items]
        self.languages = list(set(item.language for item in all_items))
    
    @property
    def items(self) -> List[AudioDataItem]:
        """Flatten items for convenience when structure doesn't matter."""
        return [item for items in self.items_by_split.values() for item in items]
# ============================================================================
# GROUND TRUTH DATA STRUCTURES
# ============================================================================

@dataclass
class GroundTruthSegment:
    """
    Single speaker segment from ground truth.
    
    Represents a continuous speech segment from one speaker.
    This is the building block of ground truth annotations.
    
    Attributes:
        start (float): Start time in seconds
        end (float): End time in seconds
        speaker_id (str): Speaker identifier (e.g., "0", "1", "spk_123")
    """
    start: float
    end: float
    speaker_id: str

@dataclass
class GroundTruthAnnotation:
    """
    Ground truth diarization for a single audio file.
    
    This is the standardized format returned by DataLoader.load_groundtruth().
    Contains all speaker segments and metadata for one audio file.
    
    Attributes:
        audio_id (str): Unique identifier (e.g., "eng_0001", "aepyx")
        dataset (str): Dataset name (e.g., "callhome", "voxconverse", "ami", "ali")
        split (str): Split/subset name (e.g., "eng" for CallHome, "dev" for VoxConverse)
        language (str): Language code (e.g., "eng", "deu", "jpn", "spa", "zho")
        segments (List[GroundTruthSegment]): All speaker segments in chronological order
        num_speakers (int): Number of unique speakers (auto-computed)
        duration (float): Total audio duration in seconds (auto-computed from max segment end)
    """
    audio_id: str
    dataset: str
    split: str
    language: str
    segments: List[GroundTruthSegment]
    num_speakers: int = field(init=False)
    duration: float = field(init=False)
    
    def __post_init__(self):
        """Auto-compute statistics after initialization."""
        # Count unique speakers
        unique_speakers = set(seg.speaker_id for seg in self.segments)
        self.num_speakers = len(unique_speakers)
        
        # Duration is the end time of the last segment
        if self.segments:
            self.duration = max(seg.end for seg in self.segments)
        else:
            self.duration = 0.0

# ============================================================================
# PREDICTION DATA STRUCTURES
# ============================================================================

@dataclass
class DiarizationSegment:
    """
    Single speaker segment from model prediction.
    
    Represents a continuous speech segment predicted by a diarization model.
    This is the building block of diarization predictions.
    
    Attributes:
        start (float): Start time in seconds
        end (float): End time in seconds
        speaker_id (str): Predicted speaker identifier (e.g., "00", "01", "spk_0")
    """
    start: float
    end: float
    speaker_id: str


@dataclass
class DiarizationResponse:
    """
    Model prediction response (standard format for all models).
    
    This is the standardized format returned by all model processors.
    All models (pyannote, nemo, diarizen, pyannoteai) must convert their output to this format.
    
    Includes both prediction data and source metadata for full traceability.
    
    Attributes:
        audio_id (str): Unique identifier (matches AudioDataItem.audio_id)
        dataset (str): Dataset name (e.g., "callhome", "voxconverse", "ami", "ali")
        split (str): Split/subset name (e.g., "eng" for CallHome, "dev" for VoxConverse)
        language (str): Language code (e.g., "eng", "deu", "jpn", "spa", "zho")
        model_used (str): Model name/path used (e.g., "pyannote/speaker-diarization-3.1")
        prompt_used (str): Prompt/instruction used (empty string for non-generative models)
        segments (List[DiarizationSegment]): All predicted speaker segments in chronological order
        timestamp (str): ISO format timestamp of when prediction was made
    """
    audio_id: str
    dataset: str
    split: str
    language: str
    model_used: str
    segments: List[DiarizationSegment]
    timestamp: str
    
    def to_dict(self) -> Dict:
        """
        Convert response to dictionary format for JSON serialization.
        
        Returns:
            Dictionary with all fields, segments converted to dicts
        """
        return {
            "audio_id": self.audio_id,
            "dataset": self.dataset,
            "split": self.split,
            "language": self.language,
            "model_used": self.model_used,
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "speaker_id": seg.speaker_id
                }
                for seg in self.segments
            ],
            "timestamp": self.timestamp
        }


# ============================================================================
# STATISTICS AND METRICS
# ============================================================================

@dataclass
class ProcessingStats:
    """
    Statistics from generation run.
    
    Tracks overall performance metrics and summary statistics for a complete generation run.
    Used to summarize how many files were processed, timing information, and efficiency metrics.
    
    Attributes:
        model (str): Model name used (e.g., "pyannote", "nemo", "diarizen", "pyannoteai")
        dataset (str): Dataset name processed (e.g., "callhome", "voxconverse", "ami", "ali")
        total_files (int): Total files attempted
        processed_files (int): Files successfully processed
        failed_files (int): Files that failed
        total_audio_duration (float): Sum of all audio durations in seconds
        total_processing_time (float): Total time spent processing in seconds
        start_time (str): ISO format timestamp when processing started
        end_time (str): ISO format timestamp when processing finished
    """
    model: str
    dataset: str
    total_files: int
    processed_files: int
    failed_files: int
    total_audio_duration: float
    total_processing_time: float
    start_time: str
    end_time: str
    
    @property
    def success_rate(self) -> float:
        """
        Calculate success rate as percentage.
        
        Returns:
            Percentage of successfully processed files (0-100)
        """
        if self.total_files > 0:
            return (self.processed_files / self.total_files) * 100
        return 0.0
    
    @property
    def real_time_factor(self) -> float:
        """
        Calculate Real-Time Factor (RTF).
        
        RTF = processing_time / audio_duration
        RTF < 1.0 means faster than real-time
        RTF > 1.0 means slower than real-time
        
        Returns:
            Real-time factor (processing_time / audio_duration)
        """
        if self.total_audio_duration > 0:
            return self.total_processing_time / self.total_audio_duration
        return 0.0
    
    @property
    def processing_speed(self) -> float:
        """
        Calculate processing speed multiplier (inverse of RTF).
        
        speed = audio_duration / processing_time
        speed > 1.0 means faster than real-time
        
        Returns:
            Processing speed multiplier
        """
        if self.total_processing_time > 0:
            return self.total_audio_duration / self.total_processing_time
        return 0.0
    
    def to_dict(self) -> Dict:
        """
        Convert stats to dictionary format for JSON serialization.
        
        Returns:
            Dictionary with all fields and computed metrics
        """
        return {
            "model": self.model,
            "dataset": self.dataset,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "total_audio_duration_seconds": round(self.total_audio_duration, 2),
            "total_audio_duration_hours": round(self.total_audio_duration / 3600, 2),
            "total_processing_time_seconds": round(self.total_processing_time, 2),
            "total_processing_time_hours": round(self.total_processing_time / 3600, 2),
            "real_time_factor": round(self.real_time_factor, 3),
            "processing_speed_xRT": round(self.processing_speed, 2),
            "success_rate_percent": round(self.success_rate, 2),
            "start_time": self.start_time,
            "end_time": self.end_time,
        }