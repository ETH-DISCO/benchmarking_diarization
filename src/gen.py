#!/usr/bin/env python3
"""
Unified diarization generation script.

Loads audio via DataLoader, loads model via ModelLoader,
runs inference, saves predictions as JSON.

Usage:
    python -m src.gen --model pyannote --dataset callhome
    python -m src.gen --model nemo --dataset voxconverse --batch-size 1
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from src.data_loader import DataLoader
from src.data_structures import (
    AudioDataBatch,
    AudioDataItem,
    DiarizationResponse,
    DiarizationSegment,
    ProcessingStats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def sanitize_model_dir(model_path: str) -> str:
    """
    Convert a HuggingFace model path to a filesystem-safe directory name.

    "nvidia/diar_sortformer_v2" → "nvidia_diar_sortformer_v2"
    "pyannote/speaker-diarization-3.1" → "pyannote_speaker-diarization-3.1"

    This is the SINGLE source of truth for the model directory name.
    Used by: save_prediction, prediction_exists, save_stats, eval discovery.
    """
    return model_path.replace("/", "_")


# =============================================================================
# PREDICTION SAVING
# =============================================================================


def save_prediction(
    response: DiarizationResponse, results_base: str
) -> Path:
    """
    Save a DiarizationResponse as JSON.

    Output path: {results_base}/{model}/{dataset}/{split}/{audio_id}_prediction.json

    Args:
        response: Standardized prediction response
        results_base: Base directory for predictions

    Returns:
        Path to saved file, or empty Path if prediction is empty
    """
    output_dir = (
        Path(results_base)
        / sanitize_model_dir(response.model_used)
        / response.dataset
        / response.split
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{response.audio_id}_prediction.json"
    file_path = output_dir / filename

    if len(response.segments) == 0:
        logger.warning(f"Empty prediction for {response.audio_id}, not saved")
        return Path()

    data = response.to_dict()
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.debug(f"Saved prediction: {file_path}")
    return file_path


def save_stats(stats: ProcessingStats, results_base: str) -> Path:
    """Save processing statistics report."""
    output_dir = Path(results_base) / stats.model / stats.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stats_file = output_dir / f"processing_stats_{timestamp}.json"

    with open(stats_file, "w") as f:
        json.dump(stats.to_dict(), f, indent=2)

    logger.info(f"Stats saved: {stats_file}")
    return stats_file


# =============================================================================
# OUTPUT FORMAT CONVERTERS (model output -> DiarizationResponse)
# =============================================================================


def pyannote_annotation_to_response(
    annotation: Any, item: AudioDataItem, model_path: str
) -> DiarizationResponse:
    """
    Convert pyannote.core.Annotation to DiarizationResponse.

    Works for both pyannote.audio and DiariZen (both return pyannote Annotation).
    """
    segments: List[DiarizationSegment] = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        speaker_id = f"SPEAKER_{speaker:02d}" if isinstance(speaker, int) else str(speaker)
        segments.append(
            DiarizationSegment(
                start=round(segment.start, 3),
                end=round(segment.end, 3),
                speaker_id=speaker_id,
            )
        )
    segments.sort(key=lambda x: x.start)

    return DiarizationResponse(
        audio_id=item.audio_id,
        dataset=item.dataset,
        split=item.split,
        language=item.language,
        model_used=model_path,
        segments=segments,
        timestamp=datetime.now().isoformat(),
    )


def nemo_output_to_response(
    raw_output: List, item: AudioDataItem, model_path: str
) -> DiarizationResponse:
    """
    Convert NeMo Sortformer output to DiarizationResponse.

    NeMo returns: [['0.000 6.400 speaker_0', '6.960 7.760 speaker_0', ...]]
    """
    segments: List[DiarizationSegment] = []

    # Flatten nested list
    flat = raw_output[0] if raw_output and isinstance(raw_output[0], list) else raw_output

    for seg_str in flat:
        parts = seg_str.split()
        if len(parts) < 3:
            continue
        try:
            start = float(parts[0])
            end = float(parts[1])
            raw_speaker = parts[2]

            # Normalize speaker ID: speaker_0 -> SPEAKER_00
            if raw_speaker.startswith("speaker_"):
                num = raw_speaker.split("_")[1]
                speaker_id = f"SPEAKER_{num.zfill(2)}"
            else:
                speaker_id = raw_speaker

            segments.append(
                DiarizationSegment(
                    start=round(start, 3),
                    end=round(end, 3),
                    speaker_id=speaker_id,
                )
            )
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse NeMo segment '{seg_str}': {e}")

    segments.sort(key=lambda x: x.start)

    return DiarizationResponse(
        audio_id=item.audio_id,
        dataset=item.dataset,
        split=item.split,
        language=item.language,
        model_used=model_path,
        segments=segments,
        timestamp=datetime.now().isoformat(),
    )


# =============================================================================
# MODEL-SPECIFIC PROCESSING
# =============================================================================


def process_pyannote(
    model: Any,
    audio_batch: AudioDataBatch,
    config: Dict,
) -> ProcessingStats:
    """
    Process audio with pyannote.audio pipeline.

    Pyannote accepts an audio path directly and returns a pyannote Annotation.
    """
    model_cfg = config.get("models", {}).get("pyannote", {})
    model_path = model_cfg.get("model_path", "pyannote/speaker-diarization-3.1")
    model_dir = sanitize_model_dir(model_path)
    results_base = config.get("paths", {}).get("results_base", "predictions")

    start_time = time.time()
    processed = 0
    failed = 0
    total_audio = 0.0
    failed_ids: List[str] = []

    for item in audio_batch.items:
        try:
            # TODO: verify pyannote pipeline call interface
            annotation = model(item.audio_path)

            response = pyannote_annotation_to_response(annotation, item, model_path)
            save_prediction(response, results_base)

            processed += 1
            total_audio += item.duration
            logger.info(f"OK {item.audio_id} ({item.duration:.1f}s)")

        except Exception as e:
            logger.error(f"FAIL {item.audio_id}: {e}")
            failed += 1
            failed_ids.append(item.audio_id)

    elapsed = time.time() - start_time

    stats = ProcessingStats(
        model=model_dir,
        dataset=audio_batch.dataset,
        total_files=processed + failed,
        processed_files=processed,
        failed_files=failed,
        total_audio_duration=total_audio,
        total_processing_time=elapsed,
        start_time=datetime.fromtimestamp(start_time).isoformat(),
        end_time=datetime.now().isoformat(),
    )
    save_stats(stats, results_base)
    return stats


def process_diarizen(
    model: Any,
    audio_batch: AudioDataBatch,
    config: Dict,
) -> ProcessingStats:
    """
    Process audio with DiariZen pipeline.

    DiariZen also returns pyannote Annotation, same interface as pyannote.
    """
    model_cfg = config.get("models", {}).get("diarizen", {})
    model_path = model_cfg.get("model_path", "BUT-FIT/diarizen-wavlm-large-s80-md")
    model_dir = sanitize_model_dir(model_path)
    results_base = config.get("paths", {}).get("results_base", "predictions")

    start_time = time.time()
    processed = 0
    failed = 0
    total_audio = 0.0

    for item in audio_batch.items:
        try:
            # TODO: verify DiariZen pipeline call interface
            annotation = model(item.audio_path)

            response = pyannote_annotation_to_response(annotation, item, model_path)
            save_prediction(response, results_base)

            processed += 1
            total_audio += item.duration
            logger.info(f"OK {item.audio_id} ({item.duration:.1f}s)")

            # GPU memory cleanup
            _try_cuda_cleanup()

        except Exception as e:
            logger.error(f"FAIL {item.audio_id}: {e}")
            failed += 1

    elapsed = time.time() - start_time

    stats = ProcessingStats(
        model=model_dir,
        dataset=audio_batch.dataset,
        total_files=processed + failed,
        processed_files=processed,
        failed_files=failed,
        total_audio_duration=total_audio,
        total_processing_time=elapsed,
        start_time=datetime.fromtimestamp(start_time).isoformat(),
        end_time=datetime.now().isoformat(),
    )
    save_stats(stats, results_base)
    return stats


def process_nemo(
    model: Any,
    audio_batch: AudioDataBatch,
    config: Dict,
) -> ProcessingStats:
    """
    Process audio with NeMo Sortformer.

    NeMo uses model.diarize(audio=path) and returns list of segment strings.
    """
    model_cfg = config.get("models", {}).get("nemo", {})
    model_path = model_cfg.get("model_path", "nvidia/diar_streaming_sortformer_4spk-v2")
    model_dir = sanitize_model_dir(model_path)
    results_base = config.get("paths", {}).get("results_base", "predictions")
    batch_size = config.get("processing", {}).get("batch_size", 1)

    start_time = time.time()
    processed = 0
    failed = 0
    total_audio = 0.0

    for item in audio_batch.items:
        try:
            # TODO: verify NeMo diarize() call interface and batch_size param
            raw_output = model.diarize(audio=item.audio_path, batch_size=batch_size)

            response = nemo_output_to_response(raw_output, item, model_path)
            save_prediction(response, results_base)

            processed += 1
            total_audio += item.duration
            logger.info(f"OK {item.audio_id} ({item.duration:.1f}s)")

        except Exception as e:
            logger.error(f"FAIL {item.audio_id}: {e}")
            failed += 1

    elapsed = time.time() - start_time

    stats = ProcessingStats(
        model=model_dir,
        dataset=audio_batch.dataset,
        total_files=processed + failed,
        processed_files=processed,
        failed_files=failed,
        total_audio_duration=total_audio,
        total_processing_time=elapsed,
        start_time=datetime.fromtimestamp(start_time).isoformat(),
        end_time=datetime.now().isoformat(),
    )
    save_stats(stats, results_base)
    return stats


def process_pyannoteai(
    client: Dict,
    audio_batch: AudioDataBatch,
    config: Dict,
) -> ProcessingStats:
    """
    Process audio with PyannoteAI API.

    This is API-based: submit job, poll for results.
    """
    # TODO: implement full PyannoteAI API flow (submit, poll, parse)
    # The API documentation is in create_diar_request_-_pyannoteAI_documentation_
    # Key steps:
    # 1. Upload audio or provide URL
    # 2. POST /v1/diarize with {url, model: "precision-2"}
    # 3. Poll job status until "succeeded"
    # 4. Parse response segments
    raise NotImplementedError(
        "PyannoteAI processing not yet implemented. "
        "See create_diar_request_-_pyannoteAI_documentation_ for API spec."
    )


# =============================================================================
# UTILITIES
# =============================================================================


def _try_cuda_cleanup():
    """Attempt to free GPU memory. Silently passes if torch unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate diarization predictions")
    parser.add_argument(
        "--model",
        required=True,
        choices=["pyannote", "nemo", "diarizen", "pyannoteai"],
        help="Model name",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["callhome", "voxconverse", "ami", "ali"],
        help="Dataset name",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Resolve paths from config
    dataset_paths = config.get("datasets", {}).get("paths", {})
    results_base = config.get("paths", {}).get("results_base", "predictions")
    override = config.get("processing", {}).get("override_predictions", False)

    # Compute model_dir early (needed for skip-check during audio loading)
    model_cfg = config.get("models", {}).get(args.model, {})
    model_path = model_cfg.get("model_path", args.model)
    model_dir = sanitize_model_dir(model_path)

    logger.info("=" * 60)
    logger.info("DIARIZATION GENERATION PIPELINE")
    logger.info(f"Model:     {args.model}")
    logger.info(f"Model dir: {model_dir}")
    logger.info(f"Dataset:   {args.dataset}")
    logger.info(f"Override:  {override}")
    logger.info("=" * 60)

    # Initialize DataLoader
    data_loader = DataLoader(dataset_paths, results_base=results_base)

    # Load audio (skips files with existing predictions unless override=True)
    logger.info("Loading audio files...")
    audio_batch = data_loader.load_audio(args.dataset, model_dir=model_dir, override=override)
    logger.info(
        f"Loaded {audio_batch.num_files} files, "
        f"{audio_batch.total_duration_seconds / 3600:.2f}h total audio"
    )

    if audio_batch.num_files == 0:
        logger.warning("No files to process!")
        return

    # Load model (lazy imports happen here)
    from src.model_loader import ModelLoader

    logger.info(f"Loading model: {args.model}...")
    model_loader = ModelLoader(args.model, config)
    model = model_loader.load()

    # Dispatch to model-specific processing
    processors = {
        "pyannote": process_pyannote,
        "nemo": process_nemo,
        "diarizen": process_diarizen,
        "pyannoteai": process_pyannoteai,
    }

    logger.info("Starting inference...")
    stats = processors[args.model](model, audio_batch, config)

    # Summary
    logger.info("=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info(f"Processed: {stats.processed_files}/{stats.total_files}")
    logger.info(f"Failed:    {stats.failed_files}")
    logger.info(f"RTF:       {stats.real_time_factor:.3f}")
    logger.info(f"Speed:     {stats.processing_speed:.2f}x real-time")
    logger.info(f"Duration:  {stats.total_processing_time:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()