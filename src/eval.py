#!/usr/bin/env python3
"""
Unified diarization evaluation script.

Loads predictions and ground truth, computes DER/JER metrics,
generates per-file and aggregate reports.

Usage:
    python -m src.eval --model pyannote --dataset callhome
    python -m src.eval --model nemo --dataset ami
"""

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

from src.data_loader import DataLoader
from src.data_structures import GroundTruthAnnotation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def sanitize_model_dir(model_path: str) -> str:
    """Convert HuggingFace model path to filesystem-safe directory name."""
    return model_path.replace("/", "_")


def resolve_model_dir(model_key: str, config: Dict) -> str:
    """
    Resolve CLI model short name to sanitized filesystem directory.

    "nemo" + config → config["models"]["nemo"]["model_path"] → sanitize
    e.g. "nemo" → "nvidia/diar_sortformer_v2" → "nvidia_diar_sortformer_v2"
    """
    model_cfg = config.get("models", {}).get(model_key, {})
    model_path = model_cfg.get("model_path", model_key)
    return sanitize_model_dir(model_path)


# =============================================================================
# EVALUATION METRICS CONTAINER
# =============================================================================


@dataclass
class FileMetrics:
    """Evaluation metrics for a single audio file."""

    audio_id: str
    split: str
    der: float
    jer: float
    missed_speech: float
    false_alarm: float
    speaker_confusion: float
    correct: float
    collar: float
    gt_speakers: int
    pred_speakers: int
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        d = {
            "audio_id": self.audio_id,
            "split": self.split,
            "der": round(self.der, 4),
            "jer": round(self.jer, 4),
            "missed_speech": round(self.missed_speech, 4),
            "false_alarm": round(self.false_alarm, 4),
            "speaker_confusion": round(self.speaker_confusion, 4),
            "correct": round(self.correct, 4),
            "collar": self.collar,
            "gt_speakers": self.gt_speakers,
            "pred_speakers": self.pred_speakers,
        }
        if self.error:
            d["error"] = self.error
        return d


# =============================================================================
# CONVERSION: our data structures <-> pyannote objects
# =============================================================================


def gt_to_pyannote(gt: GroundTruthAnnotation) -> Annotation:
    """Convert GroundTruthAnnotation to pyannote Annotation."""
    annotation = Annotation()
    for seg in gt.segments:
        if seg.start >= seg.end:
            continue
        annotation[Segment(seg.start, seg.end)] = seg.speaker_id
    return annotation


def prediction_json_to_pyannote(prediction_data: Dict) -> Annotation:
    """
    Convert prediction JSON dict to pyannote Annotation.

    Expected format: {"segments": [{"start": ..., "end": ..., "speaker_id": ...}, ...]}
    """
    annotation = Annotation()
    segments = prediction_data.get("segments", [])

    for i, seg in enumerate(segments):
        try:
            start = float(seg["start"])
            end = float(seg["end"])
            speaker_id = str(seg["speaker_id"])

            if start >= end or start < 0:
                logger.warning(f"Invalid prediction segment {i}: [{start}, {end}], skipping")
                continue

            annotation[Segment(start, end)] = speaker_id
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Error parsing prediction segment {i}: {e}, skipping")

    return annotation


# =============================================================================
# PREDICTION FILE DISCOVERY
# =============================================================================


def find_prediction_files(
    model: str, dataset: str, results_base: str
) -> Dict[str, List[Dict]]:
    """
    Discover all prediction files for a model/dataset combination.

    Returns:
        Dict[split, List[{"audio_id": ..., "path": ..., "split": ...}]]
    """
    # TODO: verify this path convention matches your prediction output layout
    pred_base = Path(results_base) / model / dataset

    if not pred_base.exists():
        logger.warning(f"Prediction directory not found: {pred_base}")
        return {}

    files_by_split: Dict[str, List[Dict]] = {}

    for split_dir in sorted(pred_base.iterdir()):
        if not split_dir.is_dir():
            continue

        split_name = split_dir.name
        split_files: List[Dict] = []

        # Prediction files in the split directory
        for pred_file in sorted(split_dir.glob("*_prediction.json")):
            audio_id = pred_file.stem.replace("_prediction", "")
            split_files.append({
                "audio_id": audio_id,
                "path": str(pred_file),
                "split": split_name,
            })

        if split_files:
            files_by_split[split_name] = split_files
            logger.info(f"Found {len(split_files)} predictions in {dataset}/{split_name}")

    return files_by_split


# =============================================================================
# SINGLE FILE EVALUATION
# =============================================================================


def evaluate_file(
    audio_id: str,
    split: str,
    pred_path: str,
    data_loader: DataLoader,
    dataset_name: str,
    collar: float,
    model: str,
) -> FileMetrics:
    """
    Evaluate a single file: load GT + prediction, compute DER/JER.

    Args:
        audio_id: Audio identifier
        split: Split name
        pred_path: Path to prediction JSON
        data_loader: DataLoader instance
        dataset_name: Dataset name
        collar: Collar in seconds
        model: Model name

    Returns:
        FileMetrics with computed metrics
    """
    try:
        # Load ground truth
        gt_annotation_obj = data_loader.load_groundtruth(audio_id, dataset_name, split)
        gt_pyannote = gt_to_pyannote(gt_annotation_obj)

        if len(gt_pyannote) == 0:
            raise ValueError(f"Empty ground truth for {audio_id}")

        # Load prediction
        with open(pred_path, "r") as f:
            pred_data = json.load(f)

        pred_pyannote = prediction_json_to_pyannote(pred_data)

        if len(pred_pyannote) == 0:
            raise ValueError(f"Empty prediction for {audio_id}")

        # Compute DER
        der_metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
        der_components = der_metric(gt_pyannote, pred_pyannote, detailed=True)

        # Compute JER
        jer_metric = JaccardErrorRate(collar=collar)
        jer_value = jer_metric(gt_pyannote, pred_pyannote)

        return FileMetrics(
            audio_id=audio_id,
            split=split,
            der=der_components["diarization error rate"],
            jer=jer_value,
            missed_speech=der_components["missed detection"],
            false_alarm=der_components["false alarm"],
            speaker_confusion=der_components["confusion"],
            correct=der_components["correct"],
            collar=collar,
            gt_speakers=len(gt_pyannote.labels()),
            pred_speakers=len(pred_pyannote.labels()),
        )

    except Exception as e:
        logger.error(f"Evaluation failed for {audio_id}: {e}")
        return FileMetrics(
            audio_id=audio_id,
            split=split,
            der=0.0,
            jer=0.0,
            missed_speech=0.0,
            false_alarm=0.0,
            speaker_confusion=0.0,
            correct=0.0,
            collar=collar,
            gt_speakers=0,
            pred_speakers=0,
            error=str(e),
        )


# =============================================================================
# AGGREGATE EVALUATION
# =============================================================================


def evaluate_dataset(
    model: str,
    dataset_name: str,
    config: Dict,
) -> Dict[str, List[FileMetrics]]:
    """
    Evaluate all predictions for a model/dataset combination.

    Args:
        model: Sanitized model directory name (e.g. "nvidia_diar_sortformer_v2")
        dataset_name: Dataset name
        config: Full config dict

    Returns:
        Dict[split, List[FileMetrics]]
    """
    results_base = config.get("paths", {}).get("results_base", "predictions")
    collar = config.get("evaluation", {}).get("collar_seconds", 0.25)
    dataset_paths = config.get("datasets", {}).get("paths", {})

    # Initialize data loader (for GT access)
    data_loader = DataLoader(dataset_paths)

    # Discover prediction files
    pred_files_by_split = find_prediction_files(model, dataset_name, results_base)

    if not pred_files_by_split:
        logger.warning(f"No predictions found for {model}/{dataset_name}")
        return {}

    # Evaluate each file
    metrics_by_split: Dict[str, List[FileMetrics]] = {}

    for split, pred_files in pred_files_by_split.items():
        split_metrics: List[FileMetrics] = []
        logger.info(f"Evaluating {dataset_name}/{split}: {len(pred_files)} files")

        for pf in pred_files:
            metrics = evaluate_file(
                audio_id=pf["audio_id"],
                split=pf["split"],
                pred_path=pf["path"],
                data_loader=data_loader,
                dataset_name=dataset_name,
                collar=collar,
                model=model,
            )
            split_metrics.append(metrics)

        metrics_by_split[split] = split_metrics
        valid = [m for m in split_metrics if m.error is None]
        if valid:
            mean_der = np.mean([m.der for m in valid])
            mean_jer = np.mean([m.jer for m in valid])
            logger.info(
                f"  {split}: DER={mean_der:.3f}, JER={mean_jer:.3f} "
                f"({len(valid)}/{len(split_metrics)} valid)"
            )

    return metrics_by_split


# =============================================================================
# REPORT GENERATION
# =============================================================================


def save_per_file_metrics(
    metrics_by_split: Dict[str, List[FileMetrics]],
    model: str,
    dataset_name: str,
    results_dir: Path,
) -> List[Path]:
    """Save per-file metrics as JSON (one file per split)."""
    saved: List[Path] = []

    for split_name, metrics_list in metrics_by_split.items():
        data = {
            "model": model,
            "dataset": dataset_name,
            "split": split_name,
            "timestamp": datetime.now().isoformat(),
            "total_files": len(metrics_list),
            "files": [m.to_dict() for m in metrics_list],
        }

        json_file = results_dir / f"metrics_{split_name}.json"
        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)

        saved.append(json_file)
        logger.info(f"Saved per-file metrics: {json_file}")

    return saved


def generate_report(
    metrics_by_split: Dict[str, List[FileMetrics]],
    model: str,
    dataset_name: str,
    collar: float,
    results_dir: Path,
) -> Path:
    """Generate text analysis report with aggregate statistics."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = results_dir / f"analysis_report_{timestamp}.txt"

    # Collect all valid metrics
    all_valid: List[FileMetrics] = []
    all_failed = 0
    for metrics_list in metrics_by_split.values():
        all_valid.extend([m for m in metrics_list if m.error is None])
        all_failed += len([m for m in metrics_list if m.error is not None])

    lines = [
        "DIARIZATION EVALUATION REPORT",
        "=" * 50,
        f"Model:   {model}",
        f"Dataset: {dataset_name}",
        f"Collar:  {collar}s",
        f"Date:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "OVERALL",
        "-" * 30,
        f"Evaluated: {len(all_valid)}",
        f"Failed:    {all_failed}",
    ]

    if all_valid:
        der_vals = [m.der for m in all_valid]
        jer_vals = [m.jer for m in all_valid]
        ms_vals = [m.missed_speech for m in all_valid]
        fa_vals = [m.false_alarm for m in all_valid]
        sc_vals = [m.speaker_confusion for m in all_valid]

        lines.extend([
            "",
            f"DER:  mean={np.mean(der_vals):.4f}  median={np.median(der_vals):.4f}  "
            f"min={np.min(der_vals):.4f}  max={np.max(der_vals):.4f}  std={np.std(der_vals):.4f}",
            f"JER:  mean={np.mean(jer_vals):.4f}  median={np.median(jer_vals):.4f}  "
            f"min={np.min(jer_vals):.4f}  max={np.max(jer_vals):.4f}  std={np.std(jer_vals):.4f}",
            "",
            "DER COMPONENTS (mean):",
            f"  Missed speech:     {np.mean(ms_vals):.4f}",
            f"  False alarm:       {np.mean(fa_vals):.4f}",
            f"  Speaker confusion: {np.mean(sc_vals):.4f}",
        ])

    # Per-split breakdown
    for split_name, metrics_list in metrics_by_split.items():
        valid = [m for m in metrics_list if m.error is None]
        failed = [m for m in metrics_list if m.error is not None]

        lines.extend([
            "",
            f"SPLIT: {split_name.upper()}",
            "-" * 30,
            f"Evaluated: {len(valid)}, Failed: {len(failed)}",
        ])

        if valid:
            der_v = [m.der for m in valid]
            jer_v = [m.jer for m in valid]
            lines.extend([
                f"DER:  mean={np.mean(der_v):.4f}  median={np.median(der_v):.4f}  "
                f"min={np.min(der_v):.4f}  max={np.max(der_v):.4f}",
                f"JER:  mean={np.mean(jer_v):.4f}  median={np.median(jer_v):.4f}  "
                f"min={np.min(jer_v):.4f}  max={np.max(jer_v):.4f}",
            ])

    with open(report_file, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"Report saved: {report_file}")
    return report_file


def create_results_dir(model: str, dataset_name: str, eval_base: str) -> Path:
    """Create timestamped results directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(eval_base) / model / dataset_name / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Evaluate diarization predictions")
    parser.add_argument(
        "--model",
        required=True,
        choices=["pyannote", "nemo", "diarizen", "pyannoteai"],
        help="Model short name (matches config.yaml keys)",
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

    # Resolve CLI short name → sanitized HF model path for filesystem
    model_dir = resolve_model_dir(args.model, config)

    logger.info("=" * 60)
    logger.info("DIARIZATION EVALUATION PIPELINE")
    logger.info(f"Model:     {args.model}")
    logger.info(f"Model dir: {model_dir}")
    logger.info(f"Dataset:   {args.dataset}")
    logger.info("=" * 60)

    # Evaluate
    metrics_by_split = evaluate_dataset(model_dir, args.dataset, config)

    if not metrics_by_split:
        logger.error("No metrics computed. Check predictions exist.")
        return

    # Save results
    eval_base = config.get("paths", {}).get("evaluation_results", "evaluation_results")
    collar = config.get("evaluation", {}).get("collar_seconds", 0.25)
    results_dir = create_results_dir(model_dir, args.dataset, eval_base)

    save_per_file_metrics(metrics_by_split, model_dir, args.dataset, results_dir)
    generate_report(metrics_by_split, model_dir, args.dataset, collar, results_dir)

    # Final summary
    total = sum(len(ml) for ml in metrics_by_split.values())
    valid = sum(len([m for m in ml if m.error is None]) for ml in metrics_by_split.values())
    logger.info("=" * 60)
    logger.info(f"EVALUATION COMPLETE: {valid}/{total} files evaluated")
    logger.info(f"Results: {results_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()