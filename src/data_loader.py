"""
DataLoader - unified dataset discovery and ground truth loading.

Single source of truth for all dataset operations.
Environment-agnostic: NO model-specific imports allowed.
Only standard library + librosa + json.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import librosa

from src.data_structures import (
    AudioDataItem,
    AudioDataBatch,
    GroundTruthSegment,
    GroundTruthAnnotation,
)

logger = logging.getLogger(__name__)


class DataLoader:
    """Load audio files and ground truth in standardized format."""

    # Dataset split definitions
    DATASET_SPLITS = {
        "callhome": ["eng", "deu", "jpn", "spa", "zho"],
        "voxconverse": ["dev", "test"],
        "ami": ["dev", "test"],
        "ali": ["test", "eval"],
    }

    # Language mapping per dataset (for monolingual datasets)
    DATASET_LANGUAGES = {
        "callhome": None,  # language = split
        "voxconverse": "eng",
        "ami": "eng",
        "ali": "zho",
    }

    def __init__(self, dataset_paths: Dict[str, str], results_base: str = "predictions"):
        """
        Args:
            dataset_paths: Mapping of dataset name to base path.
                           e.g. {"callhome": "dataset/callhome", ...}
            results_base: Base path for prediction outputs (from config).
        """
        self.dataset_paths = dataset_paths
        self.results_base = results_base

    # =========================================================================
    # AUDIO LOADING (used by gen.py)
    # =========================================================================

    def load_audio(
        self,
        dataset_name: str,
        model_dir: str = None,
        override: bool = False,
    ) -> AudioDataBatch:
        """
        Load all audio files for a dataset.

        When model_dir is provided and override is False, files with existing
        predictions are skipped before expensive audio validation / duration reads.

        Args:
            dataset_name: One of "callhome", "voxconverse", "ami", "ali"
            model_dir: Sanitized model directory name (e.g. "nvidia_diar_sortformer_v2").
                       When None, no skip filtering is applied (used by eval.py for GT only).
            override: If True, load all files regardless of existing predictions.

        Returns:
            AudioDataBatch with standardized AudioDataItem list
        """
        loaders = {
            "callhome": self._load_callhome_audio,
            "voxconverse": self._load_voxconverse_audio,
            "ami": self._load_ami_audio,
            "ali": self._load_ali_audio,
        }
        if dataset_name not in loaders:
            raise ValueError(
                f"Unsupported dataset: '{dataset_name}'. "
                f"Supported: {list(loaders.keys())}"
            )
        return loaders[dataset_name](model_dir=model_dir, override=override)

    def _load_callhome_audio(self, model_dir: str = None, override: bool = False) -> AudioDataBatch:
        """
        Load CallHome dataset: language-based splits.

        Structure: dataset/callhome/{lang}/{lang}_DDDD.wav
        """
        # TODO: verify this path resolves correctly on your cluster
        base_dir = Path(self.dataset_paths["callhome"])
        if not base_dir.exists():
            raise FileNotFoundError(f"CallHome directory not found: {base_dir}")

        items_by_split: Dict[str, List[AudioDataItem]] = {}

        for lang in self.DATASET_SPLITS["callhome"]:
            lang_dir = base_dir / lang
            if not lang_dir.exists():
                logger.warning(f"Language directory not found: {lang_dir}")
                continue

            lang_items: List[AudioDataItem] = []
            wav_files = sorted(lang_dir.glob(f"{lang}_*.wav"))

            for wav_file in wav_files:
                audio_id = wav_file.stem  # e.g. "eng_0001"

                # Validate naming: lang_DDDD
                parts = audio_id.split("_")
                if not (
                    len(parts) == 2
                    and parts[0] == lang
                    and parts[1].isdigit()
                    and len(parts[1]) == 4
                ):
                    logger.warning(f"Invalid CallHome naming, skipped: {wav_file}")
                    continue

                # Skip if prediction exists (before expensive librosa calls)
                if self._should_skip(audio_id, "callhome", lang, model_dir, override):
                    continue

                if not self._validate_audio(str(wav_file)):
                    continue

                duration = self._get_duration(str(wav_file))
                if duration is None:
                    continue

                lang_items.append(
                    AudioDataItem(
                        audio_id=audio_id,
                        audio_path=str(wav_file),
                        split=lang,
                        language=lang,
                        duration=duration,
                        dataset="callhome",
                    )
                )

            if lang_items:
                items_by_split[lang] = lang_items
                logger.info(f"CallHome/{lang}: {len(lang_items)} files")

        return AudioDataBatch(dataset="callhome", items_by_split=items_by_split)

    def _load_voxconverse_audio(self, model_dir: str = None, override: bool = False) -> AudioDataBatch:
        """
        Load VoxConverse dataset: dev/test splits, English only.

        Structure: dataset/voxconverse/{split}/eng/*.wav
        """
        # TODO: verify this path resolves correctly on your cluster
        base_dir = Path(self.dataset_paths["voxconverse"])
        if not base_dir.exists():
            raise FileNotFoundError(f"VoxConverse directory not found: {base_dir}")

        items_by_split: Dict[str, List[AudioDataItem]] = {}

        for split in self.DATASET_SPLITS["voxconverse"]:
            split_dir = base_dir / split / "eng"
            if not split_dir.exists():
                logger.warning(f"VoxConverse split not found: {split_dir}")
                continue

            split_items: List[AudioDataItem] = []
            wav_files = sorted(split_dir.glob("*.wav"))

            for wav_file in wav_files:
                audio_id = wav_file.stem

                if self._should_skip(audio_id, "voxconverse", split, model_dir, override):
                    continue

                if not self._validate_audio(str(wav_file)):
                    continue

                duration = self._get_duration(str(wav_file))
                if duration is None:
                    continue

                split_items.append(
                    AudioDataItem(
                        audio_id=audio_id,
                        audio_path=str(wav_file),
                        split=split,
                        language="eng",
                        duration=duration,
                        dataset="voxconverse",
                    )
                )

            if split_items:
                items_by_split[split] = split_items
                logger.info(f"VoxConverse/{split}: {len(split_items)} files")

        return AudioDataBatch(dataset="voxconverse", items_by_split=items_by_split)

    def _load_ami_audio(self, model_dir: str = None, override: bool = False) -> AudioDataBatch:
        """
        Load AMI dataset: dev/test splits.

        Structure: dataset/ami/{split}/audio/*.Mix-Headset.wav
        Audio ID extracted as meeting ID: EN2001a.Mix-Headset.wav -> EN2001a
        """
        # TODO: verify this path resolves correctly on your cluster
        base_dir = Path(self.dataset_paths["ami"])
        if not base_dir.exists():
            raise FileNotFoundError(f"AMI directory not found: {base_dir}")

        items_by_split: Dict[str, List[AudioDataItem]] = {}

        for split in self.DATASET_SPLITS["ami"]:
            audio_dir = base_dir / split / "audio"
            if not audio_dir.exists():
                logger.warning(f"AMI audio directory not found: {audio_dir}")
                continue

            split_items: List[AudioDataItem] = []
            wav_files = sorted(audio_dir.glob("*.wav"))

            for wav_file in wav_files:
                # EN2001a.Mix-Headset.wav -> EN2001a
                meeting_id = wav_file.stem.split(".")[0]

                if self._should_skip(meeting_id, "ami", split, model_dir, override):
                    continue

                if not self._validate_audio(str(wav_file)):
                    continue

                duration = self._get_duration(str(wav_file))
                if duration is None:
                    continue

                split_items.append(
                    AudioDataItem(
                        audio_id=meeting_id,
                        audio_path=str(wav_file),
                        split=split,
                        language="eng",
                        duration=duration,
                        dataset="ami",
                    )
                )

            if split_items:
                items_by_split[split] = split_items
                logger.info(f"AMI/{split}: {len(split_items)} files")

        return AudioDataBatch(dataset="ami", items_by_split=items_by_split)

    def _load_ali_audio(self, model_dir: str = None, override: bool = False) -> AudioDataBatch:
        """
        Load ALI (AliMeeting) dataset: test/eval splits.

        Structure: dataset/ali/{split}/audio/*.wav
        """
        # TODO: verify this path resolves correctly on your cluster
        base_dir = Path(self.dataset_paths["ali"])
        if not base_dir.exists():
            raise FileNotFoundError(f"ALI directory not found: {base_dir}")

        items_by_split: Dict[str, List[AudioDataItem]] = {}

        for split in self.DATASET_SPLITS["ali"]:
            audio_dir = base_dir / split / "audio"
            if not audio_dir.exists():
                logger.warning(f"ALI audio directory not found: {audio_dir}")
                continue

            split_items: List[AudioDataItem] = []
            wav_files = sorted(audio_dir.glob("*.wav"))

            for wav_file in wav_files:
                audio_id = wav_file.stem

                if self._should_skip(audio_id, "ali", split, model_dir, override):
                    continue

                if not self._validate_audio(str(wav_file)):
                    continue

                duration = self._get_duration(str(wav_file))
                if duration is None:
                    continue

                split_items.append(
                    AudioDataItem(
                        audio_id=audio_id,
                        audio_path=str(wav_file),
                        split=split,
                        language="zho",
                        duration=duration,
                        dataset="ali",
                    )
                )

            if split_items:
                items_by_split[split] = split_items
                logger.info(f"ALI/{split}: {len(split_items)} files")

        return AudioDataBatch(dataset="ali", items_by_split=items_by_split)

    # =========================================================================
    # GROUND TRUTH LOADING (used by eval.py)
    # =========================================================================

    def load_groundtruth(
        self, audio_id: str, dataset_name: str, split: str
    ) -> GroundTruthAnnotation:
        """
        Load ground truth for a single audio file.

        Args:
            audio_id: Audio identifier (e.g. "eng_0001", "aepyx", "EN2001a")
            dataset_name: Dataset name
            split: Split name (language code for CallHome, split for others)

        Returns:
            GroundTruthAnnotation with standardized segments
        """
        loaders = {
            "callhome": self._load_callhome_gt,
            "voxconverse": self._load_voxconverse_gt,
            "ami": self._load_ami_gt,
            "ali": self._load_ali_gt,
        }
        if dataset_name not in loaders:
            raise ValueError(f"Unsupported dataset: '{dataset_name}'")

        return loaders[dataset_name](audio_id, split)

    def _load_callhome_gt(self, audio_id: str, split: str) -> GroundTruthAnnotation:
        """
        Load CallHome GT from JSON metadata.

        Expected file: dataset/callhome/{lang}/{lang}_metadata.json
        The metadata contains arrays of timestamp_start, timestamp_end, speaker_id
        keyed by "{audio_id}.wav".
        """
        language = audio_id.split("_")[0]
        # TODO: verify this GT path matches your dataset layout
        gt_dir = Path(self.dataset_paths["callhome"]) / language
        metadata_file = gt_dir / f"{language}_metadata.json"

        if not metadata_file.exists():
            raise FileNotFoundError(f"CallHome metadata not found: {metadata_file}")

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        audio_key = f"{audio_id}.wav"
        if audio_key not in metadata:
            raise FileNotFoundError(
                f"Audio '{audio_key}' not found in CallHome metadata"
            )

        audio_data = metadata[audio_key]

        # Validate required fields
        for field in ["timestamp_start", "timestamp_end", "speaker_id"]:
            if field not in audio_data:
                raise ValueError(
                    f"Missing field '{field}' in CallHome metadata for {audio_key}"
                )

        starts = audio_data["timestamp_start"]
        ends = audio_data["timestamp_end"]
        speakers = audio_data["speaker_id"]

        if not (len(starts) == len(ends) == len(speakers)):
            raise ValueError(
                f"Mismatched array lengths in CallHome metadata for {audio_key}"
            )

        segments: List[GroundTruthSegment] = []
        for start, end, speaker in zip(starts, ends, speakers):
            if start >= end or start < 0 or end < 0:
                logger.warning(
                    f"Invalid segment for {audio_key}: [{start}, {end}], skipping"
                )
                continue
            segments.append(GroundTruthSegment(start=start, end=end, speaker_id=str(speaker)))

        return GroundTruthAnnotation(
            audio_id=audio_id,
            dataset="callhome",
            split=language,
            language=language,
            segments=segments,
        )

    def _load_voxconverse_gt(self, audio_id: str, split: str) -> GroundTruthAnnotation:
        """
        Load VoxConverse GT from RTTM file.

        Expected file: dataset/voxconverse/{split}/gt/{audio_id}.rttm
        RTTM format: SPEAKER file 1 start duration <NA> <NA> speaker_id <NA> <NA>
        """
        # TODO: verify this GT path matches your dataset layout
        rttm_path = Path(self.dataset_paths["voxconverse"]) / split / "gt" / f"{audio_id}.rttm"

        if not rttm_path.exists():
            raise FileNotFoundError(f"VoxConverse RTTM not found: {rttm_path}")

        segments: List[GroundTruthSegment] = []
        with open(rttm_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                # SPEAKER file 1 start duration <NA> <NA> speaker_id <NA> <NA>
                start_time = float(parts[3])
                duration = float(parts[4])
                end_time = start_time + duration
                speaker_id = parts[7]
                segments.append(
                    GroundTruthSegment(start=start_time, end=end_time, speaker_id=speaker_id)
                )

        return GroundTruthAnnotation(
            audio_id=audio_id,
            dataset="voxconverse",
            split=split,
            language="eng",
            segments=segments,
        )

    def _load_ami_gt(self, audio_id: str, split: str) -> GroundTruthAnnotation:
        """
        Load AMI GT from JSON file.

        Expected file: dataset/ami/{split}/gt/{audio_id}.json
        JSON format: list of {"start": float, "end": float, "speaker_id": str}
        """
        # TODO: verify this GT path matches your dataset layout
        gt_path = Path(self.dataset_paths["ami"]) / split / "gt" / f"{audio_id}.json"

        if not gt_path.exists():
            raise FileNotFoundError(f"AMI GT not found: {gt_path}")

        with open(gt_path, "r") as f:
            gt_data = json.load(f)

        segments: List[GroundTruthSegment] = []
        for seg in gt_data:
            start = float(seg["start"])
            end = float(seg["end"])
            speaker_id = seg["speaker_id"]
            segments.append(GroundTruthSegment(start=start, end=end, speaker_id=speaker_id))

        return GroundTruthAnnotation(
            audio_id=audio_id,
            dataset="ami",
            split=split,
            language="eng",
            segments=segments,
        )

    def _load_ali_gt(self, audio_id: str, split: str) -> GroundTruthAnnotation:
        """
        Load ALI GT from JSON file.

        Expected file: dataset/ali/{split}/gt/{gt_id}.json
        where gt_id = audio_id without the last underscore segment
              (e.g. "R8009_M8013_MS815" -> gt_id = "R8009_M8013")

        JSON format: {"segments": [{"start": ..., "end": ..., "speaker_id": ...}, ...]}
        """
        # TODO: verify this gt_id derivation is correct for your ALI layout
        gt_id = audio_id.rsplit("_", 1)[0]
        gt_path = Path(self.dataset_paths["ali"]) / split / "gt" / f"{gt_id}.json"

        if not gt_path.exists():
            raise FileNotFoundError(f"ALI GT not found: {gt_path}")

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        raw_segments = gt_data.get("segments", [])
        segments: List[GroundTruthSegment] = []
        for seg in raw_segments:
            start = float(seg["start"])
            end = float(seg["end"])
            if start >= end:
                logger.warning(f"Invalid ALI segment [{start}, {end}], skipping")
                continue
            segments.append(
                GroundTruthSegment(start=start, end=end, speaker_id=seg["speaker_id"])
            )

        return GroundTruthAnnotation(
            audio_id=audio_id,
            dataset="ali",
            split=split,
            language="zho",
            segments=segments,
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _should_skip(
        self, audio_id: str, dataset: str, split: str,
        model_dir: str, override: bool,
    ) -> bool:
        """
        Check if a file should be skipped (prediction already exists).

        Returns False (don't skip) when:
        - model_dir is None (no skip filtering, e.g. eval.py usage)
        - override is True (re-run everything)

        Returns True (skip) when prediction file exists on disk.
        """
        if model_dir is None or override:
            return False
        if self.prediction_exists(audio_id, dataset, split, model_dir, self.results_base):
            logger.debug(f"Skipping {audio_id} — prediction exists")
            return True
        return False

    def _validate_audio(self, path: str) -> bool:
        """Check that an audio file exists and has positive duration."""
        try:
            if not Path(path).exists():
                return False
            duration = librosa.get_duration(path=path)
            return duration > 0
        except Exception as e:
            logger.debug(f"Audio validation failed for {path}: {e}")
            return False

    def _get_duration(self, path: str) -> Optional[float]:
        """Get audio duration, returning None on failure."""
        try:
            return librosa.get_duration(path=path)
        except Exception as e:
            logger.warning(f"Could not get duration for {path}: {e}")
            return None

    @staticmethod
    def prediction_exists(
        audio_id: str, dataset: str, split: str, model: str, results_base: str
    ) -> bool:
        """Check if a prediction JSON already exists."""
        pred_file = Path(results_base) / model / dataset / split / f"{audio_id}_prediction.json"
        return pred_file.exists()

    # =========================================================================
    # FORMAT CONVERTERS
    # =========================================================================

    @staticmethod
    def to_nemo_manifest(items: List[AudioDataItem], output_path: str) -> str:
        """
        Write AudioDataItems as a NeMo JSONL manifest.

        Each line: {"audio_filepath": ..., "offset": 0, "duration": ..., "audio_id": ...}

        Args:
            items: List of AudioDataItem
            output_path: Where to write the manifest

        Returns:
            Path to created manifest file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for item in items:
                entry = {
                    "audio_filepath": item.audio_path,
                    "offset": 0,
                    "duration": item.duration,
                    "audio_id": item.audio_id,
                }
                f.write(json.dumps(entry) + "\n")

        logger.info(f"Created NeMo manifest: {len(items)} files -> {output_path}")
        return str(output_path)

    @staticmethod
    def items_by_split_to_flat_list(
        items_by_split: Dict[str, List[AudioDataItem]],
    ) -> List[AudioDataItem]:
        """Flatten items_by_split dict into a single list."""
        return [item for items in items_by_split.values() for item in items]