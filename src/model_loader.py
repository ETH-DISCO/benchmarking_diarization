"""
ModelLoader - load diarization models with lazy imports.

CRITICAL: All model-specific imports happen INSIDE methods.
This allows gen.py to run in any conda environment without import errors.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and instantiate diarization models with lazy imports."""

    SUPPORTED_MODELS = ["pyannote", "nemo", "diarizen", "pyannoteai"]

    def __init__(self, model_name: str, config: Dict):
        """
        Args:
            model_name: Model to load ("pyannote", "nemo", "diarizen", "pyannoteai")
            config: Full config dict (from config.yaml)
        """
        self.model_name = model_name
        self.config = config

        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: '{self.model_name}'. "
                f"Supported: {self.SUPPORTED_MODELS}"
            )

    def load(self) -> Any:
        """
        Load and return the model instance.

        Returns:
            Loaded model ready for inference
        """
        logger.info(f"ModelLoader.load() called for: {self.model_name}")
        loaders = {
            "pyannote": self._load_pyannote,
            "nemo": self._load_nemo,
            "diarizen": self._load_diarizen,
            "pyannoteai": self._load_pyannoteai,
        }
        return loaders[self.model_name]()

    def _load_pyannote(self) -> Any:
        """
        Load pyannote.audio speaker diarization pipeline.

        Requires: pyannote.audio, torch
        Environment: pyannote_env
        """
        # TODO: verify model path and HF token
        try:
            logger.info("Importing pyannote.audio...")
            from pyannote.audio import Pipeline
            import torch
            logger.info("Imports OK")

            model_cfg = self.config.get("models", {}).get("pyannote", {})
            model_path = model_cfg.get("model_path", "pyannote/speaker-diarization-3.1")
            device = model_cfg.get("device", "cuda")

            # TODO: HF auth token — use environment variable in production
            import os
            hf_token = os.environ.get("HF_TOKEN", None)
            logger.info(f"HF_TOKEN: {'set' if hf_token else 'NOT SET'}")

            logger.info(f"Downloading/loading model: {model_path} ...")
            pipeline = Pipeline.from_pretrained(model_path, use_auth_token=hf_token)
            logger.info("Model loaded into memory")

            if device == "cuda" and torch.cuda.is_available():
                logger.info(f"Moving model to GPU (CUDA)...")
                pipeline = pipeline.to(torch.device("cuda"))
                logger.info(f"Pyannote ready on GPU: {model_path}")
            else:
                logger.info(f"Pyannote ready on CPU: {model_path}")

            return pipeline

        except ImportError as e:
            raise ImportError(
                f"pyannote.audio not available. Activate pyannote_env. Error: {e}"
            )

    def _load_nemo(self) -> Any:
        """
        Load NeMo Sortformer diarization model.

        Requires: nemo_toolkit, torch
        Environment: nemo_env
        """
        # TODO: verify model path — v1 vs v2 distinction
        try:
            logger.info("Importing NeMo toolkit...")
            from nemo.collections.asr.models import SortformerEncLabelModel
            import torch
            logger.info("Imports OK")

            model_cfg = self.config.get("models", {}).get("nemo", {})
            model_path = model_cfg.get("model_path", "nvidia/diar_streaming_sortformer_4spk-v2")
            device = model_cfg.get("device", "cuda")

            logger.info(f"Downloading/loading model: {model_path} ...")
            model = SortformerEncLabelModel.from_pretrained(model_path)
            model.eval()
            logger.info("Model loaded into memory")

            if device == "cuda" and torch.cuda.is_available():
                logger.info("Moving model to GPU (CUDA)...")
                model = model.to(torch.device("cuda"))
                logger.info(f"NeMo Sortformer ready on GPU: {model_path}")
            else:
                logger.info(f"NeMo Sortformer ready on CPU: {model_path}")

            return model

        except ImportError as e:
            raise ImportError(
                f"NeMo not available. Activate nemo_env. Error: {e}"
            )

    def _load_diarizen(self) -> Any:
        """
        Load DiariZen diarization pipeline.

        Requires: diarizen, torch
        Environment: diarizen_env
        """
        # TODO: verify model path on HuggingFace
        try:
            logger.info("Importing DiariZen...")
            from diarizen.pipelines.inference import DiariZenPipeline
            import torch
            logger.info("Imports OK")

            model_cfg = self.config.get("models", {}).get("diarizen", {})
            model_path = model_cfg.get(
                "model_path", "BUT-FIT/diarizen-wavlm-large-s80-md"
            )
            device = model_cfg.get("device", "cuda")

            logger.info(f"Downloading/loading model: {model_path} ...")
            pipeline = DiariZenPipeline.from_pretrained(model_path)
            logger.info("Model loaded into memory")

            if device == "cuda" and torch.cuda.is_available():
                logger.info("Moving model to GPU (CUDA)...")
                pipeline = pipeline.to(torch.device("cuda"))
                logger.info(f"DiariZen ready on GPU: {model_path}")
            else:
                logger.info(f"DiariZen ready on CPU: {model_path}")

            return pipeline

        except ImportError as e:
            raise ImportError(
                f"DiariZen not available. Activate diarizen_env. Error: {e}"
            )

    def _load_pyannoteai(self) -> Any:
        """
        Load PyannoteAI API client (no heavy ML dependencies).

        Returns a config dict with API settings (no model to load).
        The actual API calls happen in gen.py.

        Requires: requests (standard)
        Environment: any (API-based, no GPU)
        """
        # TODO: verify API endpoint and auth mechanism
        try:
            import os

            model_cfg = self.config.get("models", {}).get("pyannoteai", {})
            api_endpoint = model_cfg.get(
                "api_endpoint", "https://api.pyannote.ai"
            )
            api_key = os.environ.get("PYANNOTEAI_API_KEY", None)

            if api_key is None:
                raise ValueError(
                    "PYANNOTEAI_API_KEY environment variable not set. "
                    "Required for PyannoteAI API access."
                )

            client = {
                "api_endpoint": api_endpoint,
                "api_key": api_key,
            }
            logger.info(f"PyannoteAI client configured: {api_endpoint}")
            return client

        except Exception as e:
            raise RuntimeError(f"Failed to configure PyannoteAI client: {e}")