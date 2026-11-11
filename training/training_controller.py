"""
Echo — Training Controller and Atomic Adapter Versioning (Phase 1, Step 4)

Manages the end-to-end training lifecycle of personal LoRA adapters:
1. Validating input writing samples.
2. Creating uniquely named training datasets and manifests in the training-output area.
3. Resolving next adapter versions without overwriting existing adapters.
4. Training into version-isolated directories.
5. Coordinating with the generation lock to prevent CPU/RAM contention.
6. Validating adapter weights and configs before publishing.
7. Safely activating the newly trained adapter using the verified reload mechanism.
8. Enforcing single-job concurrency and recording detailed lifecycle status.

Lifecycle:
idle -> preparing -> training -> validating -> activating -> completed (or error)
"""

import contextlib
import json
import logging
import re
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

try:
    from train_lora_v2 import train_adapter, BASE_MODEL, EPOCHS, LEARNING_RATE
    from generate_with_adapter import (
        DEFAULT_BASE_MODEL,
        DEFAULT_ADAPTER_PATH,
        DEFAULT_METADATA_PATH,
        load_active_adapter_metadata,
        save_active_adapter_metadata,
        validate_adapter_directory,
        PersonalizedWritingGenerator,
    )
except ImportError:
    from training.train_lora_v2 import train_adapter, BASE_MODEL, EPOCHS, LEARNING_RATE
    from training.generate_with_adapter import (
        DEFAULT_BASE_MODEL,
        DEFAULT_ADAPTER_PATH,
        DEFAULT_METADATA_PATH,
        load_active_adapter_metadata,
        save_active_adapter_metadata,
        validate_adapter_directory,
        PersonalizedWritingGenerator,
    )

logger = logging.getLogger("EchoTrainingController")


class TrainingLifecycleState(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    TRAINING = "training"
    VALIDATING = "validating"
    ACTIVATING = "activating"
    COMPLETED = "completed"
    ERROR = "error"


DEFAULT_MIN_SAMPLES = 25
DEFAULT_ADAPTER_PREFIX = "qwen3.5-0.8b-lora"


def resolve_next_version(
    output_root: Union[Path, str] = "training-output",
    base_prefix: str = DEFAULT_ADAPTER_PREFIX,
    metadata_path: Optional[Path] = None,
) -> Tuple[str, Path]:
    """
    Scans the training-output directory to determine the next sequential adapter version (e.g. 'v3').
    Read-only inspection: does not create or reserve the directory on disk.
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    highest_version_num = 1

    # Check existing directories matching the prefix or version pattern
    for item in root.iterdir():
        if item.is_dir():
            match = re.search(r"\bv(\d+)\b", item.name, re.IGNORECASE)
            if match:
                highest_version_num = max(highest_version_num, int(match.group(1)))

    # Also inspect active adapter metadata if present
    active_meta = load_active_adapter_metadata(
        metadata_path=metadata_path,
        default_adapter_path=str(root / f"{base_prefix}-v2"),
    )
    active_version_str = active_meta.get("active_version", "")
    match_active = re.search(r"v(\d+)", active_version_str, re.IGNORECASE)
    if match_active:
        highest_version_num = max(highest_version_num, int(match_active.group(1)))

    candidate_num = highest_version_num + 1
    while True:
        candidate_version = f"v{candidate_num}"
        candidate_dir = root / f"{base_prefix}-{candidate_version}"
        if not candidate_dir.exists():
            return candidate_version, candidate_dir
        candidate_num += 1


def reserve_next_version(
    output_root: Union[Path, str] = "training-output",
    base_prefix: str = DEFAULT_ADAPTER_PREFIX,
    metadata_path: Optional[Path] = None,
    job_id: Optional[str] = None,
) -> Tuple[str, Path]:
    """
    Atomically allocates and reserves the next sequential adapter version directory using
    filesystem mkdir(exist_ok=False) as an atomic concurrency guard.
    Guarantees no two concurrent callers or processes can reserve the same directory.
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    highest_version_num = 1
    for item in root.iterdir():
        if item.is_dir():
            match = re.search(r"\bv(\d+)\b", item.name, re.IGNORECASE)
            if match:
                highest_version_num = max(highest_version_num, int(match.group(1)))

    active_meta = load_active_adapter_metadata(
        metadata_path=metadata_path,
        default_adapter_path=str(root / f"{base_prefix}-v2"),
    )
    active_version_str = active_meta.get("active_version", "")
    match_active = re.search(r"v(\d+)", active_version_str, re.IGNORECASE)
    if match_active:
        highest_version_num = max(highest_version_num, int(match_active.group(1)))

    candidate_num = highest_version_num + 1
    while True:
        candidate_version = f"v{candidate_num}"
        candidate_dir = root / f"{base_prefix}-{candidate_version}"
        try:
            candidate_dir.mkdir(parents=True, exist_ok=False)
            # Successfully reserved atomically!
            reservation_info = {
                "version": candidate_version,
                "jobId": job_id,
                "reservedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            (candidate_dir / ".reservation").write_text(
                json.dumps(reservation_info, indent=2), encoding="utf-8"
            )
            return candidate_version, candidate_dir
        except FileExistsError:
            # Another concurrent caller or existing directory claimed this candidate; try next
            candidate_num += 1


def normalize_samples(samples: Sequence[Any]) -> List[str]:
    """
    Extracts and normalizes text strings from input samples (strings or dicts).
    Discards empty or whitespace-only items.
    """
    normalized: List[str] = []
    if not samples:
        return normalized

    for item in samples:
        if isinstance(item, str):
            text = item.strip()
            if text:
                normalized.append(text)
        elif isinstance(item, dict):
            # Support {"text": ...} or {"content": ...}
            raw = item.get("text") or item.get("content")
            if isinstance(raw, str):
                text = raw.strip()
                if text:
                    normalized.append(text)

    return normalized


def write_training_dataset(samples: List[str], dataset_path: Path) -> int:
    """
    Atomically writes samples into a JSONL training dataset file.
    Returns the count of samples written.
    """
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dataset_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps({"text": sample}, ensure_ascii=False) + "\n")
    tmp_path.replace(dataset_path)
    return len(samples)


def write_training_manifest(manifest_data: Dict[str, Any], manifest_path: Path) -> None:
    """
    Atomically writes job manifest JSON to disk.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(manifest_path)


class TrainingController:
    """
    Coordinates versioned personal LoRA adapter training, validation, and safe activation.
    Guarantees thread-safe single-job execution and coordination with generation.
    """

    _process_training_lock = threading.Lock()

    def __init__(
        self,
        output_root: Union[Path, str] = "training-output",
        base_model: str = DEFAULT_BASE_MODEL,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        generator: Optional[PersonalizedWritingGenerator] = None,
        generation_lock: Optional[threading.Lock] = None,
        training_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        reload_callback: Optional[Callable[[str, Optional[str]], Dict[str, Any]]] = None,
        metadata_path: Optional[Path] = None,
    ):
        self.output_root = Path(output_root)
        self.base_model = base_model
        self.min_samples = min_samples
        self.generator = generator
        self.generation_lock = generation_lock
        self.training_fn = training_fn or train_adapter
        self.reload_callback = reload_callback
        self.metadata_path = metadata_path

        self._training_lock = threading.Lock()
        self._state_lock = threading.Lock()

        # Status tracking
        self._state: TrainingLifecycleState = TrainingLifecycleState.IDLE
        self._current_job_id: Optional[str] = None
        self._target_version: Optional[str] = None
        self._target_adapter_path: Optional[Path] = None
        self._sample_count: int = 0
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._error_message: Optional[str] = None
        self._active_adapter_before: Optional[str] = None
        self._active_adapter_after: Optional[str] = None
        self._last_manifest_path: Optional[Path] = None
        self._last_dataset_path: Optional[Path] = None
        self._training_metrics: Dict[str, Any] = {}

    def get_status(self) -> Dict[str, Any]:
        """Returns a snapshot of the current training controller status and metrics."""
        with self._state_lock:
            elapsed = (
                (self._end_time or time.time()) - self._start_time
                if self._start_time is not None
                else 0.0
            )
            return {
                "state": self._state.value,
                "jobId": self._current_job_id,
                "targetVersion": self._target_version,
                "targetAdapterPath": str(self._target_adapter_path) if self._target_adapter_path else None,
                "sampleCount": self._sample_count,
                "startTime": self._start_time,
                "endTime": self._end_time,
                "durationSec": round(elapsed, 2) if self._start_time else 0.0,
                "errorMessage": self._error_message,
                "activeAdapterBefore": self._active_adapter_before,
                "activeAdapterAfter": self._active_adapter_after,
                "manifestPath": str(self._last_manifest_path) if self._last_manifest_path else None,
                "datasetPath": str(self._last_dataset_path) if self._last_dataset_path else None,
                "metrics": dict(self._training_metrics),
            }

    def _set_state(
        self,
        new_state: TrainingLifecycleState,
        error_message: Optional[str] = None,
    ) -> None:
        """Internal helper to transition state thread-safely."""
        with self._state_lock:
            self._state = new_state
            if error_message is not None:
                self._error_message = error_message
            if new_state in (TrainingLifecycleState.COMPLETED, TrainingLifecycleState.ERROR):
                self._end_time = time.time()

    def run_training_job(
        self,
        samples: Sequence[Any],
        version: Optional[str] = None,
        job_id: Optional[str] = None,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        base_prefix: str = DEFAULT_ADAPTER_PREFIX,
    ) -> Dict[str, Any]:
        """
        Executes an end-to-end versioned LoRA training job across the full lifecycle:
        idle -> preparing -> training -> validating -> activating -> completed

        Raises:
            RuntimeError: If a training job is already in progress.
            ValueError: If input samples are empty/insufficient or target directory is invalid.
            FileExistsError: If target adapter directory already exists.
            Exception: On any training, validation, or activation failure.
        """
        # Acquire process-wide class lock to prevent concurrent jobs across controller instances
        if not self._process_training_lock.acquire(blocking=False):
            raise RuntimeError("A training job is already in progress. Concurrent training is prohibited.")

        # Acquire instance lock
        if not self._training_lock.acquire(blocking=False):
            self._process_training_lock.release()
            raise RuntimeError("A training job is already in progress. Concurrent training is prohibited.")

        try:
            # 1. PREPARING PHASE
            with self._state_lock:
                self._start_time = time.time()
                self._end_time = None
                self._error_message = None
                self._training_metrics = {}
                self._state = TrainingLifecycleState.PREPARING

            # Validate input samples
            valid_samples = normalize_samples(samples)
            if not valid_samples:
                err = "Training samples cannot be empty."
                self._set_state(TrainingLifecycleState.ERROR, error_message=err)
                raise ValueError(err)

            if len(valid_samples) < self.min_samples:
                err = (
                    f"Insufficient training samples: got {len(valid_samples)}, "
                    f"minimum required is {self.min_samples}."
                )
                self._set_state(TrainingLifecycleState.ERROR, error_message=err)
                raise ValueError(err)

            # Inspect currently active adapter
            active_meta = load_active_adapter_metadata(
                metadata_path=self.metadata_path,
                default_adapter_path=str(self.output_root / f"{base_prefix}-v2"),
            )
            active_adapter_path_str = active_meta.get("active_adapter_path", "")
            active_adapter_resolved = (
                Path(active_adapter_path_str).resolve()
                if active_adapter_path_str
                else None
            )

            # Resolve version and atomically reserve target directory
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")

            if version is not None:
                clean_version = version.strip().lower()
                target_version = clean_version if clean_version.startswith("v") else f"v{clean_version}"
                target_adapter_dir = self.output_root / f"{base_prefix}-{target_version}"

                # Strict Safety Check: Never train directly into currently active adapter
                if active_adapter_resolved and target_adapter_dir.resolve() == active_adapter_resolved:
                    err = f"Cannot train directly into the active adapter directory: {target_adapter_dir}"
                    self._set_state(TrainingLifecycleState.ERROR, error_message=err)
                    raise ValueError(err)

                allocated_job_id = job_id or f"train_{target_version}_{timestamp_str}"

                # Atomically reserve target directory on disk
                try:
                    target_adapter_dir.mkdir(parents=True, exist_ok=False)
                    reservation_info = {
                        "version": target_version,
                        "jobId": allocated_job_id,
                        "reservedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    (target_adapter_dir / ".reservation").write_text(
                        json.dumps(reservation_info, indent=2), encoding="utf-8"
                    )
                except FileExistsError:
                    err = f"Target adapter directory already exists and cannot be overwritten: {target_adapter_dir}"
                    self._set_state(TrainingLifecycleState.ERROR, error_message=err)
                    raise FileExistsError(err)
            else:
                prelim_job_id = job_id or f"train_pending_{timestamp_str}"
                target_version, target_adapter_dir = reserve_next_version(
                    output_root=self.output_root,
                    base_prefix=base_prefix,
                    metadata_path=self.metadata_path,
                    job_id=prelim_job_id,
                )

                # Strict Safety Check: Never train directly into currently active adapter
                if active_adapter_resolved and target_adapter_dir.resolve() == active_adapter_resolved:
                    err = f"Cannot train directly into the active adapter directory: {target_adapter_dir}"
                    self._set_state(TrainingLifecycleState.ERROR, error_message=err)
                    raise ValueError(err)

                allocated_job_id = job_id or f"train_{target_version}_{timestamp_str}"

            dataset_path = self.output_root / "datasets" / f"dataset_{target_version}_{timestamp_str}.jsonl"
            manifest_path = self.output_root / "manifests" / f"manifest_{target_version}_{timestamp_str}.json"

            with self._state_lock:
                self._current_job_id = allocated_job_id
                self._target_version = target_version
                self._target_adapter_path = target_adapter_dir
                self._sample_count = len(valid_samples)
                self._active_adapter_before = active_adapter_path_str
                self._last_dataset_path = dataset_path
                self._last_manifest_path = manifest_path

            # Write training dataset JSONL
            write_training_dataset(valid_samples, dataset_path)

            # Write initial manifest
            manifest_data = {
                "jobId": allocated_job_id,
                "targetVersion": target_version,
                "targetAdapterPath": str(target_adapter_dir),
                "datasetPath": str(dataset_path),
                "sampleCount": len(valid_samples),
                "baseModel": self.base_model,
                "epochs": epochs or EPOCHS,
                "learningRate": learning_rate or LEARNING_RATE,
                "activeAdapterBefore": active_adapter_path_str,
                "status": "preparing",
                "createdAt": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            write_training_manifest(manifest_data, manifest_path)

            # Concurrency & Coordination: Hold generation_lock to eliminate CPU/RAM contention
            lock_context = self.generation_lock if self.generation_lock is not None else contextlib.nullcontext()

            with lock_context:
                # 2. TRAINING PHASE
                self._set_state(TrainingLifecycleState.TRAINING)
                manifest_data["status"] = "training"
                write_training_manifest(manifest_data, manifest_path)

                logger.info(
                    "Starting LoRA training for version %s into '%s' (%d samples)...",
                    target_version,
                    target_adapter_dir,
                    len(valid_samples),
                )

                try:
                    train_kwargs = {
                        "dataset_path": dataset_path,
                        "output_dir": target_adapter_dir,
                        "base_model": self.base_model,
                    }
                    if epochs is not None:
                        train_kwargs["epochs"] = epochs
                    if learning_rate is not None:
                        train_kwargs["learning_rate"] = learning_rate

                    metrics = self.training_fn(**train_kwargs)
                    if isinstance(metrics, dict):
                        self._training_metrics = metrics
                except Exception as train_exc:
                    err = f"Training execution failed: {train_exc}"
                    logger.exception(err)
                    manifest_data["status"] = "error"
                    manifest_data["error"] = err
                    write_training_manifest(manifest_data, manifest_path)
                    self._set_state(TrainingLifecycleState.ERROR, error_message=err)
                    raise RuntimeError(err) from train_exc

                # 3. VALIDATING PHASE
                self._set_state(TrainingLifecycleState.VALIDATING)
                manifest_data["status"] = "validating"
                write_training_manifest(manifest_data, manifest_path)

                logger.info("Validating newly trained adapter in '%s'...", target_adapter_dir)
                try:
                    adapter_config = validate_adapter_directory(target_adapter_dir)
                    manifest_data["adapterConfig"] = adapter_config
                except Exception as val_exc:
                    err = f"Trained adapter validation failed: {val_exc}"
                    logger.error(err)
                    manifest_data["status"] = "error"
                    manifest_data["error"] = err
                    write_training_manifest(manifest_data, manifest_path)
                    self._set_state(TrainingLifecycleState.ERROR, error_message=err)
                    raise RuntimeError(err) from val_exc

                # 4. ACTIVATING PHASE
                self._set_state(TrainingLifecycleState.ACTIVATING)
                manifest_data["status"] = "activating"
                write_training_manifest(manifest_data, manifest_path)

                logger.info("Activating newly trained adapter '%s' (%s)...", target_adapter_dir, target_version)
                try:
                    # Single-owner lock strategy: caller/controller already holds generation_lock.
                    # generator.reload_adapter or internal callback is invoked directly without re-acquiring.
                    if self.generator is not None:
                        self.generator.reload_adapter(str(target_adapter_dir), version=target_version)
                    elif self.reload_callback is not None:
                        self.reload_callback(str(target_adapter_dir), target_version)
                    else:
                        # Fallback for standalone headless training without in-memory generator
                        new_meta = {
                            "active_adapter_path": str(target_adapter_dir),
                            "active_version": target_version,
                            "activated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "previous_adapter_path": active_adapter_path_str,
                            "previous_version": active_meta.get("active_version"),
                            "status": "ready",
                        }
                        save_active_adapter_metadata(new_meta, metadata_path=self.metadata_path)

                    self._active_adapter_after = str(target_adapter_dir)
                except Exception as act_exc:
                    err = f"Adapter activation failed: {act_exc}"
                    logger.exception(err)
                    manifest_data["status"] = "error"
                    manifest_data["error"] = err
                    write_training_manifest(manifest_data, manifest_path)
                    self._set_state(TrainingLifecycleState.ERROR, error_message=err)
                    raise RuntimeError(err) from act_exc

            # 5. COMPLETED PHASE
            self._set_state(TrainingLifecycleState.COMPLETED)
            manifest_data["status"] = "completed"
            manifest_data["completedAt"] = time.strftime("%Y-%m-%d %H:%M:%S")
            manifest_data["activeAdapterAfter"] = self._active_adapter_after
            write_training_manifest(manifest_data, manifest_path)

            return {
                "status": "completed",
                "jobId": allocated_job_id,
                "version": target_version,
                "adapterPath": str(target_adapter_dir),
                "sampleCount": len(valid_samples),
                "durationSec": self.get_status()["durationSec"],
                "activeAdapterBefore": self._active_adapter_before,
                "activeAdapterAfter": self._active_adapter_after,
                "manifestPath": str(manifest_path),
                "datasetPath": str(dataset_path),
            }

        finally:
            if self._training_lock.locked():
                self._training_lock.release()
            if self._process_training_lock.locked():
                self._process_training_lock.release()
