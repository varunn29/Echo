"""
Echo — Local Generation Service V1
Persistent local HTTP service exposing the trained LoRA V2 personal writing generator.
Binds strictly to 127.0.0.1 and enforces exact Echo Chrome extension origin validation.
"""

import argparse
import json
import logging
import re
import signal
import sys
import threading
import time
import uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import psutil

# Import existing generator
try:
    from generate_with_adapter import (
        PersonalizedWritingGenerator,
        DEFAULT_BASE_MODEL,
        DEFAULT_ADAPTER_PATH,
        DEFAULT_NUM_THREADS,
        load_active_adapter_metadata,
        save_active_adapter_metadata,
        validate_adapter_directory,
    )
    from training_controller import (
        TrainingController,
        TrainingLifecycleState,
        DEFAULT_MIN_SAMPLES,
        normalize_samples,
    )
except ImportError:
    from training.generate_with_adapter import (
        PersonalizedWritingGenerator,
        DEFAULT_BASE_MODEL,
        DEFAULT_ADAPTER_PATH,
        DEFAULT_NUM_THREADS,
        load_active_adapter_metadata,
        save_active_adapter_metadata,
        validate_adapter_directory,
    )
    from training.training_controller import (
        TrainingController,
        TrainingLifecycleState,
        DEFAULT_MIN_SAMPLES,
        normalize_samples,
    )

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11435
DEFAULT_ALLOWED_ORIGIN = "chrome-extension://mbghefdhlgoladjadphnagkiclmckohf"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("EchoGenerationService")


def sanitize_error_message(err: Optional[str]) -> Optional[str]:
    """Removes local filesystem paths, stack frames, and sensitive data from error messages."""
    if not err:
        return None
    sanitized = str(err).strip()
    # Strip Windows drive paths (e.g. C:\... or D:\...)
    sanitized = re.sub(r"[a-zA-Z]:\\[^\s'\"]+", "<path>", sanitized)
    # Strip POSIX paths (e.g. /home/... or /tmp/...)
    sanitized = re.sub(r"/(?:[a-zA-Z0-9_\-\.]+/)+[a-zA-Z0-9_\-\.]*", "<path>", sanitized)
    # If multi-line, take only the last summary line
    lines = [l.strip() for l in sanitized.splitlines() if l.strip()]
    return lines[-1] if lines else sanitized


# Global Service State
state_lock = threading.Lock()
generator_lock = threading.Lock()
training_submission_lock = threading.Lock()
generator: Optional[PersonalizedWritingGenerator] = None
training_controller: Optional[TrainingController] = None
active_training_thread: Optional[threading.Thread] = None
active_training_job_id: Optional[str] = None
is_model_ready = False
model_error_msg: Optional[str] = None
service_start_time = time.time()
active_allowed_origin = DEFAULT_ALLOWED_ORIGIN
server_instance: Optional[ThreadingHTTPServer] = None


class GenerationHTTPRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests with strict origin checking, host validation, and serialized generation."""

    def log_message(self, format: str, *args: Any) -> None:
        """Route internal server logs through our logger."""
        logger.info("%s - - %s", self.address_string(), format % args)

    def _send_json_response(
        self,
        status_code: int,
        data: Dict[str, Any],
        origin: Optional[str] = None,
    ) -> None:
        """Sends a JSON response with appropriate headers and origin matching."""
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.close_connection = True

        # Exact origin reflection only when authorized
        if origin and origin == active_allowed_origin:
            self.send_header("Access-Control-Allow-Origin", active_allowed_origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
            self.send_header("Vary", "Origin")

        self.end_headers()
        self.wfile.write(body)

    def _validate_host_header(self) -> bool:
        """Validates that Host header points to local loopback (DNS rebinding protection)."""
        host_header = self.headers.get("Host", "").strip().lower()
        if not host_header:
            return False

        # Allow 127.0.0.1, localhost, and with port
        allowed_hosts = {
            "127.0.0.1",
            "localhost",
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        }
        return host_header in allowed_hosts

    def _check_origin(self) -> tuple[bool, Optional[str]]:
        """
        Validates request origin.
        Returns (is_allowed, origin_str).
        Missing origin is allowed for local direct loopback requests (e.g. CLI tools/curl).
        Present origin MUST match active_allowed_origin exactly.
        """
        origin = self.headers.get("Origin")
        if origin is None:
            # Local loopback client without browser origin
            return True, None

        origin = origin.strip()
        if origin == active_allowed_origin:
            return True, origin

        # Foreign or unauthorized origin
        return False, origin

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests from the extension."""
        if not self._validate_host_header():
            self._send_json_response(400, {"error": "Invalid or disallowed Host header"})
            return

        is_allowed, origin = self._check_origin()
        if not is_allowed:
            logger.warning("Rejected unauthorized preflight Origin: %s", origin)
            self._send_json_response(403, {"error": "Forbidden: Unauthorized origin"})
            return

        self.send_response(204)
        if origin:
            self.send_header("Access-Control-Allow-Origin", active_allowed_origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
            self.send_header("Access-Control-Max-Age", "86400")
            self.send_header("Vary", "Origin")
        self.end_headers()

    def do_GET(self) -> None:
        """Handle GET requests (e.g., /health)."""
        if not self._validate_host_header():
            self._send_json_response(400, {"error": "Invalid or disallowed Host header"})
            return

        is_allowed, origin = self._check_origin()
        if not is_allowed:
            logger.warning("Rejected unauthorized GET request from Origin: %s", origin)
            self._send_json_response(403, {"error": "Forbidden: Unauthorized origin"})
            return

        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path in ("", "/health", "/api/health"):
            with state_lock:
                if is_model_ready and generator is not None:
                    current_ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                    self._send_json_response(
                        200,
                        {
                            "status": "ready",
                            "model": generator.base_model_name,
                            "adapter": generator.adapter_path,
                            "adapterVersion": getattr(generator, "active_version", "v2"),
                            "lastReloadStatus": getattr(generator, "last_reload_status", "ready"),
                            "lastReloadTime": getattr(generator, "last_reload_time", None),
                            "previousAdapter": getattr(generator, "previous_adapter_path", None),
                            "loadTimeSec": round(generator.load_time_sec, 2),
                            "residentRamMb": round(current_ram_mb, 1),
                            "uptimeSec": round(time.time() - service_start_time, 1),
                            "allowedOrigin": active_allowed_origin,
                        },
                        origin=origin,
                    )
                else:
                    self._send_json_response(
                        503,
                        {
                            "status": "error",
                            "error": "Model is unavailable. Check server logs.",
                        },
                        origin=origin,
                    )
            return

        # Handle GET /training/status
        if path in ("/training/status", "/api/training/status"):
            with state_lock:
                if training_controller is None:
                    self._send_json_response(
                        200,
                        {
                            "status": "idle",
                            "jobId": None,
                            "stage": "idle",
                            "sampleCount": 0,
                            "adapterVersion": None,
                            "startedAt": None,
                            "completedAt": None,
                            "durationSec": 0.0,
                            "error": None,
                        },
                        origin=origin,
                    )
                    return

                status = training_controller.get_status()
                started_at_str = (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(status["startTime"]))
                    if status.get("startTime")
                    else None
                )
                completed_at_str = (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(status["endTime"]))
                    if status.get("endTime")
                    else None
                )
                raw_err = status.get("errorMessage")
                sanitized_err = sanitize_error_message(raw_err)

                self._send_json_response(
                    200,
                    {
                        "status": status.get("state", "idle"),
                        "jobId": status.get("jobId"),
                        "stage": status.get("state", "idle"),
                        "sampleCount": status.get("sampleCount", 0),
                        "adapterVersion": status.get("targetVersion"),
                        "startedAt": started_at_str,
                        "completedAt": completed_at_str,
                        "durationSec": status.get("durationSec", 0.0),
                        "error": sanitized_err,
                    },
                    origin=origin,
                )
                return

        self._send_json_response(404, {"error": f"Endpoint not found: {self.path}"}, origin=origin)

    def do_POST(self) -> None:
        """Handle POST requests (/generate and /reload)."""
        if not self._validate_host_header():
            self._send_json_response(400, {"error": "Invalid or disallowed Host header"})
            return

        is_allowed, origin = self._check_origin()
        if not is_allowed:
            logger.warning("Rejected unauthorized POST request from Origin: %s", origin)
            self._send_json_response(403, {"error": "Forbidden: Unauthorized origin"})
            return

        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path not in ("/generate", "/api/generate", "/reload", "/api/reload", "/train", "/api/train"):
            self._send_json_response(404, {"error": f"Endpoint not found: {self.path}"}, origin=origin)
            return

        # Handle POST /train
        if path in ("/train", "/api/train"):
            global active_training_thread, active_training_job_id
            with state_lock:
                if not is_model_ready or generator is None or training_controller is None:
                    self._send_json_response(
                        503,
                        {"error": "Training service is unavailable. Check server logs."},
                        origin=origin,
                    )
                    return

                current_status = training_controller.get_status()
                is_active_state = current_status.get("state") in (
                    TrainingLifecycleState.PREPARING.value,
                    TrainingLifecycleState.TRAINING.value,
                    TrainingLifecycleState.VALIDATING.value,
                    TrainingLifecycleState.ACTIVATING.value,
                )
                is_thread_alive = active_training_thread is not None and active_training_thread.is_alive()
                is_reserved = active_training_job_id is not None

                if is_active_state or is_thread_alive or is_reserved:
                    active_id = active_training_job_id or current_status.get("jobId")
                    logger.warning(
                        "Rejected /train request: training job is already in progress (%s)",
                        active_id,
                    )
                    self._send_json_response(
                        409,
                        {
                            "status": "conflict",
                            "error": "A training job is already in progress. Concurrent training is prohibited.",
                            "activeJobId": active_id,
                        },
                        origin=origin,
                    )
                    return

            try:
                content_length = int(self.headers.get("Content-Length", 0))
            except (ValueError, TypeError):
                content_length = 0

            if content_length <= 0:
                self._send_json_response(400, {"error": "Missing or empty request body"}, origin=origin)
                return

            MAX_TRAIN_BODY_SIZE = 2 * 1024 * 1024  # 2 MB limit
            if content_length > MAX_TRAIN_BODY_SIZE:
                self._send_json_response(413, {"error": "Request body too large (max 2MB)"}, origin=origin)
                return

            try:
                body_bytes = self.rfile.read(content_length)
                payload = json.loads(body_bytes.decode("utf-8"))
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as e:
                logger.warning("Malformed JSON in /train request: %s", str(e))
                self._send_json_response(400, {"error": "Malformed JSON request body"}, origin=origin)
                return

            if not isinstance(payload, dict):
                self._send_json_response(400, {"error": "Request body must be a JSON object"}, origin=origin)
                return

            # Reject any client attempt to override paths or hyperparameters
            disallowed_keys = {
                "output_dir", "outputDir", "dataset_path", "datasetPath",
                "adapter_path", "adapterPath", "output_root", "outputRoot",
                "metadata_path", "metadataPath", "epochs", "learning_rate",
                "learningRate", "batch_size", "batchSize", "base_model", "baseModel",
            }
            present_disallowed = disallowed_keys.intersection(payload.keys())
            if present_disallowed:
                self._send_json_response(
                    400,
                    {
                        "error": (
                            "Custom filesystem paths or training configurations are not permitted: "
                            f"{sorted(present_disallowed)}"
                        )
                    },
                    origin=origin,
                )
                return

            samples = payload.get("samples")
            if samples is None or not isinstance(samples, list):
                self._send_json_response(
                    400,
                    {"error": "Field 'samples' is required and must be a list of writing sample strings or objects"},
                    origin=origin,
                )
                return

            if len(samples) == 0:
                self._send_json_response(400, {"error": "Field 'samples' cannot be empty"}, origin=origin)
                return

            MAX_SAMPLES_LIMIT = 500
            if len(samples) > MAX_SAMPLES_LIMIT:
                self._send_json_response(
                    400,
                    {"error": f"Field 'samples' exceeds maximum limit of {MAX_SAMPLES_LIMIT} items"},
                    origin=origin,
                )
                return

            valid_samples = normalize_samples(samples)
            min_required = training_controller.min_samples
            if len(valid_samples) < min_required:
                self._send_json_response(
                    400,
                    {
                        "error": (
                            f"Insufficient training samples: got {len(valid_samples)}, "
                            f"minimum required is {min_required}"
                        )
                    },
                    origin=origin,
                )
                return

            job_id = f"train_api_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

            # Atomically reserve the training slot before spawning the worker thread
            with training_submission_lock:
                with state_lock:
                    current_status = training_controller.get_status()
                    is_active_state = current_status.get("state") in (
                        TrainingLifecycleState.PREPARING.value,
                        TrainingLifecycleState.TRAINING.value,
                        TrainingLifecycleState.VALIDATING.value,
                        TrainingLifecycleState.ACTIVATING.value,
                    )
                    is_thread_alive = active_training_thread is not None and active_training_thread.is_alive()
                    is_reserved = active_training_job_id is not None

                    if is_active_state or is_thread_alive or is_reserved:
                        active_id = active_training_job_id or current_status.get("jobId")
                        logger.warning(
                            "Rejected concurrent /train request: training job is already in progress or reserved (%s)",
                            active_id,
                        )
                        self._send_json_response(
                            409,
                            {
                                "status": "conflict",
                                "error": "A training job is already in progress. Concurrent training is prohibited.",
                                "activeJobId": active_id,
                            },
                            origin=origin,
                        )
                        return

                    # Reserve slot atomically
                    active_training_job_id = job_id

            def training_worker():
                global active_training_thread, active_training_job_id
                logger.info("Background training worker started for job %s (%d samples)...", job_id, len(valid_samples))
                try:
                    training_controller.run_training_job(valid_samples, job_id=job_id)
                    logger.info("Background training worker completed successfully for job %s.", job_id)
                except Exception as exc:
                    logger.exception("Background training worker failed for job %s: %s", job_id, str(exc))
                finally:
                    with state_lock:
                        if active_training_job_id == job_id:
                            active_training_job_id = None

            try:
                t = threading.Thread(target=training_worker, name=f"EchoTrainingWorker-{job_id}", daemon=True)
                with state_lock:
                    active_training_thread = t
                t.start()
            except Exception as thread_err:
                logger.exception("Failed to spawn background training worker thread for job %s: %s", job_id, str(thread_err))
                with state_lock:
                    if active_training_job_id == job_id:
                        active_training_job_id = None
                    active_training_thread = None
                sanitized_err = sanitize_error_message(str(thread_err)) or "Failed to start background training worker"
                self._send_json_response(
                    500,
                    {
                        "error": f"Failed to initialize training worker thread: {sanitized_err}",
                        "status": "error",
                    },
                    origin=origin,
                )
                return

            logger.info("Accepted training job %s with %d samples.", job_id, len(valid_samples))
            self._send_json_response(
                202,
                {
                    "status": "accepted",
                    "jobId": job_id,
                    "message": "Training job accepted and running in background.",
                    "sampleCount": len(valid_samples),
                },
                origin=origin,
            )
            return

        # Handle POST /reload
        if path in ("/reload", "/api/reload"):
            with state_lock:
                if not is_model_ready or generator is None:
                    self._send_json_response(
                        503,
                        {"error": "Model service is unavailable. Check server logs."},
                        origin=origin,
                    )
                    return

            target_adapter_path = None
            version = None

            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 0:
                if content_length > 65536:
                    self._send_json_response(413, {"error": "Request body too large (max 64KB)"}, origin=origin)
                    return
                try:
                    body_bytes = self.rfile.read(content_length)
                    payload = json.loads(body_bytes.decode("utf-8")) if body_bytes.strip() else {}
                    if isinstance(payload, dict):
                        target_adapter_path = payload.get("adapterPath") or payload.get("adapter_path")
                        version = payload.get("version")
                except Exception as e:
                    logger.warning("Malformed JSON in reload request: %s", str(e))
                    self._send_json_response(400, {"error": "Malformed JSON request body"}, origin=origin)
                    return

            if not target_adapter_path:
                meta = load_active_adapter_metadata(default_adapter_path=generator.adapter_path)
                target_adapter_path = meta.get("active_adapter_path", generator.adapter_path)
                version = version or meta.get("active_version")

            if not isinstance(target_adapter_path, str) or not target_adapter_path.strip():
                self._send_json_response(
                    400,
                    {"error": "Field 'adapterPath' must be a non-empty string"},
                    origin=origin,
                )
                return

            if len(target_adapter_path) > 500:
                self._send_json_response(
                    400,
                    {"error": "Field 'adapterPath' exceeds 500 characters"},
                    origin=origin,
                )
                return

            logger.info(
                "Executing safe adapter reload to: '%s' (version: %s)...",
                target_adapter_path,
                version or "auto",
            )

            with generator_lock:
                with state_lock:
                    try:
                        reload_result = generator.reload_adapter(target_adapter_path.strip(), version=version)
                        self._send_json_response(200, reload_result, origin=origin)
                        return
                    except (ValueError, FileNotFoundError) as ve:
                        logger.warning("Adapter reload rejected: %s", str(ve))
                        self._send_json_response(
                            400,
                            {
                                "status": "error",
                                "error": str(ve),
                                "activeAdapter": generator.adapter_path,
                                "activeVersion": getattr(generator, "active_version", "v2"),
                            },
                            origin=origin,
                        )
                        return
                    except Exception as e:
                        logger.exception("Unexpected error during adapter reload: %s", str(e))
                        self._send_json_response(
                            500,
                            {
                                "status": "error",
                                "error": f"Reload failed: {str(e)}",
                                "activeAdapter": generator.adapter_path,
                                "activeVersion": getattr(generator, "active_version", "v2"),
                            },
                            origin=origin,
                        )
                        return

        # Check model availability for generation
        with state_lock:
            if not is_model_ready or generator is None:
                self._send_json_response(
                    503,
                    {"error": "Model is unavailable. Check server logs."},
                    origin=origin,
                )
                return

        # Read request body
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length <= 0:
                self._send_json_response(400, {"error": "Missing or empty request body"}, origin=origin)
                return
            if content_length > 65536: # 64 KB limit
                self._send_json_response(413, {"error": "Request body too large (max 64KB)"}, origin=origin)
                return

            body_bytes = self.rfile.read(content_length)
            payload = json.loads(body_bytes.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.warning("Malformed JSON request received: %s", str(e))
            self._send_json_response(400, {"error": "Malformed JSON request body"}, origin=origin)
            return

        # Validate input schema with granular boundaries
        topic = payload.get("topic")
        if not topic or not isinstance(topic, str) or not topic.strip():
            self._send_json_response(
                400,
                {"error": "Field 'topic' is required and must be a non-empty string"},
                origin=origin,
            )
            return
        if len(topic) > 500:
            self._send_json_response(
                400,
                {"error": "Field 'topic' exceeds maximum length of 500 characters"},
                origin=origin,
            )
            return

        personal_knowledge = payload.get("personalKnowledge")
        if personal_knowledge is None:
            personal_knowledge = []
        elif not isinstance(personal_knowledge, list):
            self._send_json_response(
                400,
                {"error": "Field 'personalKnowledge' must be a list of strings"},
                origin=origin,
            )
            return
        elif len(personal_knowledge) > 10:
            self._send_json_response(
                400,
                {"error": "Field 'personalKnowledge' exceeds maximum limit of 10 items"},
                origin=origin,
            )
            return
        else:
            # Ensure elements are strings and within individual length bounds
            if not all(isinstance(k, str) for k in personal_knowledge):
                self._send_json_response(
                    400,
                    {"error": "All items in 'personalKnowledge' must be strings"},
                    origin=origin,
                )
                return
            if any(len(k) > 500 for k in personal_knowledge):
                self._send_json_response(
                    400,
                    {"error": "Each item in 'personalKnowledge' must not exceed 500 characters"},
                    origin=origin,
                )
                return

        lead_in = payload.get("leadIn", "")
        if lead_in is not None and not isinstance(lead_in, str):
            self._send_json_response(
                400,
                {"error": "Field 'leadIn' must be a string"},
                origin=origin,
            )
            return
        lead_in = lead_in or ""
        if len(lead_in) > 100:
            self._send_json_response(
                400,
                {"error": "Field 'leadIn' exceeds maximum length of 100 characters"},
                origin=origin,
            )
            return

        mode = payload.get("mode", "sampled").lower()
        if mode not in ("sampled", "greedy"):
            mode = "sampled"

        raw_length_mode = payload.get("lengthMode") or payload.get("length_mode") or "medium"
        if isinstance(raw_length_mode, str) and raw_length_mode.lower() in ("short", "medium", "long"):
            length_mode = raw_length_mode.lower()
        else:
            length_mode = "medium"

        has_explicit_max_tokens = "maxTokens" in payload or "max_tokens" in payload
        max_tokens = None
        if has_explicit_max_tokens:
            try:
                parsed_max = int(payload.get("maxTokens") or payload.get("max_tokens"))
                max_tokens = max(10, min(parsed_max, 256))
            except (ValueError, TypeError):
                max_tokens = None

        try:
            temperature = float(payload.get("temperature", 0.65))
            temperature = max(0.1, min(temperature, 1.5))
        except (ValueError, TypeError):
            temperature = 0.65

        try:
            repetition_penalty = float(payload.get("repetitionPenalty", 1.15))
            repetition_penalty = max(1.0, min(repetition_penalty, 2.0))
        except (ValueError, TypeError):
            repetition_penalty = 1.15

        raw_seed = payload.get("seed")
        if raw_seed is not None and isinstance(raw_seed, int):
            seed = raw_seed
        else:
            seed = int(time.time() * 1000) % 100000

        # Serialize inference with concurrency lock to avoid CPU thrashing
        logger.info(
            "Generating personalized draft for topic: '%s' (mode: %s, length: %s, seed: %d)...",
            topic.strip()[:60],
            mode,
            length_mode,
            seed,
        )
        with generator_lock:
            try:
                res = generator.generate(
                    topic=topic.strip(),
                    personal_knowledge=personal_knowledge,
                    lead_in=lead_in.strip(),
                    length_mode=length_mode,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    do_sample=(mode == "sampled"),
                    seed=seed,
                )
            except Exception as e:
                logger.exception("Error during model generation: %s", str(e))
                self._send_json_response(
                    500,
                    {"error": "Model generation failed. Check server logs."},
                    origin=origin,
                )
                return

        response_data = {
            "completeDraft": res["complete_draft"],
            "generatedText": res["generated_text"],
            "rawGeneratedText": res.get("raw_generated_text", res["generated_text"]),
            "leadIn": res["lead_in"],
            "tokensGenerated": res["tokens_generated"],
            "tokenCount": res.get("final_token_count", res.get("tokens_generated", 0)),
            "finalTokenCount": res.get("final_token_count", res.get("tokens_generated", 0)),
            "rawTokenCount": res.get("raw_token_count", res.get("tokens_generated", 0)),
            "wordCount": res.get("word_count", len(res["complete_draft"].split())),
            "sentenceCount": res.get("sentence_count", 1),
            "paragraphCount": res.get("paragraph_count", 1),
            "source": res.get("source", "lora_sampled"),
            "elapsedSec": res["elapsed_sec"],
            "tokensPerSec": res["tokens_per_sec"],
            "peakRamMb": res["peak_ram_mb"],
            "groundingCheck": {
                "isGrounded": res.get("grounding_check", {}).get("is_grounded", True),
                "flaggedIssues": res.get("grounding_check", {}).get("flagged_issues", []),
                "fallbackUsed": res.get("grounding_check", {}).get("fallback_used", False),
            },
            "metadata": {
                "topic": topic.strip(),
                "mode": mode,
                "lengthMode": length_mode,
                "model": generator.base_model_name,
                "adapter": generator.adapter_path,
                "repetitionPenalty": repetition_penalty,
                "maxTokens": res.get("settings", {}).get("max_new_tokens", max_tokens or 85),
                "source": res.get("source", "lora_sampled"),
                "seed": seed,
                "tokenCount": res.get("final_token_count", res.get("tokens_generated", 0)),
                "finalTokenCount": res.get("final_token_count", res.get("tokens_generated", 0)),
                "rawTokenCount": res.get("raw_token_count", res.get("tokens_generated", 0)),
            },
        }

        self._send_json_response(200, response_data, origin=origin)


def start_service(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    base_model_name: str = DEFAULT_BASE_MODEL,
    adapter_path: str = DEFAULT_ADAPTER_PATH,
    allowed_origin: str = DEFAULT_ALLOWED_ORIGIN,
    num_threads: int = DEFAULT_NUM_THREADS,
) -> None:
    """Initializes the model once and starts the HTTP server."""
    global generator, training_controller, is_model_ready, model_error_msg, active_allowed_origin, server_instance

    active_allowed_origin = allowed_origin

    logger.info("=" * 70)
    logger.info("Echo — Local Generation Service V1 Starting")
    logger.info("Host: %s (Loopback only)", host)
    logger.info("Port: %d", port)
    logger.info("Allowed Extension Origin: %s", active_allowed_origin)
    logger.info("Base Model: %s", base_model_name)
    logger.info("LoRA Adapter: %s", adapter_path)
    logger.info("=" * 70)

    # 1. Start Threaded HTTP Server bound strictly to host (127.0.0.1) first
    server_address = (host, port)
    try:
        httpd = ThreadingHTTPServer(server_address, GenerationHTTPRequestHandler)
        httpd.daemon_threads = True
        server_instance = httpd
    except Exception as e:
        logger.error("Failed to bind HTTP server to %s:%d: %s", host, port, str(e))
        sys.exit(1)

    def handle_signal(sig: int, frame: Any) -> None:
        logger.info("Received termination signal (%d). Shutting down cleanly...", sig)
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    if threading.current_thread() is threading.main_thread():
        try:
            signal.signal(signal.SIGINT, handle_signal)
            signal.signal(signal.SIGTERM, handle_signal)
        except (ValueError, AttributeError):
            pass

    logger.info("HTTP Service bound to http://%s:%d", host, port)

    # 2. Load Model Once
    try:
        gen = PersonalizedWritingGenerator(
            base_model_name=base_model_name,
            adapter_path=adapter_path,
            num_threads=num_threads,
        )
        gen.load_model()
        ctrl = TrainingController(
            output_root="training-output",
            base_model=base_model_name,
            generator=gen,
            generation_lock=generator_lock,
        )
        with state_lock:
            generator = gen
            training_controller = ctrl
            is_model_ready = True
            model_error_msg = None
        logger.info("Model and training controller initialization complete and ready to serve requests.")
    except Exception as e:
        logger.exception("Failed to initialize model: %s", str(e))
        with state_lock:
            generator = None
            training_controller = None
            is_model_ready = False
            model_error_msg = str(e)
        logger.warning("Service running in degraded/error state. Model loading failed.")

    logger.info("HTTP Service listening on http://%s:%d", host, port)
    logger.info("Health endpoint: http://%s:%d/health", host, port)
    logger.info("Generate endpoint: http://%s:%d/generate", host, port)
    logger.info("Reload endpoint: http://%s:%d/reload", host, port)
    logger.info("Training status: http://%s:%d/training/status", host, port)
    logger.info("Training trigger: http://%s:%d/train", host, port)

    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        logger.info("HTTP server closed. Clean exit.")


def main():
    parser = argparse.ArgumentParser(description="Echo Local LoRA V2 Generation Service")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Host address to bind (must be 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on (default: 11435)")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL, help="Base model path")
    parser.add_argument("--adapter", type=str, default=DEFAULT_ADAPTER_PATH, help="LoRA adapter path")
    parser.add_argument("--allowed-origin", type=str, default=DEFAULT_ALLOWED_ORIGIN, help="Allowed Chrome extension origin")
    parser.add_argument("--threads", type=int, default=DEFAULT_NUM_THREADS, help="PyTorch CPU threads")

    args = parser.parse_args()

    # Enforce loopback binding boundary
    if args.host not in ("127.0.0.1", "localhost"):
        logger.error("Security violation: Binding to external network interface '%s' is disallowed. Must bind to 127.0.0.1.", args.host)
        sys.exit(1)

    start_service(
        host=args.host,
        port=args.port,
        base_model_name=args.base_model,
        adapter_path=args.adapter,
        allowed_origin=args.allowed_origin,
        num_threads=args.threads,
    )


if __name__ == "__main__":
    main()
