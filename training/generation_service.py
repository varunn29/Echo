"""
Echo — Local Generation Service V1
Persistent local HTTP service exposing the trained LoRA V2 personal writing generator.
Binds strictly to 127.0.0.1 and enforces exact Echo Chrome extension origin validation.
"""

import argparse
import json
import logging
import signal
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import psutil

# Import existing generator
from generate_with_adapter import (
    PersonalizedWritingGenerator,
    DEFAULT_BASE_MODEL,
    DEFAULT_ADAPTER_PATH,
    DEFAULT_NUM_THREADS,
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

# Global Service State
state_lock = threading.Lock()
generator_lock = threading.Lock()
generator: Optional[PersonalizedWritingGenerator] = None
is_model_ready = False
model_error_msg: Optional[str] = None
service_start_time = time.time()
active_allowed_origin = DEFAULT_ALLOWED_ORIGIN
server_instance: Optional[HTTPServer] = None


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

        self._send_json_response(404, {"error": f"Endpoint not found: {self.path}"}, origin=origin)

    def do_POST(self) -> None:
        """Handle POST requests (e.g., /generate)."""
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

        if path not in ("/generate", "/api/generate"):
            self._send_json_response(404, {"error": f"Endpoint not found: {self.path}"}, origin=origin)
            return

        # Check model availability
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

        try:
            max_tokens = int(payload.get("maxTokens", 85))
            max_tokens = max(10, min(max_tokens, 256))
        except (ValueError, TypeError):
            max_tokens = 85

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

        seed = payload.get("seed", 42)
        if not isinstance(seed, int):
            seed = 42

        # Serialize inference with concurrency lock to avoid CPU thrashing
        logger.info("Generating personalized draft for topic: '%s' (mode: %s)...", topic.strip()[:60], mode)
        with generator_lock:
            try:
                res = generator.generate(
                    topic=topic.strip(),
                    personal_knowledge=personal_knowledge,
                    lead_in=lead_in.strip(),
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
            "leadIn": res["lead_in"],
            "tokensGenerated": res["tokens_generated"],
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
                "model": generator.base_model_name,
                "adapter": generator.adapter_path,
                "repetitionPenalty": repetition_penalty,
                "maxTokens": max_tokens,
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
    global generator, is_model_ready, model_error_msg, active_allowed_origin, server_instance

    active_allowed_origin = allowed_origin

    logger.info("=" * 70)
    logger.info("Echo — Local Generation Service V1 Starting")
    logger.info("Host: %s (Loopback only)", host)
    logger.info("Port: %d", port)
    logger.info("Allowed Extension Origin: %s", active_allowed_origin)
    logger.info("Base Model: %s", base_model_name)
    logger.info("LoRA Adapter: %s", adapter_path)
    logger.info("=" * 70)

    # 1. Start HTTP Server bound strictly to host (127.0.0.1) first
    server_address = (host, port)
    try:
        httpd = HTTPServer(server_address, GenerationHTTPRequestHandler)
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
        with state_lock:
            generator = gen
            is_model_ready = True
            model_error_msg = None
        logger.info("Model initialization complete and ready to serve requests.")
    except Exception as e:
        logger.exception("Failed to initialize model: %s", str(e))
        with state_lock:
            generator = None
            is_model_ready = False
            model_error_msg = str(e)
        logger.warning("Service running in degraded/error state. Model loading failed.")

    logger.info("HTTP Service listening on http://%s:%d", host, port)
    logger.info("Health endpoint: http://%s:%d/health", host, port)
    logger.info("Generate endpoint: http://%s:%d/generate", host, port)

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
