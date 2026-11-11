"""
Echo — Training Service Bridge Unit Tests (Phase 1, Step 5)

Tests POST /train and GET /training/status endpoints on generation_service.py:
- Request validation, schema limits, origin/host guards
- Background worker execution and 202 acceptance
- Concurrent conflict 409 rejection
- Status reporting across idle, training, completed, and sanitized error states
- Backward compatibility of /health, /generate, and /reload
- Lock coordination between training and generation
"""

import http.client
import io
import json
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import training.generation_service as gen_service_mod
from training.generation_service import (
    GenerationHTTPRequestHandler,
    DEFAULT_ALLOWED_ORIGIN,
    DEFAULT_PORT,
    sanitize_error_message,
    validate_allowed_origin,
)
from training.training_controller import (
    TrainingController,
    TrainingLifecycleState,
    DEFAULT_MIN_SAMPLES,
)


class MockServer:
    server_port = DEFAULT_PORT


class TestTrainingServiceBridge(unittest.TestCase):

    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.output_root = Path(self.temp_dir.name)
        self.metadata_path = self.output_root / "active_adapter.json"

        # Mock generator
        self.mock_gen = MagicMock()
        self.mock_gen.base_model_name = "Qwen/Qwen3.5-0.8B-Base"
        self.mock_gen.adapter_path = "training-output/qwen3.5-0.8b-lora-v2"
        self.mock_gen.active_version = "v2"
        self.mock_gen.last_reload_status = "ready"
        self.mock_gen.last_reload_time = None
        self.mock_gen.previous_adapter_path = None
        self.mock_gen.load_time_sec = 1.2
        self.mock_gen.generate.return_value = {
            "complete_draft": "Draft text",
            "generated_text": "text",
            "lead_in": "",
            "tokens_generated": 10,
            "elapsed_sec": 0.2,
            "tokens_per_sec": 50.0,
            "peak_ram_mb": 150.0,
            "grounding_check": {"is_grounded": True, "flagged_issues": []},
        }

        # Mock training controller
        self.mock_controller = MagicMock(spec=TrainingController)
        self.mock_controller.min_samples = 25
        self.mock_controller.get_status.return_value = {
            "state": "idle",
            "jobId": None,
            "targetVersion": None,
            "sampleCount": 0,
            "startTime": None,
            "endTime": None,
            "durationSec": 0.0,
            "errorMessage": None,
        }

        # Bind to generation_service globals
        gen_service_mod.generator = self.mock_gen
        gen_service_mod.training_controller = self.mock_controller
        gen_service_mod.active_training_thread = None
        gen_service_mod.active_training_job_id = None
        gen_service_mod.is_model_ready = True
        gen_service_mod.active_allowed_origin = DEFAULT_ALLOWED_ORIGIN

        self.valid_25_samples = [
            f"Engineering standard sample {i:02d}: Software design with modular state boundaries."
            for i in range(25)
        ]

    def tearDown(self):
        self.temp_dir.cleanup()
        gen_service_mod.generator = None
        gen_service_mod.training_controller = None
        gen_service_mod.active_training_thread = None
        gen_service_mod.active_training_job_id = None
        gen_service_mod.is_model_ready = False
        if gen_service_mod.generator_lock.locked():
            try:
                gen_service_mod.generator_lock.release()
            except RuntimeError:
                pass

    def _execute_request(
        self,
        method: str,
        path: str,
        body: Any = None,
        headers: Optional[Dict[str, str]] = None,
        raw_body: Optional[bytes] = None,
    ) -> tuple[int, Dict[str, Any], Dict[str, str]]:
        """Simulates an HTTP request against GenerationHTTPRequestHandler."""
        if raw_body is not None:
            body_bytes = raw_body
        elif body is not None:
            body_bytes = json.dumps(body).encode("utf-8")
        else:
            body_bytes = b""

        rfile = io.BytesIO(body_bytes)
        wfile = io.BytesIO()

        req = MagicMock(spec=GenerationHTTPRequestHandler)
        req.server = MockServer()
        req.command = method
        req.path = path
        req.rfile = rfile
        req.wfile = wfile

        default_headers = {
            "Host": f"127.0.0.1:{DEFAULT_PORT}",
            "Origin": DEFAULT_ALLOWED_ORIGIN,
            "Content-Length": str(len(body_bytes)),
        }
        if headers:
            for k, v in headers.items():
                if v is None:
                    default_headers.pop(k, None)
                else:
                    default_headers[k] = v
        req.headers = default_headers

        captured_code = 0
        captured_data = {}
        captured_headers = {}

        def mock_send_response(code: int):
            nonlocal captured_code
            captured_code = code

        def mock_send_header(k: str, v: str):
            captured_headers[k] = v

        def mock_end_headers():
            pass

        req.send_response = mock_send_response
        req.send_header = mock_send_header
        req.end_headers = mock_end_headers
        req.log_message = MagicMock()

        # Bind methods to test
        req._send_json_response = lambda code, data, origin=None: GenerationHTTPRequestHandler._send_json_response(req, code, data, origin)
        req._validate_host_header = lambda: GenerationHTTPRequestHandler._validate_host_header(req)
        req._check_origin = lambda: GenerationHTTPRequestHandler._check_origin(req)

        if method == "GET":
            GenerationHTTPRequestHandler.do_GET(req)
        elif method == "POST":
            GenerationHTTPRequestHandler.do_POST(req)
        elif method == "OPTIONS":
            GenerationHTTPRequestHandler.do_OPTIONS(req)

        # Parse written response
        written = wfile.getvalue()
        if written:
            try:
                captured_data = json.loads(written.decode("utf-8"))
            except Exception:
                captured_data = {"raw": written.decode("utf-8", errors="replace")}

        return captured_code, captured_data, captured_headers

    # -------------------------------------------------------------------------
    # 1. Valid POST /train returns HTTP 202 and Job ID
    # -------------------------------------------------------------------------
    def test_valid_train_request_returns_202_and_job_id(self):
        code, data, _ = self._execute_request(
            "POST",
            "/train",
            body={"samples": self.valid_25_samples},
        )
        self.assertEqual(code, 202)
        self.assertEqual(data["status"], "accepted")
        self.assertTrue(data["jobId"].startswith("train_api_"))
        self.assertEqual(data["sampleCount"], 25)

        # Verify background thread was created and started
        time.sleep(0.05)
        self.mock_controller.run_training_job.assert_called_once()

    def test_api_train_alias_returns_202(self):
        code, data, _ = self._execute_request(
            "POST",
            "/api/train",
            body={"samples": self.valid_25_samples},
        )
        self.assertEqual(code, 202)
        self.assertEqual(data["status"], "accepted")

    # -------------------------------------------------------------------------
    # 2. Schema and Input Validation Failures (HTTP 400)
    # -------------------------------------------------------------------------
    def test_invalid_json_rejected(self):
        code, data, _ = self._execute_request(
            "POST",
            "/train",
            raw_body=b"{invalid_json: true,",
        )
        self.assertEqual(code, 400)
        self.assertIn("Malformed JSON", data["error"])

    def test_missing_samples_field_rejected(self):
        code, data, _ = self._execute_request(
            "POST",
            "/train",
            body={"wrong_field": "data"},
        )
        self.assertEqual(code, 400)
        self.assertIn("Field 'samples' is required", data["error"])

    def test_empty_samples_rejected(self):
        code, data, _ = self._execute_request(
            "POST",
            "/train",
            body={"samples": []},
        )
        self.assertEqual(code, 400)
        self.assertIn("cannot be empty", data["error"])

    def test_insufficient_samples_rejected(self):
        code, data, _ = self._execute_request(
            "POST",
            "/train",
            body={"samples": self.valid_25_samples[:10]},  # Only 10 samples
        )
        self.assertEqual(code, 400)
        self.assertIn("Insufficient training samples", data["error"])

    def test_disallowed_parameters_rejected(self):
        # Client cannot inject custom directories or hyperparameters
        code, data, _ = self._execute_request(
            "POST",
            "/train",
            body={
                "samples": self.valid_25_samples,
                "output_dir": "/malicious/path",
            },
        )
        self.assertEqual(code, 400)
        self.assertIn("Custom filesystem paths or training configurations are not permitted", data["error"])

    def test_excessive_sample_count_rejected(self):
        code, data, _ = self._execute_request(
            "POST",
            "/train",
            body={"samples": ["sample text"] * 501},
        )
        self.assertEqual(code, 400)
        self.assertIn("exceeds maximum limit of 500 items", data["error"])

    def test_oversized_payload_rejected_413(self):
        # Header claiming content length > 2 MB
        code, data, _ = self._execute_request(
            "POST",
            "/train",
            body={"samples": self.valid_25_samples},
            headers={"Content-Length": str(3 * 1024 * 1024)},
        )
        self.assertEqual(code, 413)
        self.assertIn("too large", data["error"])

    # -------------------------------------------------------------------------
    # 3. Security: Origin and Host Header Validation
    # -------------------------------------------------------------------------
    def test_untrusted_origin_rejected_403(self):
        code, data, headers = self._execute_request(
            "POST",
            "/train",
            body={"samples": self.valid_25_samples},
            headers={"Origin": "https://unauthorized-site.com"},
        )
        self.assertEqual(code, 403)
        self.assertIn("Forbidden", data["error"])
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_differing_chrome_extension_id_rejected_403(self):
        code, data, headers = self._execute_request(
            "POST",
            "/train",
            body={"samples": self.valid_25_samples},
            headers={"Origin": "chrome-extension://ponmlkjihgfedcbaponmlkjihgfedcba"},
        )
        self.assertEqual(code, 403)
        self.assertIn("Forbidden", data["error"])
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_custom_allowed_origin_succeeds_and_returns_cors_header(self):
        custom_origin = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
        gen_service_mod.active_allowed_origin = custom_origin

        code, data, headers = self._execute_request(
            "GET",
            "/training/status",
            headers={"Origin": custom_origin},
        )
        self.assertEqual(code, 200)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), custom_origin)

    def test_custom_allowed_origin_with_trailing_slash_in_header_succeeds(self):
        custom_origin = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
        gen_service_mod.active_allowed_origin = custom_origin

        code, data, headers = self._execute_request(
            "GET",
            "/training/status",
            headers={"Origin": f"{custom_origin}/"},
        )
        self.assertEqual(code, 200)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), custom_origin)

    def test_previous_default_origin_rejected_when_custom_origin_is_configured_403(self):
        custom_origin = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
        gen_service_mod.active_allowed_origin = custom_origin

        code, data, headers = self._execute_request(
            "GET",
            "/training/status",
            headers={"Origin": DEFAULT_ALLOWED_ORIGIN},
        )
        self.assertEqual(code, 403)
        self.assertIn("Forbidden", data["error"])
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_preflight_options_with_authorized_custom_origin_204(self):
        custom_origin = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
        gen_service_mod.active_allowed_origin = custom_origin

        code, data, headers = self._execute_request(
            "OPTIONS",
            "/train",
            headers={"Origin": custom_origin},
        )
        self.assertEqual(code, 204)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), custom_origin)
        self.assertIn("POST", headers.get("Access-Control-Allow-Methods", ""))

    def test_preflight_options_with_unauthorized_origin_403(self):
        code, data, headers = self._execute_request(
            "OPTIONS",
            "/train",
            headers={"Origin": "https://attacker-preflight.com"},
        )
        self.assertEqual(code, 403)
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_direct_loopback_client_without_origin_header_allowed(self):
        # Direct CLI/curl client without Origin header
        code, data, headers = self._execute_request(
            "GET",
            "/training/status",
            headers={"Origin": None},  # No Origin header
        )
        self.assertEqual(code, 200)
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_disallowed_host_header_rejected_400(self):
        code, data, _ = self._execute_request(
            "POST",
            "/train",
            body={"samples": self.valid_25_samples},
            headers={"Host": "evil.external-dns.com"},
        )
        self.assertEqual(code, 400)
        self.assertIn("Invalid or disallowed Host header", data["error"])

    # -------------------------------------------------------------------------
    # 4. Concurrency: Active Job Conflict (HTTP 409)
    # -------------------------------------------------------------------------
    def test_second_training_request_returns_409_conflict(self):
        # Mock active training state
        self.mock_controller.get_status.return_value = {
            "state": "training",
            "jobId": "train_v3_active_run",
            "sampleCount": 25,
            "targetVersion": "v3",
        }

        code, data, _ = self._execute_request(
            "POST",
            "/train",
            body={"samples": self.valid_25_samples},
        )
        self.assertEqual(code, 409)
        self.assertEqual(data["status"], "conflict")
        self.assertIn("A training job is already in progress", data["error"])
        self.assertEqual(data["activeJobId"], "train_v3_active_run")

    # -------------------------------------------------------------------------
    # 5. GET /training/status Lifecycle States and Error Sanitization
    # -------------------------------------------------------------------------
    def test_training_status_idle(self):
        self.mock_controller.get_status.return_value = {
            "state": "idle",
            "jobId": None,
            "targetVersion": None,
            "sampleCount": 0,
            "startTime": None,
            "endTime": None,
            "durationSec": 0.0,
            "errorMessage": None,
        }

        code, data, _ = self._execute_request("GET", "/training/status")
        self.assertEqual(code, 200)
        self.assertEqual(data["status"], "idle")
        self.assertIsNone(data["jobId"])
        self.assertEqual(data["sampleCount"], 0)
        self.assertIsNone(data["error"])

    def test_training_status_running(self):
        self.mock_controller.get_status.return_value = {
            "state": "training",
            "jobId": "train_v3_20261111",
            "targetVersion": "v3",
            "sampleCount": 25,
            "startTime": 1700000000.0,
            "endTime": None,
            "durationSec": 15.2,
            "errorMessage": None,
        }

        code, data, _ = self._execute_request("GET", "/training/status")
        self.assertEqual(code, 200)
        self.assertEqual(data["status"], "training")
        self.assertEqual(data["stage"], "training")
        self.assertEqual(data["jobId"], "train_v3_20261111")
        self.assertEqual(data["adapterVersion"], "v3")
        self.assertEqual(data["durationSec"], 15.2)

    def test_training_status_completed(self):
        self.mock_controller.get_status.return_value = {
            "state": "completed",
            "jobId": "train_v3_completed",
            "targetVersion": "v3",
            "sampleCount": 25,
            "startTime": 1700000000.0,
            "endTime": 1700000045.0,
            "durationSec": 45.0,
            "errorMessage": None,
        }

        code, data, _ = self._execute_request("GET", "/training/status")
        self.assertEqual(code, 200)
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["adapterVersion"], "v3")

    def test_training_status_error_sanitized(self):
        # Error with internal filesystem paths that must be stripped
        raw_error = (
            "File 'C:\\Users\\testuser\\AppData\\Local\\Temp\\weights.safetensors' failed to load: "
            "Corrupted header at /home/user/workspace/model.bin"
        )
        self.mock_controller.get_status.return_value = {
            "state": "error",
            "jobId": "train_v3_failed",
            "targetVersion": "v3",
            "sampleCount": 25,
            "startTime": 1700000000.0,
            "endTime": 1700000010.0,
            "durationSec": 10.0,
            "errorMessage": raw_error,
        }

        code, data, _ = self._execute_request("GET", "/training/status")
        self.assertEqual(code, 200)
        self.assertEqual(data["status"], "error")
        # Assert no local Windows drive or POSIX path is exposed
        self.assertNotIn("C:\\Users", data["error"])
        self.assertNotIn("/home/user", data["error"])
        self.assertIn("<path>", data["error"])

    # -------------------------------------------------------------------------
    # 6. Backward Compatibility: /health, /generate, and /reload
    # -------------------------------------------------------------------------
    def test_health_endpoint_still_functional(self):
        code, data, _ = self._execute_request("GET", "/health")
        self.assertEqual(code, 200)
        self.assertEqual(data["status"], "ready")
        self.assertEqual(data["model"], "Qwen/Qwen3.5-0.8B-Base")

    def test_generate_endpoint_still_functional(self):
        code, data, _ = self._execute_request(
            "POST",
            "/generate",
            body={"topic": "Personal coding journey"},
        )
        self.assertEqual(code, 200)
        self.assertEqual(data["completeDraft"], "Draft text")

    def test_reload_endpoint_still_functional(self):
        self.mock_gen.reload_adapter.return_value = {
            "status": "success",
            "activeAdapter": "training-output/qwen3.5-0.8b-lora-v2",
        }
        code, data, _ = self._execute_request(
            "POST",
            "/reload",
            body={"adapterPath": "training-output/qwen3.5-0.8b-lora-v2"},
        )
        self.assertEqual(code, 200)
        self.assertEqual(data["status"], "success")

    # -------------------------------------------------------------------------
    # 7. Lock Coordination: Generation and Training Respect Same Lock
    # -------------------------------------------------------------------------
    def test_generation_and_training_respect_same_lock(self):
        """
        Verifies that while generator_lock is held (as during training),
        POST /generate requests wait safely until released.
        """
        lock_held = False
        gen_lock = gen_service_mod.generator_lock

        # Acquire lock to simulate active training in progress
        gen_lock.acquire()

        generate_thread_completed = False

        def call_generate():
            nonlocal generate_thread_completed
            code, data, _ = self._execute_request(
                "POST",
                "/generate",
                body={"topic": "Lock testing"},
            )
            self.assertEqual(code, 200)
            generate_thread_completed = True

        t = threading.Thread(target=call_generate)
        t.start()

        # Brief delay; generate should be blocked waiting for gen_lock
        time.sleep(0.05)
        self.assertFalse(generate_thread_completed, "Generate must be blocked while generator_lock is held")

        # Release lock; generate thread should unblock and complete
        gen_lock.release()
        t.join(timeout=2.0)
        self.assertTrue(generate_thread_completed, "Generate must complete after generator_lock is released")


class TestHTTPConcurrencyAndJobSubmission(unittest.TestCase):
    """
    Verifies HTTP Server Concurrency (ThreadingHTTPServer) and Atomic Job Submission:
    1. /generate waiting on generator_lock does NOT block GET /training/status.
    2. Concurrent /train submissions atomically reserve slot: 1 receives 202, others receive 409.
    3. Thread creation failure safely rolls back reservation and returns sanitized HTTP 500.
    """

    def setUp(self):
        self.mock_gen = MagicMock()
        self.mock_gen.base_model_name = "Qwen/Qwen3.5-0.8B-Base"
        self.mock_gen.adapter_path = "training-output/qwen3.5-0.8b-lora-v2"
        self.mock_gen.active_version = "v2"
        self.mock_gen.load_time_sec = 1.0
        self.mock_gen.generate.return_value = {
            "complete_draft": "Concurrent generation output",
            "generated_text": "output",
            "lead_in": "",
            "tokens_generated": 10,
            "elapsed_sec": 0.1,
            "tokens_per_sec": 100.0,
            "peak_ram_mb": 150.0,
            "grounding_check": {"is_grounded": True, "flagged_issues": []},
        }

        self.mock_controller = MagicMock(spec=TrainingController)
        self.mock_controller.min_samples = 25
        self.mock_controller.get_status.return_value = {
            "state": "idle",
            "jobId": None,
            "targetVersion": None,
            "sampleCount": 0,
            "startTime": None,
            "endTime": None,
            "durationSec": 0.0,
            "errorMessage": None,
        }

        gen_service_mod.generator = self.mock_gen
        gen_service_mod.training_controller = self.mock_controller
        gen_service_mod.active_training_thread = None
        gen_service_mod.active_training_job_id = None
        gen_service_mod.is_model_ready = True
        gen_service_mod.active_allowed_origin = DEFAULT_ALLOWED_ORIGIN

        # Start real ThreadingHTTPServer on ephemeral loopback port
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), GenerationHTTPRequestHandler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

        self.valid_samples = [f"Sample text for concurrency testing {i:02d}" for i in range(25)]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=2.0)

        gen_service_mod.generator = None
        gen_service_mod.training_controller = None
        gen_service_mod.active_training_thread = None
        gen_service_mod.active_training_job_id = None
        gen_service_mod.is_model_ready = False

        if gen_service_mod.generator_lock.locked():
            try:
                gen_service_mod.generator_lock.release()
            except RuntimeError:
                pass

    def test_threaded_server_status_unblocked_while_generate_waiting(self):
        """
        Regression test:
        1. Holds generator_lock to simulate training.
        2. Sends a /generate request (blocks on generator_lock).
        3. Sends a /training/status request while generation is waiting.
        4. Verifies status returns successfully before the generation lock is released.
        5. Releases the lock and verifies generation completes.
        """
        gen_lock = gen_service_mod.generator_lock
        gen_lock.acquire()

        generate_completed = threading.Event()
        generate_result = {}

        def send_generate():
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
            headers = {
                "Origin": DEFAULT_ALLOWED_ORIGIN,
                "Host": f"127.0.0.1:{self.port}",
                "Content-Type": "application/json",
                "Connection": "close",
            }
            body = json.dumps({"topic": "Concurrency test topic"}).encode("utf-8")
            conn.request("POST", "/generate", body=body, headers=headers)
            resp = conn.getresponse()
            generate_result["status"] = resp.status
            generate_result["data"] = json.loads(resp.read().decode("utf-8"))
            conn.close()
            generate_completed.set()

        t_gen = threading.Thread(target=send_generate, daemon=True)
        t_gen.start()

        # Allow request to reach the server and block on generator_lock
        time.sleep(0.08)
        self.assertFalse(generate_completed.is_set(), "POST /generate must be blocked while generator_lock is held")

        # Send GET /training/status while /generate is blocked
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {
            "Origin": DEFAULT_ALLOWED_ORIGIN,
            "Host": f"127.0.0.1:{self.port}",
        }
        conn.request("GET", "/training/status", headers=headers)
        status_resp = conn.getresponse()
        self.assertEqual(status_resp.status, 200, "/training/status must respond with 200 while /generate is waiting")
        status_data = json.loads(status_resp.read().decode("utf-8"))
        conn.close()

        self.assertEqual(status_data["status"], "idle")
        # Verify /generate is STILL waiting (not completed)
        self.assertFalse(generate_completed.is_set(), "POST /generate must still be waiting on generator_lock")

        # Step 5: Release generator_lock and verify generation completes
        gen_lock.release()
        generate_completed.wait(timeout=3.0)
        self.assertTrue(generate_completed.is_set(), "POST /generate must complete after generator_lock is released")
        self.assertEqual(generate_result["status"], 200)
        self.assertEqual(generate_result["data"]["completeDraft"], "Concurrent generation output")

    def test_concurrent_http_train_submission_atomic_reservation(self):
        """
        Verifies atomic job submission when multiple concurrent HTTP requests arrive at /train:
        Exactly one request must reserve the slot and receive HTTP 202;
        All concurrent requests must receive HTTP 409 Conflict.
        """
        # Slow down training job so that the slot remains occupied during concurrent calls
        job_running_event = threading.Event()
        release_job_event = threading.Event()

        def slow_training_job(samples, job_id=None):
            job_running_event.set()
            release_job_event.wait(timeout=5.0)

        self.mock_controller.run_training_job.side_effect = slow_training_job

        num_requests = 10
        barrier = threading.Barrier(num_requests)
        results = []
        results_lock = threading.Lock()

        def send_train_request():
            barrier.wait()  # Synchronize all threads so they strike simultaneously
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
            headers = {
                "Origin": DEFAULT_ALLOWED_ORIGIN,
                "Host": f"127.0.0.1:{self.port}",
                "Content-Type": "application/json",
                "Connection": "close",
            }
            body = json.dumps({"samples": self.valid_samples}).encode("utf-8")
            try:
                conn.request("POST", "/train", body=body, headers=headers)
                resp = conn.getresponse()
                data = json.loads(resp.read().decode("utf-8"))
                with results_lock:
                    results.append((resp.status, data))
            finally:
                conn.close()

        threads = [threading.Thread(target=send_train_request, daemon=True) for _ in range(num_requests)]
        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=5.0)

        # Release the slow training job
        release_job_event.set()

        # Verify results
        status_codes = [status for status, _ in results]
        accepted = [data for status, data in results if status == 202]
        conflicts = [data for status, data in results if status == 409]

        self.assertEqual(len(results), num_requests, f"Expected {num_requests} responses, got {len(results)}")
        self.assertEqual(len(accepted), 1, f"Expected exactly 1 HTTP 202 Accepted, got {len(accepted)} ({status_codes})")
        self.assertEqual(len(conflicts), num_requests - 1, f"Expected {num_requests - 1} HTTP 409 Conflicts, got {len(conflicts)}")

        accepted_job_id = accepted[0]["jobId"]
        self.assertTrue(accepted_job_id.startswith("train_api_"))

        for c in conflicts:
            self.assertEqual(c["status"], "conflict")
            self.assertIn("already in progress", c["error"])
            self.assertEqual(c["activeJobId"], accepted_job_id)

    def test_atomic_reservation_rollback_on_worker_thread_failure(self):
        """
        Verifies that if background thread creation or startup fails,
        the atomic reservation is rolled back, active_training_job_id is cleared,
        a sanitized 500 error is returned, and subsequent requests can succeed.
        """
        original_start = threading.Thread.start

        def failing_start(thread_self, *args, **kwargs):
            if "EchoTrainingWorker" in getattr(thread_self, "name", ""):
                raise RuntimeError("OS resource exhaustion: Cannot allocate thread stack at C:\\internal\\kernel\\stack.c")
            return original_start(thread_self, *args, **kwargs)

        headers = {
            "Origin": DEFAULT_ALLOWED_ORIGIN,
            "Host": f"127.0.0.1:{self.port}",
            "Content-Type": "application/json",
        }
        body = json.dumps({"samples": self.valid_samples}).encode("utf-8")

        threading.Thread.start = failing_start
        try:
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
            conn.request("POST", "/train", body=body, headers=headers)
            resp = conn.getresponse()
            status_code = resp.status
            data = json.loads(resp.read().decode("utf-8"))
            conn.close()

            self.assertEqual(status_code, 500)
            self.assertIn("Failed to initialize training worker thread", data["error"])
            # Verify paths are sanitized
            self.assertNotIn("C:\\internal", data["error"])
            self.assertNotIn("stack.c", data["error"])
        finally:
            threading.Thread.start = original_start

        # Verify reservation was rolled back
        self.assertIsNone(gen_service_mod.active_training_job_id)
        self.assertIsNone(gen_service_mod.active_training_thread)

        # Verify subsequent request succeeds with 202
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", "/train", body=body, headers=headers)
        resp2 = conn.getresponse()
        status_code2 = resp2.status
        data2 = json.loads(resp2.read().decode("utf-8"))
        conn.close()

        self.assertEqual(status_code2, 202)
        self.assertEqual(data2["status"], "accepted")


class TestValidateAllowedOrigin(unittest.TestCase):
    """Unit tests for strict Chrome extension origin validation."""

    def test_accepts_default_chrome_extension_origin(self):
        result = validate_allowed_origin(DEFAULT_ALLOWED_ORIGIN)
        self.assertEqual(result, DEFAULT_ALLOWED_ORIGIN)

    def test_accepts_different_valid_chrome_extension_id(self):
        valid_other = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
        result = validate_allowed_origin(valid_other)
        self.assertEqual(result, valid_other)

    def test_rejects_ids_that_are_too_short_or_too_long(self):
        short_ids = [
            "chrome-extension://",
            "chrome-extension://a",
            "chrome-extension://abcdefghijklmnop",
            "chrome-extension://abcdefghijklmnopabcdefghijklmno",  # 31 chars
        ]
        for val in short_ids:
            with self.assertRaises(ValueError) as ctx:
                validate_allowed_origin(val)
            self.assertIn("invalid chrome extension origin format", str(ctx.exception).lower())

        long_ids = [
            "chrome-extension://abcdefghijklmnopabcdefghijklmnopx",  # 33 chars
            "chrome-extension://abcdefghijklmnopabcdefghijklmnopabcdefghijklmnop",  # 48 chars
        ]
        for val in long_ids:
            with self.assertRaises(ValueError) as ctx:
                validate_allowed_origin(val)
            self.assertIn("invalid chrome extension origin format", str(ctx.exception).lower())

    def test_rejects_uppercase_ids(self):
        uppercase_ids = [
            "chrome-extension://MBGHEFDHLGOLADJADPHNAGKICLMCKOHF",
            "chrome-extension://Abcdefghijklmnopabcdefghijklmnop",
            "chrome-extension://abcdefghijklmnopABCDEFGHIKLMNOPA",
        ]
        for val in uppercase_ids:
            with self.assertRaises(ValueError) as ctx:
                validate_allowed_origin(val)
            self.assertIn("invalid chrome extension origin format", str(ctx.exception).lower())

    def test_rejects_numeric_ids(self):
        numeric_ids = [
            "chrome-extension://12345678901234567890123456789012",
            "chrome-extension://abcdefghijklmnop1234567890abcdef",
        ]
        for val in numeric_ids:
            with self.assertRaises(ValueError) as ctx:
                validate_allowed_origin(val)
            self.assertIn("invalid chrome extension origin format", str(ctx.exception).lower())

    def test_rejects_ids_containing_characters_outside_a_to_p(self):
        out_of_range_ids = [
            "chrome-extension://qrstuvwxyzqrstuvwxyzqrstuvwxyz12",
            "chrome-extension://abcdefghijklmnopqrstuvwxyzabcdef",  # q-z
            "chrome-extension://abcdefghijklmnopabcdefghijklmnoz",  # 'z' is out of range
        ]
        for val in out_of_range_ids:
            with self.assertRaises(ValueError) as ctx:
                validate_allowed_origin(val)
            self.assertIn("invalid chrome extension origin format", str(ctx.exception).lower())

    def test_rejects_hyphens_and_underscores(self):
        hyphen_underscore_ids = [
            "chrome-extension://abcdefghijklmnop-abcdefghijklmno",
            "chrome-extension://abcdefghijklmnop_abcdefghijklmno",
            "chrome-extension://my_echo-extension_id_abcdefghij",
        ]
        for val in hyphen_underscore_ids:
            with self.assertRaises(ValueError) as ctx:
                validate_allowed_origin(val)
            self.assertIn("invalid chrome extension origin format", str(ctx.exception).lower())

    def test_rejects_mozilla_extension_origins(self):
        moz_origins = [
            "moz-extension://12345678-1234-1234-1234-123456789abc",
            "moz-extension://abcdefghijklmnopabcdefghijklmnop",
        ]
        for val in moz_origins:
            with self.assertRaises(ValueError) as ctx:
                validate_allowed_origin(val)
            self.assertIn("mozilla extension origin", str(ctx.exception).lower())

    def test_rejects_website_origins(self):
        web_origins = [
            "https://google.com",
            "http://localhost:3000",
            "http://127.0.0.1:8080",
            "ftp://evil.com",
            "file:///etc/passwd",
        ]
        for val in web_origins:
            with self.assertRaises(ValueError) as ctx:
                validate_allowed_origin(val)
            self.assertIn("web origin", str(ctx.exception).lower())

    def test_rejects_wildcard_origins(self):
        wildcards = ["*", "chrome-extension://*", "*://*"]
        for val in wildcards:
            with self.assertRaises(ValueError) as ctx:
                validate_allowed_origin(val)
            self.assertIn("wildcard", str(ctx.exception).lower())

    def test_strips_trailing_slash_normalization(self):
        origin = "chrome-extension://mbghefdhlgoladjadphnagkiclmckohf/"
        result = validate_allowed_origin(origin)
        self.assertEqual(result, DEFAULT_ALLOWED_ORIGIN)

        valid_other = "chrome-extension://abcdefghijklmnopabcdefghijklmnop/"
        result_other = validate_allowed_origin(valid_other)
        self.assertEqual(result_other, "chrome-extension://abcdefghijklmnopabcdefghijklmnop")

    def test_strips_surrounding_whitespace(self):
        origin = "   chrome-extension://mbghefdhlgoladjadphnagkiclmckohf   "
        result = validate_allowed_origin(origin)
        self.assertEqual(result, DEFAULT_ALLOWED_ORIGIN)

    def test_rejects_empty_and_whitespace(self):
        for val in ("", "   ", None):
            with self.assertRaises(ValueError) as ctx:
                validate_allowed_origin(val)
            self.assertIn("cannot be empty", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
