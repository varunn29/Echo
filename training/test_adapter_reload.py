"""
Echo — Adapter Versioning, Safe Reload & Concurrency Test Suite
Tests:
1. Initial V2 adapter metadata and directory validation.
2. Safe adapter reload and version switching.
3. Invalid adapter path rejection (missing dir, missing config, corrupted config, missing weights).
4. Rollback and active adapter preservation on load failure.
5. Concurrency safety between generation and reload.
6. Generation service HTTP endpoints (/health with version info, /reload, /generate).
7. Host header and origin security restrictions on /reload.
"""

import io
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    from training.generate_with_adapter import (
        DEFAULT_ADAPTER_PATH,
        PersonalizedWritingGenerator,
        load_active_adapter_metadata,
        save_active_adapter_metadata,
        validate_adapter_directory,
    )
    from training.generation_service import (
        DEFAULT_ALLOWED_ORIGIN,
        GenerationHTTPRequestHandler,
        generator_lock,
        state_lock,
    )
    import training.generation_service as gen_service_mod
except ImportError:
    from generate_with_adapter import (
        DEFAULT_ADAPTER_PATH,
        PersonalizedWritingGenerator,
        load_active_adapter_metadata,
        save_active_adapter_metadata,
        validate_adapter_directory,
    )
    from generation_service import (
        DEFAULT_ALLOWED_ORIGIN,
        GenerationHTTPRequestHandler,
        generator_lock,
        state_lock,
    )
    import generation_service as gen_service_mod


class TestAdapterMetadataAndValidation(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="echo_test_adapters_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_default_metadata_creation(self):
        """When metadata file is missing, defaults to initial V2 adapter."""
        nonexistent = Path(self.test_dir) / "missing_active_adapter.json"
        meta = load_active_adapter_metadata(
            metadata_path=nonexistent,
            default_adapter_path="training-output/qwen3.5-0.8b-lora-v2",
        )
        self.assertEqual(meta["active_adapter_path"], "training-output/qwen3.5-0.8b-lora-v2")
        self.assertEqual(meta["active_version"], "v2")
        self.assertEqual(meta["status"], "ready")
        self.assertIsNone(meta["previous_adapter_path"])

    def test_atomic_metadata_save_and_load(self):
        """Tests that saving metadata writes valid JSON atomically and reloads correctly."""
        meta_file = Path(self.test_dir) / "active_adapter.json"
        data = {
            "active_adapter_path": "training-output/qwen3.5-0.8b-lora-v3",
            "active_version": "v3",
            "activated_at": "2026-11-11 12:00:00",
            "previous_adapter_path": "training-output/qwen3.5-0.8b-lora-v2",
            "previous_version": "v2",
            "status": "ready",
        }
        save_active_adapter_metadata(data, metadata_path=meta_file)
        self.assertTrue(meta_file.exists())

        loaded = load_active_adapter_metadata(metadata_path=meta_file)
        self.assertEqual(loaded["active_adapter_path"], data["active_adapter_path"])
        self.assertEqual(loaded["active_version"], "v3")
        self.assertEqual(loaded["previous_version"], "v2")

    def test_validate_existing_v2_adapter(self):
        """Validates the live V2 adapter directory on disk."""
        v2_path = Path("training-output/qwen3.5-0.8b-lora-v2")
        self.assertTrue(v2_path.exists(), "Live V2 adapter must exist.")
        config = validate_adapter_directory(v2_path)
        self.assertEqual(config.get("base_model_name_or_path"), "Qwen/Qwen3.5-0.8B-Base")
        self.assertEqual(config.get("r"), 4)
        self.assertEqual(config.get("lora_alpha"), 8)

    def test_validate_nonexistent_directory(self):
        """Non-existent directory must raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            validate_adapter_directory(Path(self.test_dir) / "does_not_exist")

    def test_validate_missing_config(self):
        """Directory without adapter_config.json must raise FileNotFoundError."""
        empty_dir = Path(self.test_dir) / "no_config"
        empty_dir.mkdir()
        with self.assertRaises(FileNotFoundError):
            validate_adapter_directory(empty_dir)

    def test_validate_corrupted_config(self):
        """Corrupted JSON in adapter_config.json must raise ValueError."""
        bad_dir = Path(self.test_dir) / "bad_config"
        bad_dir.mkdir()
        (bad_dir / "adapter_config.json").write_text("{not: valid: json", encoding="utf-8")
        with self.assertRaises(ValueError):
            validate_adapter_directory(bad_dir)

    def test_validate_missing_weights(self):
        """Valid config but missing safetensors/bin weights must raise FileNotFoundError."""
        no_weights = Path(self.test_dir) / "no_weights"
        no_weights.mkdir()
        (no_weights / "adapter_config.json").write_text('{"peft_type": "LORA"}', encoding="utf-8")
        with self.assertRaises(FileNotFoundError):
            validate_adapter_directory(no_weights)


class TestGeneratorAdapterLifecycle(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="echo_test_gen_")
        self.meta_file = Path(self.test_dir) / "active_adapter.json"

        # Create mock candidate adapter v3
        self.v3_dir = Path(self.test_dir) / "qwen3.5-0.8b-lora-v3"
        self.v3_dir.mkdir()
        (self.v3_dir / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-0.8B-Base", "r": 4, "lora_alpha": 8}),
            encoding="utf-8",
        )
        (self.v3_dir / "adapter_model.safetensors").write_bytes(b"dummy_weights")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_mock_generator(self):
        gen = PersonalizedWritingGenerator(
            base_model_name="Qwen/Qwen3.5-0.8B-Base",
            adapter_path="training-output/qwen3.5-0.8b-lora-v2",
            metadata_path=self.meta_file,
        )
        # Mock PyTorch and PEFT components for lightweight, fast testing
        gen.base_model = MagicMock()
        mock_model = MagicMock()
        mock_model.peft_config = {"default": MagicMock()}
        gen.model = mock_model
        gen.tokenizer = MagicMock()
        gen.active_adapter_name = "default"
        gen.active_version = "v2"
        gen.last_reload_status = "ready"
        return gen

    def test_successful_reload_activates_candidate(self):
        """Candidate adapter is attached, set as active, metadata updated, and old adapter freed from memory."""
        gen = self._create_mock_generator()
        result = gen.reload_adapter(str(self.v3_dir), version="v3")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["activeAdapter"], str(self.v3_dir))
        self.assertEqual(result["activeVersion"], "v3")
        self.assertEqual(result["previousVersion"], "v2")
        self.assertEqual(gen.active_version, "v3")
        self.assertEqual(gen.adapter_path, str(self.v3_dir))
        self.assertEqual(gen.last_reload_status, "success")

        # Verify PEFT calls: loaded new adapter, set active, deleted old from memory
        gen.model.load_adapter.assert_called_once()
        gen.model.set_adapter.assert_called_once()
        gen.model.delete_adapter.assert_called_with("default")

        # Verify disk files of v2 were NOT touched
        self.assertTrue(Path("training-output/qwen3.5-0.8b-lora-v2").exists())

    def test_invalid_adapter_path_preserves_active(self):
        """Invalid adapter path raises error immediately and leaves current model state untouched."""
        gen = self._create_mock_generator()

        with self.assertRaises(FileNotFoundError):
            gen.reload_adapter(str(Path(self.test_dir) / "nonexistent"))

        self.assertEqual(gen.active_version, "v2")
        self.assertEqual(gen.adapter_path, "training-output/qwen3.5-0.8b-lora-v2")
        gen.model.load_adapter.assert_not_called()
        gen.model.set_adapter.assert_not_called()

    def test_failed_load_adapter_preserves_active(self):
        """If load_adapter fails unexpectedly, old adapter remains active and error is tracked."""
        gen = self._create_mock_generator()
        gen.model.load_adapter.side_effect = RuntimeError("Simulated corrupt weights")

        with self.assertRaises(RuntimeError) as ctx:
            gen.reload_adapter(str(self.v3_dir), version="v3")

        self.assertIn("Failed to activate candidate adapter", str(ctx.exception))

        # Active version and path preserved
        self.assertEqual(gen.active_version, "v2")
        self.assertEqual(gen.adapter_path, "training-output/qwen3.5-0.8b-lora-v2")
        self.assertEqual(gen.last_reload_status, "error")

        # Attempted fallback set_adapter back to default
        gen.model.set_adapter.assert_called_with("default")

    def test_metadata_save_failure_preserves_old_adapter_and_cleans_candidate(self):
        """If save_active_adapter_metadata fails after load_adapter, old adapter is kept active and candidate purged."""
        gen = self._create_mock_generator()

        with patch("training.generate_with_adapter.save_active_adapter_metadata", side_effect=IOError("Simulated disk full")):
            with self.assertRaises(RuntimeError) as ctx:
                gen.reload_adapter(str(self.v3_dir), version="v3")

            self.assertIn("Failed to activate candidate adapter", str(ctx.exception))

        # Active version and path preserved
        self.assertEqual(gen.active_version, "v2")
        self.assertEqual(gen.adapter_path, "training-output/qwen3.5-0.8b-lora-v2")
        self.assertEqual(gen.last_reload_status, "error")

        # Candidate adapter must be deleted from in-memory config
        delete_calls = [call[0][0] for call in gen.model.delete_adapter.call_args_list]
        self.assertTrue(any(name.startswith("adapter_v3_") for name in delete_calls))
        # Old adapter 'default' must NEVER have been deleted!
        self.assertNotIn("default", delete_calls)

        # Old adapter must be re-activated
        set_calls = [call[0][0] for call in gen.model.set_adapter.call_args_list]
        self.assertIn("default", set_calls)

    def test_activation_failure_restores_metadata_and_preserves_old_adapter(self):
        """If model.set_adapter fails after metadata write, candidate is purged, old adapter is kept, and metadata restored."""
        gen = self._create_mock_generator()

        # Seed initial metadata file for v2
        initial_meta = {
            "active_adapter_path": "training-output/qwen3.5-0.8b-lora-v2",
            "active_version": "v2",
            "activated_at": "2026-11-09 12:00:00",
            "status": "ready",
        }
        self.meta_file.write_text(json.dumps(initial_meta), encoding="utf-8")

        # Simulate set_adapter failing when activating candidate adapter
        def set_adapter_side_effect(name):
            if name.startswith("adapter_v3_"):
                raise RuntimeError("Simulated set_adapter failure")
            return None

        gen.model.set_adapter.side_effect = set_adapter_side_effect

        with self.assertRaises(RuntimeError) as ctx:
            gen.reload_adapter(str(self.v3_dir), version="v3")

        self.assertIn("Failed to activate candidate adapter", str(ctx.exception))
        self.assertEqual(gen.last_reload_status, "error")

        # 1. Previous adapter remains active in memory
        self.assertEqual(gen.active_version, "v2")
        self.assertEqual(gen.adapter_path, "training-output/qwen3.5-0.8b-lora-v2")
        self.assertEqual(gen.active_adapter_name, "default")

        # 2. Candidate adapter removed from memory
        delete_calls = [call[0][0] for call in gen.model.delete_adapter.call_args_list]
        self.assertTrue(any(name.startswith("adapter_v3_") for name in delete_calls))
        # 3. Old adapter weights never deleted
        self.assertNotIn("default", delete_calls)

        # 4. Old adapter re-activated
        set_calls = [call[0][0] for call in gen.model.set_adapter.call_args_list]
        self.assertIn("default", set_calls)

        # 5. Metadata on disk restored to previous adapter
        restored_meta = load_active_adapter_metadata(metadata_path=self.meta_file)
        self.assertEqual(restored_meta["active_adapter_path"], "training-output/qwen3.5-0.8b-lora-v2")
        self.assertEqual(restored_meta["active_version"], "v2")

    def test_activation_failure_with_metadata_restore_failure_reports_discrepancy(self):
        """If set_adapter fails AND metadata restoration fails, exception clearly reports the state discrepancy."""
        gen = self._create_mock_generator()

        # Seed initial metadata
        initial_meta = {
            "active_adapter_path": "training-output/qwen3.5-0.8b-lora-v2",
            "active_version": "v2",
            "activated_at": "2026-11-09 12:00:00",
            "status": "ready",
        }
        self.meta_file.write_text(json.dumps(initial_meta), encoding="utf-8")

        # Simulate set_adapter failing on candidate
        def set_adapter_side_effect(name):
            if name.startswith("adapter_v3_"):
                raise RuntimeError("Simulated set_adapter failure")
            return None

        gen.model.set_adapter.side_effect = set_adapter_side_effect

        call_count = [0]
        orig_save = save_active_adapter_metadata
        def mock_save(data, metadata_path=None):
            call_count[0] += 1
            if call_count[0] == 1:
                # Candidate metadata write succeeds
                orig_save(data, metadata_path=metadata_path)
            else:
                # Rollback metadata restore fails
                raise OSError("Disk full during rollback")

        with patch("training.generate_with_adapter.save_active_adapter_metadata", side_effect=mock_save):
            with self.assertRaises(RuntimeError) as ctx:
                gen.reload_adapter(str(self.v3_dir), version="v3")

            err_str = str(ctx.exception)
            self.assertIn("In-memory state and persisted active_adapter.json may differ", err_str)
            self.assertIn("Rollback issues", err_str)

        self.assertEqual(gen.last_reload_status, "error")
        self.assertEqual(gen.active_version, "v2")
        self.assertEqual(gen.adapter_path, "training-output/qwen3.5-0.8b-lora-v2")

    @patch("training.generate_with_adapter.AutoTokenizer.from_pretrained")
    @patch("training.generate_with_adapter.Qwen3_5ForCausalLM.from_pretrained")
    @patch("training.generate_with_adapter.PeftModel.from_pretrained")
    def test_startup_recovery_with_valid_metadata(self, mock_peft, mock_base, mock_tok):
        """Startup reads active_adapter.json and initializes with the recorded version."""
        meta_data = {
            "active_adapter_path": str(self.v3_dir),
            "active_version": "v3",
            "activated_at": "2026-11-11 12:00:00",
            "status": "ready",
        }
        self.meta_file.write_text(json.dumps(meta_data), encoding="utf-8")

        gen = PersonalizedWritingGenerator(
            base_model_name="Qwen/Qwen3.5-0.8B-Base",
            adapter_path="training-output/qwen3.5-0.8b-lora-v2",
            metadata_path=self.meta_file,
        )
        gen.load_model()

        self.assertEqual(gen.adapter_path, str(self.v3_dir))
        self.assertEqual(gen.active_version, "v3")
        self.assertEqual(gen.last_reload_status, "ready")

    @patch("training.generate_with_adapter.AutoTokenizer.from_pretrained")
    @patch("training.generate_with_adapter.Qwen3_5ForCausalLM.from_pretrained")
    @patch("training.generate_with_adapter.PeftModel.from_pretrained")
    def test_startup_recovery_with_missing_metadata_falls_back_to_v2(self, mock_peft, mock_base, mock_tok):
        """Startup without metadata file safely defaults to V2 adapter."""
        nonexistent = Path(self.test_dir) / "missing_meta.json"
        gen = PersonalizedWritingGenerator(
            base_model_name="Qwen/Qwen3.5-0.8B-Base",
            adapter_path="training-output/qwen3.5-0.8b-lora-v2",
            metadata_path=nonexistent,
        )
        gen.load_model()

        self.assertEqual(gen.adapter_path, "training-output/qwen3.5-0.8b-lora-v2")
        self.assertEqual(gen.active_version, "v2")

    @patch("training.generate_with_adapter.AutoTokenizer.from_pretrained")
    @patch("training.generate_with_adapter.Qwen3_5ForCausalLM.from_pretrained")
    @patch("training.generate_with_adapter.PeftModel.from_pretrained")
    def test_startup_recovery_with_malformed_metadata_falls_back_to_v2(self, mock_peft, mock_base, mock_tok):
        """Startup with malformed metadata file safely falls back to V2 adapter without error."""
        self.meta_file.write_text("not { valid json content", encoding="utf-8")
        gen = PersonalizedWritingGenerator(
            base_model_name="Qwen/Qwen3.5-0.8B-Base",
            adapter_path="training-output/qwen3.5-0.8b-lora-v2",
            metadata_path=self.meta_file,
        )
        gen.load_model()

        self.assertEqual(gen.adapter_path, "training-output/qwen3.5-0.8b-lora-v2")
        self.assertEqual(gen.active_version, "v2")

    @patch("training.generate_with_adapter.AutoTokenizer.from_pretrained")
    @patch("training.generate_with_adapter.Qwen3_5ForCausalLM.from_pretrained")
    @patch("training.generate_with_adapter.PeftModel.from_pretrained")
    def test_startup_recovery_with_invalid_adapter_directory_falls_back_to_v2(self, mock_peft, mock_base, mock_tok):
        """Startup pointing to non-existent adapter directory safely falls back to V2 adapter."""
        invalid_meta = {
            "active_adapter_path": str(Path(self.test_dir) / "corrupted_adapter_v99"),
            "active_version": "v99",
            "status": "ready",
        }
        self.meta_file.write_text(json.dumps(invalid_meta), encoding="utf-8")

        gen = PersonalizedWritingGenerator(
            base_model_name="Qwen/Qwen3.5-0.8B-Base",
            adapter_path="training-output/qwen3.5-0.8b-lora-v2",
            metadata_path=self.meta_file,
        )
        gen.load_model()

        self.assertEqual(gen.adapter_path, "training-output/qwen3.5-0.8b-lora-v2")
        self.assertEqual(gen.active_version, "v2")


class TestConcurrencySafety(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="echo_test_concurrency_")
        self.v3_dir = Path(self.test_dir) / "qwen3.5-0.8b-lora-v3"
        self.v3_dir.mkdir()
        (self.v3_dir / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-0.8B-Base", "r": 4, "lora_alpha": 8}),
            encoding="utf-8",
        )
        (self.v3_dir / "adapter_model.safetensors").write_bytes(b"dummy_weights")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_concurrent_generation_and_reload_are_serialized(self):
        """Ensures reload cannot execute while generation is running under generator_lock."""
        execution_order = []

        def slow_generation():
            with generator_lock:
                execution_order.append("gen_start")
                time.sleep(0.05)  # Simulate generation latency
                execution_order.append("gen_end")

        def reload_call():
            time.sleep(0.01)  # Ensure generation thread starts first
            with generator_lock:
                execution_order.append("reload_executed")

        t1 = threading.Thread(target=slow_generation)
        t2 = threading.Thread(target=reload_call)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        # Reload must strictly execute after generation finishes, never interleaved
        self.assertEqual(execution_order, ["gen_start", "gen_end", "reload_executed"])


class MockServer:
    server_port = 11435


class TestGenerationServiceHTTP(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="echo_test_http_")
        self.v3_dir = Path(self.test_dir) / "qwen3.5-0.8b-lora-v3"
        self.v3_dir.mkdir()
        (self.v3_dir / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-0.8B-Base", "r": 4, "lora_alpha": 8}),
            encoding="utf-8",
        )
        (self.v3_dir / "adapter_model.safetensors").write_bytes(b"dummy_weights")

        # Mock global generator
        self.mock_gen = MagicMock()
        self.mock_gen.base_model_name = "Qwen/Qwen3.5-0.8B-Base"
        self.mock_gen.adapter_path = "training-output/qwen3.5-0.8b-lora-v2"
        self.mock_gen.active_version = "v2"
        self.mock_gen.last_reload_status = "ready"
        self.mock_gen.last_reload_time = None
        self.mock_gen.previous_adapter_path = None
        self.mock_gen.load_time_sec = 2.5
        self.mock_gen.process = MagicMock()
        self.mock_gen.process.memory_info.return_value.rss = 1024 * 1024 * 1500

        gen_service_mod.generator = self.mock_gen
        gen_service_mod.is_model_ready = True
        gen_service_mod.active_allowed_origin = DEFAULT_ALLOWED_ORIGIN

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        gen_service_mod.generator = None
        gen_service_mod.is_model_ready = False

    def _execute_request(self, method: str, path: str, body: dict = None, headers: dict = None):
        """Simulates an HTTP request against GenerationHTTPRequestHandler without binding sockets."""
        body_bytes = json.dumps(body).encode("utf-8") if body is not None else b""
        rfile = io.BytesIO(body_bytes)
        wfile = io.BytesIO()

        req = MagicMock(spec=GenerationHTTPRequestHandler)
        req.server = MockServer()
        req.command = method
        req.path = path
        req.rfile = rfile
        req.wfile = wfile

        default_headers = {
            "Host": "127.0.0.1:11435",
            "Origin": DEFAULT_ALLOWED_ORIGIN,
            "Content-Length": str(len(body_bytes)),
        }
        if headers:
            default_headers.update(headers)
        req.headers = default_headers

        # Bind methods
        req.log_message = MagicMock()
        req._send_json_response = lambda code, data, origin=None: GenerationHTTPRequestHandler._send_json_response(req, code, data, origin)
        req._validate_host_header = lambda: GenerationHTTPRequestHandler._validate_host_header(req)
        req._check_origin = lambda: GenerationHTTPRequestHandler._check_origin(req)

        # Intercept written response
        status_code = [None]
        response_data = [None]

        orig_send_response = req.send_response
        def capture_send_response(code):
            status_code[0] = code
        req.send_response = capture_send_response

        def capture_write(data):
            try:
                response_data[0] = json.loads(data.decode("utf-8"))
            except Exception:
                pass
        req.wfile.write = capture_write

        if method == "GET":
            GenerationHTTPRequestHandler.do_GET(req)
        elif method == "POST":
            GenerationHTTPRequestHandler.do_POST(req)
        elif method == "OPTIONS":
            GenerationHTTPRequestHandler.do_OPTIONS(req)

        return status_code[0], response_data[0]

    def test_health_reports_active_version(self):
        """GET /health reports status, model, adapter, and active adapterVersion."""
        code, data = self._execute_request("GET", "/health")
        self.assertEqual(code, 200)
        self.assertEqual(data["status"], "ready")
        self.assertEqual(data["adapterVersion"], "v2")
        self.assertEqual(data["adapter"], "training-output/qwen3.5-0.8b-lora-v2")
        self.assertEqual(data["lastReloadStatus"], "ready")

    def test_post_reload_success(self):
        """POST /reload switches adapter safely and returns 200."""
        self.mock_gen.reload_adapter.return_value = {
            "status": "success",
            "message": "Adapter successfully switched to v3",
            "activeAdapter": str(self.v3_dir),
            "activeVersion": "v3",
            "previousAdapter": "training-output/qwen3.5-0.8b-lora-v2",
            "previousVersion": "v2",
            "reloadTimeSec": 0.05,
        }

        code, data = self._execute_request(
            "POST",
            "/reload",
            body={"adapterPath": str(self.v3_dir), "version": "v3"},
        )
        self.assertEqual(code, 200)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["activeVersion"], "v3")
        self.mock_gen.reload_adapter.assert_called_once_with(str(self.v3_dir), version="v3")

    def test_post_reload_invalid_path_returns_400(self):
        """POST /reload with missing directory returns 400 with active adapter preserved."""
        self.mock_gen.reload_adapter.side_effect = FileNotFoundError("Adapter directory does not exist")

        code, data = self._execute_request(
            "POST",
            "/reload",
            body={"adapterPath": "nonexistent/dir"},
        )
        self.assertEqual(code, 400)
        self.assertEqual(data["status"], "error")
        self.assertIn("does not exist", data["error"])
        self.assertEqual(data["activeVersion"], "v2")

    def test_post_reload_unauthorized_origin_returns_403(self):
        """POST /reload from untrusted foreign origin returns 403 Forbidden."""
        code, data = self._execute_request(
            "POST",
            "/reload",
            body={"adapterPath": str(self.v3_dir)},
            headers={"Origin": "https://malicious-website.com"},
        )
        self.assertEqual(code, 403)
        self.mock_gen.reload_adapter.assert_not_called()

    def test_post_reload_disallowed_host_returns_400(self):
        """POST /reload with DNS rebinding host returns 400 Bad Request."""
        code, data = self._execute_request(
            "POST",
            "/reload",
            body={"adapterPath": str(self.v3_dir)},
            headers={"Host": "attacker.com"},
        )
        self.assertEqual(code, 400)
        self.mock_gen.reload_adapter.assert_not_called()

    def test_generate_endpoint_continues_working(self):
        """Existing POST /generate endpoint works normally alongside /reload."""
        self.mock_gen.generate.return_value = {
            "complete_draft": "I started learning React and build projects with it.",
            "generated_text": "I started learning React and build projects with it.",
            "lead_in": "",
            "tokens_generated": 10,
            "elapsed_sec": 0.2,
            "tokens_per_sec": 50.0,
            "peak_ram_mb": 1500.0,
            "grounding_check": {"is_grounded": True, "flagged_issues": []},
        }

        code, data = self._execute_request(
            "POST",
            "/generate",
            body={"topic": "Web development journey"},
        )
        self.assertEqual(code, 200)
        self.assertIn("completeDraft", data)
        self.assertEqual(data["metadata"]["adapter"], "training-output/qwen3.5-0.8b-lora-v2")


if __name__ == "__main__":
    unittest.main()
