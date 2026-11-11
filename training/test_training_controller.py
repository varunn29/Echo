"""
Echo — Training Controller and Atomic Adapter Versioning Unit Tests (Phase 1, Step 4)

Covers all lifecycle states, version allocation, failure preservation,
readiness threshold consistency, atomic reservations, generation lock deadlock audit,
and concurrency safety without loading real models.
"""

import json
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from training.generate_with_adapter import (
    save_active_adapter_metadata,
    load_active_adapter_metadata,
)
from training.training_controller import (
    TrainingController,
    TrainingLifecycleState,
    DEFAULT_MIN_SAMPLES,
    resolve_next_version,
    reserve_next_version,
    normalize_samples,
    write_training_dataset,
    write_training_manifest,
)


def create_dummy_adapter_dir(path: Path, version: str = "v3") -> None:
    """Helper to create a valid mock adapter directory for validation tests."""
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "base_model_name_or_path": "Qwen/Qwen3.5-0.8B-Base",
        "peft_type": "LORA",
        "r": 4,
        "lora_alpha": 8,
        "target_modules": ["q_proj", "v_proj"],
        "version": version,
    }
    (path / "adapter_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"dummy_lora_weights_bytes")


class TestTrainingController(unittest.TestCase):

    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.output_root = Path(self.temp_dir.name)
        self.metadata_path = self.output_root / "active_adapter.json"

        # Seed an initial active adapter (v2) in the fixture
        self.v2_dir = self.output_root / "qwen3.5-0.8b-lora-v2"
        create_dummy_adapter_dir(self.v2_dir, version="v2")
        save_active_adapter_metadata(
            {
                "active_adapter_path": str(self.v2_dir),
                "active_version": "v2",
                "activated_at": "2026-11-09 12:00:00",
                "status": "ready",
            },
            metadata_path=self.metadata_path,
        )

        self.valid_samples = [
            "We built an indexed database architecture for offline-first local storage.",
            "TypeScript interfaces keep our data contracts strictly defined across modules.",
            "Decoupled state management ensures the background worker remains lightweight.",
            "Clean architecture separates UI components from database query logic.",
            "Local AI inference eliminates network latency for personalized writing drafts.",
        ]

        # 25 samples for full readiness threshold testing
        self.valid_25_samples = [
            f"Personal engineering conviction sample #{i:02d}: Architectural consistency and robust error handling."
            for i in range(25)
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # 1. Valid samples produce a training dataset & manifest
    # -------------------------------------------------------------------------
    def test_valid_samples_produce_training_dataset_and_manifest(self):
        def mock_train(dataset_path: Path, output_dir: Path, **kwargs) -> Dict[str, Any]:
            create_dummy_adapter_dir(output_dir)
            return {"training_time_seconds": 1.5, "samples": 5}

        controller = TrainingController(
            output_root=self.output_root,
            min_samples=5,
            training_fn=mock_train,
            metadata_path=self.metadata_path,
        )

        result = controller.run_training_job(self.valid_samples)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["version"], "v3")

        # Verify dataset JSONL file exists and contains valid samples
        dataset_path = Path(result["datasetPath"])
        self.assertTrue(dataset_path.exists())
        lines = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(lines), 5)
        self.assertEqual(lines[0]["text"], self.valid_samples[0])

        # Verify manifest JSON file exists and contains metadata
        manifest_path = Path(result["manifestPath"])
        self.assertTrue(manifest_path.exists())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["jobId"], result["jobId"])
        self.assertEqual(manifest["targetVersion"], "v3")
        self.assertEqual(manifest["sampleCount"], 5)
        self.assertEqual(manifest["status"], "completed")

    # -------------------------------------------------------------------------
    # 2. Training readiness threshold consistency
    # -------------------------------------------------------------------------
    def test_default_controller_threshold_matches_readiness_foundation(self):
        controller = TrainingController(
            output_root=self.output_root,
            metadata_path=self.metadata_path,
        )
        self.assertEqual(DEFAULT_MIN_SAMPLES, 25)
        self.assertEqual(controller.min_samples, 25)

    def test_below_threshold_dataset_rejected_by_default(self):
        controller = TrainingController(
            output_root=self.output_root,
            metadata_path=self.metadata_path,
        )
        # 24 samples (below the default 25 threshold)
        samples_24 = self.valid_25_samples[:24]
        with self.assertRaises(ValueError) as ctx:
            controller.run_training_job(samples_24)
        self.assertIn("Insufficient training samples: got 24, minimum required is 25", str(ctx.exception))
        self.assertEqual(controller.get_status()["state"], TrainingLifecycleState.ERROR.value)

    def test_explicitly_configured_threshold_accepted(self):
        def mock_train(dataset_path: Path, output_dir: Path, **kwargs) -> Dict[str, Any]:
            create_dummy_adapter_dir(output_dir)
            return {"samples": 5}

        # Override threshold to 5
        controller = TrainingController(
            output_root=self.output_root,
            min_samples=5,
            training_fn=mock_train,
            metadata_path=self.metadata_path,
        )
        result = controller.run_training_job(self.valid_samples)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["sampleCount"], 5)

    def test_empty_samples_rejected(self):
        controller = TrainingController(
            output_root=self.output_root,
            min_samples=5,
            metadata_path=self.metadata_path,
        )

        with self.assertRaises(ValueError) as ctx:
            controller.run_training_job([])
        self.assertIn("cannot be empty", str(ctx.exception))
        self.assertEqual(controller.get_status()["state"], TrainingLifecycleState.ERROR.value)

    def test_whitespace_and_empty_strings_filtered(self):
        controller = TrainingController(
            output_root=self.output_root,
            min_samples=3,
            metadata_path=self.metadata_path,
        )
        samples_with_blanks = ["Valid 1", "   ", "", "\n\t", "Valid 2"]
        with self.assertRaises(ValueError) as ctx:
            controller.run_training_job(samples_with_blanks)
        self.assertIn("got 2, minimum required is 3", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 3. New training runs receive unique version directories & atomic reservation
    # -------------------------------------------------------------------------
    def test_unique_version_directory_allocation(self):
        version1, dir1 = resolve_next_version(self.output_root, metadata_path=self.metadata_path)
        self.assertEqual(version1, "v3")
        self.assertEqual(dir1.name, "qwen3.5-0.8b-lora-v3")

        # If v3 directory is now created on disk, next resolution should yield v4
        create_dummy_adapter_dir(dir1, version="v3")
        version2, dir2 = resolve_next_version(self.output_root, metadata_path=self.metadata_path)
        self.assertEqual(version2, "v4")
        self.assertEqual(dir2.name, "qwen3.5-0.8b-lora-v4")

    def test_atomic_directory_reservation_prevents_directory_collision(self):
        # Concurrently reserve 5 versions using reserve_next_version
        def reserve_one():
            return reserve_next_version(
                output_root=self.output_root,
                metadata_path=self.metadata_path,
            )

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(reserve_one) for _ in range(5)]
            results = [f.result() for f in futures]

        versions = [r[0] for r in results]
        dirs = [r[1] for r in results]

        # Verify all 5 versions are strictly unique (e.g. v3, v4, v5, v6, v7)
        self.assertEqual(len(set(versions)), 5)
        self.assertEqual(len(set(dirs)), 5)

        # Verify all directories exist on disk and contain reservation markers
        for d in dirs:
            self.assertTrue(d.exists())
            self.assertTrue((d / ".reservation").exists())

    # -------------------------------------------------------------------------
    # 4. Existing adapter directories are never overwritten
    # -------------------------------------------------------------------------
    def test_existing_adapter_directory_never_overwritten(self):
        # Manually create v3 directory
        v3_dir = self.output_root / "qwen3.5-0.8b-lora-v3"
        create_dummy_adapter_dir(v3_dir, version="v3")
        marker_file = v3_dir / "original_file.txt"
        marker_file.write_text("must_not_be_overwritten", encoding="utf-8")

        controller = TrainingController(
            output_root=self.output_root,
            min_samples=5,
            metadata_path=self.metadata_path,
        )

        # Attempt to explicitly train into v3
        with self.assertRaises(FileExistsError) as ctx:
            controller.run_training_job(self.valid_samples, version="v3")

        self.assertIn("already exists and cannot be overwritten", str(ctx.exception))
        self.assertTrue(marker_file.exists())
        self.assertEqual(marker_file.read_text(encoding="utf-8"), "must_not_be_overwritten")
        self.assertEqual(controller.get_status()["state"], TrainingLifecycleState.ERROR.value)

    def test_cannot_train_into_currently_active_adapter(self):
        controller = TrainingController(
            output_root=self.output_root,
            min_samples=5,
            metadata_path=self.metadata_path,
        )

        # Active adapter is v2
        with self.assertRaises(Exception) as ctx:
            controller.run_training_job(self.valid_samples, version="v2")
        self.assertTrue(
            isinstance(ctx.exception, (FileExistsError, ValueError)),
            f"Expected FileExistsError or ValueError, got {type(ctx.exception)}",
        )

    # -------------------------------------------------------------------------
    # 5. Training failure preserves active adapter
    # -------------------------------------------------------------------------
    def test_training_failure_preserves_active_adapter(self):
        def failing_train(dataset_path: Path, output_dir: Path, **kwargs):
            raise RuntimeError("CUDA out of memory during simulated training")

        controller = TrainingController(
            output_root=self.output_root,
            min_samples=5,
            training_fn=failing_train,
            metadata_path=self.metadata_path,
        )

        with self.assertRaises(RuntimeError) as ctx:
            controller.run_training_job(self.valid_samples)

        self.assertIn("Training execution failed", str(ctx.exception))
        self.assertEqual(controller.get_status()["state"], TrainingLifecycleState.ERROR.value)

        # Active adapter metadata is untouched
        active_meta = load_active_adapter_metadata(metadata_path=self.metadata_path)
        self.assertEqual(active_meta["active_version"], "v2")
        self.assertEqual(active_meta["active_adapter_path"], str(self.v2_dir))

    # -------------------------------------------------------------------------
    # 6. Validation failure preserves active adapter
    # -------------------------------------------------------------------------
    def test_validation_failure_preserves_active_adapter(self):
        def empty_training_output(dataset_path: Path, output_dir: Path, **kwargs):
            # Leaves directory missing adapter weights and config
            return {}

        controller = TrainingController(
            output_root=self.output_root,
            min_samples=5,
            training_fn=empty_training_output,
            metadata_path=self.metadata_path,
        )

        with self.assertRaises(RuntimeError) as ctx:
            controller.run_training_job(self.valid_samples)

        self.assertIn("Trained adapter validation failed", str(ctx.exception))
        self.assertEqual(controller.get_status()["state"], TrainingLifecycleState.ERROR.value)

        # Active adapter remains untouched
        active_meta = load_active_adapter_metadata(metadata_path=self.metadata_path)
        self.assertEqual(active_meta["active_version"], "v2")
        self.assertEqual(active_meta["active_adapter_path"], str(self.v2_dir))

    # -------------------------------------------------------------------------
    # 7. Successful training invokes safe adapter activation
    # -------------------------------------------------------------------------
    def test_successful_training_invokes_safe_adapter_activation(self):
        reload_mock = MagicMock(return_value={"status": "success"})

        def mock_train(dataset_path: Path, output_dir: Path, **kwargs):
            create_dummy_adapter_dir(output_dir, version="v3")
            return {"epochs": 3}

        controller = TrainingController(
            output_root=self.output_root,
            min_samples=5,
            training_fn=mock_train,
            reload_callback=reload_mock,
            metadata_path=self.metadata_path,
        )

        result = controller.run_training_job(self.valid_samples)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["version"], "v3")

        # Verify reload callback was called with new adapter path and version
        target_dir = self.output_root / "qwen3.5-0.8b-lora-v3"
        reload_mock.assert_called_once_with(str(target_dir), "v3")

        status = controller.get_status()
        self.assertEqual(status["state"], TrainingLifecycleState.COMPLETED.value)
        self.assertEqual(status["activeAdapterAfter"], str(target_dir))

    # -------------------------------------------------------------------------
    # 8. Activation failure is reported correctly & preserves active state
    # -------------------------------------------------------------------------
    def test_activation_failure_reported_correctly(self):
        def mock_train(dataset_path: Path, output_dir: Path, **kwargs):
            create_dummy_adapter_dir(output_dir, version="v3")
            return {}

        def failing_reload(adapter_path: str, version: Optional[str]):
            raise RuntimeError("Simulated activation crash (PEFT set_adapter failure)")

        controller = TrainingController(
            output_root=self.output_root,
            min_samples=5,
            training_fn=mock_train,
            reload_callback=failing_reload,
            metadata_path=self.metadata_path,
        )

        with self.assertRaises(RuntimeError) as ctx:
            controller.run_training_job(self.valid_samples)

        self.assertIn("Adapter activation failed", str(ctx.exception))
        status = controller.get_status()
        self.assertEqual(status["state"], TrainingLifecycleState.ERROR.value)
        self.assertIn("Simulated activation crash", status["errorMessage"])

        # Manifest records error
        manifest_path = Path(status["manifestPath"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "error")
        self.assertIn("Simulated activation crash", manifest["error"])

    # -------------------------------------------------------------------------
    # 9. Concurrent training requests & cross-instance safety
    # -------------------------------------------------------------------------
    def test_concurrent_training_requests_rejected(self):
        slow_event = threading.Event()

        def slow_train(dataset_path: Path, output_dir: Path, **kwargs):
            create_dummy_adapter_dir(output_dir, version="v3")
            slow_event.wait(timeout=2.0)
            return {}

        controller = TrainingController(
            output_root=self.output_root,
            min_samples=5,
            training_fn=slow_train,
            metadata_path=self.metadata_path,
        )

        t1_error = None
        t2_error = None

        def run_job1():
            nonlocal t1_error
            try:
                controller.run_training_job(self.valid_samples)
            except Exception as e:
                t1_error = e

        def run_job2():
            nonlocal t2_error
            try:
                controller.run_training_job(self.valid_samples)
            except Exception as e:
                t2_error = e

        thread1 = threading.Thread(target=run_job1)
        thread1.start()

        time.sleep(0.05)

        thread2 = threading.Thread(target=run_job2)
        thread2.start()
        thread2.join()

        slow_event.set()
        thread1.join()

        self.assertIsNone(t1_error)
        self.assertIsNotNone(t2_error)
        self.assertIsInstance(t2_error, RuntimeError)
        self.assertIn("A training job is already in progress", str(t2_error))

    def test_concurrent_controller_instances_rejected_by_process_lock(self):
        slow_event = threading.Event()

        def slow_train(dataset_path: Path, output_dir: Path, **kwargs):
            create_dummy_adapter_dir(output_dir, version="v3")
            slow_event.wait(timeout=2.0)
            return {}

        c1 = TrainingController(
            output_root=self.output_root,
            min_samples=5,
            training_fn=slow_train,
            metadata_path=self.metadata_path,
        )
        c2 = TrainingController(
            output_root=self.output_root,
            min_samples=5,
            training_fn=slow_train,
            metadata_path=self.metadata_path,
        )

        c1_error = None
        c2_error = None

        def run_c1():
            nonlocal c1_error
            try:
                c1.run_training_job(self.valid_samples)
            except Exception as e:
                c1_error = e

        def run_c2():
            nonlocal c2_error
            try:
                c2.run_training_job(self.valid_samples)
            except Exception as e:
                c2_error = e

        t1 = threading.Thread(target=run_c1)
        t1.start()

        time.sleep(0.05)

        t2 = threading.Thread(target=run_c2)
        t2.start()
        t2.join()

        slow_event.set()
        t1.join()

        self.assertIsNone(c1_error)
        self.assertIsNotNone(c2_error)
        self.assertIsInstance(c2_error, RuntimeError)
        self.assertIn("A training job is already in progress", str(c2_error))

    # -------------------------------------------------------------------------
    # 10. Generation lock deadlock audit & service lock arrangement
    # -------------------------------------------------------------------------
    def test_generation_lock_single_owner_regression_with_service_lock_arrangement(self):
        """
        Regression test proving that a complete mocked training job reaches 'completed'
        using the exact non-reentrant threading.Lock arrangement from generation_service.py.
        Proves:
        - Training holds generator_lock without deadlocking on generator.reload_adapter().
        - Concurrent generation requests wait safely until training completes.
        - Once training finishes, generation executes successfully using the updated state.
        - Both locks are completely released afterwards.
        """
        generator_lock = threading.Lock()  # Regular non-reentrant Lock as in generation_service.py
        mock_generator = MagicMock()
        mock_generator.reload_adapter.return_value = {
            "status": "success",
            "activeAdapter": "v3",
        }
        mock_generator.generate.return_value = {"draft": "Personalized draft generated with v3"}

        training_started = threading.Event()
        generation_waited = False
        generation_completed = False

        def mock_train(dataset_path: Path, output_dir: Path, **kwargs):
            create_dummy_adapter_dir(output_dir, version="v3")
            training_started.set()
            time.sleep(0.15)  # Simulate model training under lock
            return {"loss": 0.35}

        controller = TrainingController(
            output_root=self.output_root,
            min_samples=5,
            generator=mock_generator,
            generation_lock=generator_lock,
            training_fn=mock_train,
            metadata_path=self.metadata_path,
        )

        def simulate_incoming_generate_request():
            nonlocal generation_waited, generation_completed
            training_started.wait(timeout=2.0)
            # Try to acquire without waiting; must fail because training holds it
            if not generator_lock.acquire(blocking=False):
                generation_waited = True
            else:
                generator_lock.release()

            # Now block on generator_lock as generation_service.py do_POST does
            with generator_lock:
                mock_generator.generate(topic="Test topic")
                generation_completed = True

        gen_thread = threading.Thread(target=simulate_incoming_generate_request)
        gen_thread.start()

        # Run training job (holds generator_lock throughout training, validating, activating)
        result = controller.run_training_job(self.valid_samples)

        gen_thread.join(timeout=3.0)

        # Assertions
        self.assertEqual(result["status"], "completed")
        self.assertEqual(controller.get_status()["state"], TrainingLifecycleState.COMPLETED.value)
        mock_generator.reload_adapter.assert_called_once()
        self.assertTrue(generation_waited, "Generation request must wait while training holds generator_lock")
        self.assertTrue(generation_completed, "Generation request must complete after training releases lock")
        mock_generator.generate.assert_called_once_with(topic="Test topic")

        # Verify both locks are completely released
        self.assertTrue(generator_lock.acquire(blocking=False))
        generator_lock.release()
        self.assertFalse(controller._training_lock.locked())
        self.assertFalse(TrainingController._process_training_lock.locked())

    def test_generation_lock_released_on_training_failure(self):
        """Proves that generator_lock is unconditionally released if training raises an exception."""
        generator_lock = threading.Lock()

        def failing_train(dataset_path: Path, output_dir: Path, **kwargs):
            raise RuntimeError("Fatal training crash")

        controller = TrainingController(
            output_root=self.output_root,
            min_samples=5,
            generation_lock=generator_lock,
            training_fn=failing_train,
            metadata_path=self.metadata_path,
        )

        with self.assertRaises(RuntimeError):
            controller.run_training_job(self.valid_samples)

        # generator_lock must be unlocked
        self.assertTrue(generator_lock.acquire(blocking=False))
        generator_lock.release()
        self.assertFalse(controller._training_lock.locked())
        self.assertFalse(TrainingController._process_training_lock.locked())

    # -------------------------------------------------------------------------
    # 11. Training status and metrics tracking
    # -------------------------------------------------------------------------
    def test_status_and_lifecycle_reporting(self):
        controller = TrainingController(
            output_root=self.output_root,
            min_samples=5,
            metadata_path=self.metadata_path,
        )

        idle_status = controller.get_status()
        self.assertEqual(idle_status["state"], TrainingLifecycleState.IDLE.value)
        self.assertIsNone(idle_status["jobId"])

        def mock_train(dataset_path: Path, output_dir: Path, **kwargs):
            create_dummy_adapter_dir(output_dir, version="v3")
            return {"loss": 0.42, "steps": 100}

        controller.training_fn = mock_train
        controller.run_training_job(self.valid_samples)

        completed_status = controller.get_status()
        self.assertEqual(completed_status["state"], TrainingLifecycleState.COMPLETED.value)
        self.assertEqual(completed_status["sampleCount"], 5)
        self.assertEqual(completed_status["targetVersion"], "v3")
        self.assertEqual(completed_status["metrics"]["loss"], 0.42)
        self.assertGreater(completed_status["durationSec"], 0.0)


if __name__ == "__main__":
    unittest.main()
