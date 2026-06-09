from __future__ import annotations

import unittest
from pathlib import Path
import threading
import time

from PIL import Image

from isaacsim_vla.backends import FileBridgeBackend, MockBackend
from isaacsim_vla.config import ServerConfig
from isaacsim_vla.store import ArtifactStore


class StoreAndBackendTests(unittest.TestCase):
    def test_store_draws_overlay(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sample = tmp_path / "sample.png"
            Image.new("RGB", (320, 240), (200, 200, 200)).save(sample)

            store = ArtifactStore(tmp_path / "work")
            obs = store.import_observation(sample, camera_id="top")
            overlay = store.save_overlay(
                camera_id="top",
                red_box=[20, 30, 80, 90],
                green_box=[160, 80, 240, 170],
            )

            self.assertTrue(Path(overlay.overlay_path).is_file())
            self.assertEqual(overlay.width, 320)
            self.assertEqual(overlay.height, 240)
            self.assertEqual(overlay.observation_id, obs.observation_id)
            self.assertIsNotNone(overlay.box_layer_id)

    def test_store_reuses_box_layer_on_new_observation(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sample1 = tmp_path / "sample1.png"
            sample2 = tmp_path / "sample2.png"
            Image.new("RGB", (320, 240), (200, 200, 200)).save(sample1)
            Image.new("RGB", (320, 240), (120, 160, 220)).save(sample2)

            store = ArtifactStore(tmp_path / "work")
            obs1 = store.import_observation(sample1, camera_id="top", observation_id="obs_1")
            layer = store.save_box_layer(
                camera_id="top",
                observation_id=obs1.observation_id,
                red_box=[20, 30, 80, 90],
                green_box=[160, 80, 240, 170],
            )
            obs2 = store.import_observation(sample2, camera_id="top", observation_id="obs_2")
            overlay = store.render_layer_on_observation(
                box_layer_id=layer.box_layer_id,
                observation_id=obs2.observation_id,
            )

            self.assertEqual(layer.reference_observation_id, obs1.observation_id)
            self.assertEqual(overlay.observation_id, obs2.observation_id)
            self.assertEqual(overlay.box_layer_id, layer.box_layer_id)
            self.assertNotEqual(overlay.box_overlay_id, layer.preview_overlay_id)
            self.assertEqual(overlay.original_image_path, obs2.image_path)
            self.assertTrue(Path(overlay.overlay_path).is_file())

    def test_store_converts_normalized_1000_boxes_to_pixels(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sample = tmp_path / "sample.png"
            Image.new("RGB", (640, 480), (200, 200, 200)).save(sample)

            store = ArtifactStore(tmp_path / "work")
            obs = store.import_observation(sample, camera_id="top", observation_id="obs_norm")
            layer = store.save_box_layer(
                camera_id="top",
                observation_id=obs.observation_id,
                red_box=[609, 510, 734, 635],
                green_box=[656, 281, 859, 416],
                coordinate_system="normalized_1000",
            )

            self.assertEqual(layer.red_box, [389, 244, 469, 304])
            self.assertEqual(layer.green_box, [419, 134, 549, 199])
            self.assertEqual(layer.metadata["coordinate_system"], "normalized_1000")
            self.assertEqual(layer.metadata["red_box_input"], [609, 510, 734, 635])
            self.assertTrue(Path(layer.preview_overlay_path).is_file())

    def test_mock_backend_vla_execute(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sample = tmp_path / "sample.png"
            Image.new("RGB", (640, 480), (200, 200, 200)).save(sample)
            cfg = ServerConfig(
                backend="mock",
                workdir=tmp_path / "work",
                bridge_dir=None,
                http_url="",
                timeout_sec=1.0,
                sample_top=sample,
                sample_wrist=None,
                repo_root=tmp_path,
            )
            store = ArtifactStore(cfg.workdir)
            backend = MockBackend(cfg, store)
            obs = store.import_observation(backend.look_camera("top")["image_path"], camera_id="top")
            overlay = store.save_overlay(
                camera_id="top",
                red_box=[20, 30, 100, 120],
                green_box=[300, 180, 430, 310],
            )
            result = backend.vla_execute("move red thing to green thing", overlay, "pick_and_place")

            self.assertTrue(result["success"])
            self.assertEqual(result["atomic_action"], "pick_and_place")
            self.assertEqual(result["box_overlay_id"], overlay.box_overlay_id)
            self.assertEqual(result["box_layer_id"], overlay.box_layer_id)

    def test_file_bridge_backend_roundtrip(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sample = tmp_path / "sample.png"
            Image.new("RGB", (640, 480), (200, 200, 200)).save(sample)
            bridge_dir = tmp_path / "bridge"
            cfg = ServerConfig(
                backend="file",
                workdir=tmp_path / "work",
                bridge_dir=bridge_dir,
                http_url="",
                timeout_sec=2.0,
                sample_top=None,
                sample_wrist=None,
                repo_root=tmp_path,
            )
            store = ArtifactStore(cfg.workdir)
            backend = FileBridgeBackend(cfg, store)

            stop_event = threading.Event()
            worker_error: list[BaseException] = []
            worker = threading.Thread(
                target=_file_bridge_worker,
                args=(bridge_dir, sample, stop_event, worker_error),
                daemon=True,
            )
            worker.start()

            try:
                camera_result = backend.look_camera("top")
                self.assertEqual(camera_result["image_path"], str(sample))
                obs = store.import_observation(camera_result["image_path"], camera_id="top")
                overlay = store.save_overlay(
                    camera_id="top",
                    observation_id=obs.observation_id,
                    red_box=[20, 30, 100, 120],
                    green_box=[300, 180, 430, 310],
                )
                exec_result = backend.vla_execute("move red box to green box", overlay, "pick_and_place")

                self.assertTrue(exec_result["success"])
                self.assertEqual(exec_result["backend"], "test-file-worker")
                self.assertEqual(exec_result["box_overlay_id"], overlay.box_overlay_id)
                self.assertEqual(exec_result["box_layer_id"], overlay.box_layer_id)
                requests = sorted((bridge_dir / "requests").glob("*.json"))
                self.assertEqual([json.loads(path.read_text())["operation"] for path in requests], ["look_camera", "vla_execute"])
            finally:
                stop_event.set()
                worker.join(timeout=1.0)
            if worker_error:
                raise worker_error[0]


def _file_bridge_worker(
    bridge_dir: Path,
    sample: Path,
    stop_event: threading.Event,
    worker_error: list[BaseException],
) -> None:
    import json

    try:
        requests_dir = bridge_dir / "requests"
        responses_dir = bridge_dir / "responses"
        handled: set[str] = set()
        while not stop_event.is_set():
            for request_path in sorted(requests_dir.glob("*.json")):
                if request_path.name in handled:
                    continue
                try:
                    request = json.loads(request_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                operation = request["operation"]
                payload = request.get("payload") or {}
                if operation == "look_camera":
                    response = {"success": True, "result": {"image_path": str(sample), "camera_id": payload["camera_id"]}}
                elif operation == "vla_execute":
                    overlay = payload["overlay"]
                    response = {
                        "success": True,
                        "result": {
                            "success": True,
                            "backend": "test-file-worker",
                            "result_id": request["request_id"],
                            "instruction": payload["instruction"],
                            "atomic_action": payload["atomic_action"],
                            "box_layer_id": overlay.get("box_layer_id"),
                            "box_overlay_id": overlay["box_overlay_id"],
                        },
                    }
                else:
                    response = {"success": False, "error": f"unsupported operation: {operation}"}
                response_path = responses_dir / request_path.name
                response_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
                handled.add(request_path.name)
            time.sleep(0.01)
    except BaseException as exc:
        worker_error.append(exc)


if __name__ == "__main__":
    unittest.main()
