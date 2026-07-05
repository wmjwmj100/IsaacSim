from __future__ import annotations

import unittest
from pathlib import Path
import threading
import time

from PIL import Image

from isaacsim_vla.backends import FileBridgeBackend, MockBackend
from isaacsim_vla.config import ServerConfig
from isaacsim_vla.store import ArtifactStore
from isaacsim_vla.real_robot_file_bridge_worker import _claim_request, _exclusive_worker_lock, build_parser, handle_request
from isaacsim_vla.workspace_guard import WorkspaceGuard


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
                label_red="source",
                label_green="target",
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

    def test_store_confirms_box_layer(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sample = tmp_path / "sample.png"
            Image.new("RGB", (320, 240), (200, 200, 200)).save(sample)
            store = ArtifactStore(tmp_path / "work")
            obs = store.import_observation(sample, camera_id="top", observation_id="obs_confirm")
            layer = store.save_box_layer(
                camera_id="top",
                observation_id=obs.observation_id,
                red_box=[20, 30, 80, 90],
                green_box=[160, 80, 240, 170],
                label_red="glue",
                label_green="between blocks",
            )
            self.assertFalse(layer.metadata["validated"])

            confirmed = store.confirm_box_layer(
                layer.box_layer_id,
                source_confirmation="green source box encloses the glue bottle",
                target_confirmation="blue target box covers the between-block placement region",
                preview_observation_id=obs.observation_id,
            )

            self.assertTrue(confirmed.metadata["validated"])
            self.assertEqual(
                confirmed.metadata["validation"]["preview_observation_id"],
                obs.observation_id,
            )

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

    def test_workspace_guard_masks_and_rejects_unreachable_boxes(self) -> None:
        import tempfile

        site_config = {
            "camera": {
                "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0},
                "world_from_camera": {
                    "matrix": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                },
            },
            "robot": {
                "safety_limits": {
                    "workspace_min_xyz_m": [-0.2, -0.2, 0.0],
                    "workspace_max_xyz_m": [0.2, 0.2, 0.2],
                }
            },
            "object": {"localization_plane_z_m": 0.0},
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sample = tmp_path / "sample.png"
            Image.new("RGB", (100, 100), (240, 240, 240)).save(sample)
            store = ArtifactStore(tmp_path / "work")
            obs = store.import_observation(sample, camera_id="top", observation_id="obs_guard")
            guard = WorkspaceGuard.from_site_config(
                arm_target="arm-test",
                config_path=tmp_path / "site_config.json",
                site_config=site_config,
            )

            masked_obs = guard.mask_observation(obs)
            self.assertTrue(Path(masked_obs.image_path).is_file())
            with Image.open(masked_obs.image_path).convert("RGB") as image:
                self.assertEqual(image.getpixel((50, 50)), (240, 240, 240))
                self.assertEqual(image.getpixel((5, 5)), (24, 24, 24))

            inside = guard.validate_box_input(
                [45, 45, 55, 55],
                width=100,
                height=100,
                coordinate_system="pixel",
                name="source_box",
            )
            self.assertTrue(inside["reachable"])
            with self.assertRaisesRegex(ValueError, "source_box center projects outside"):
                guard.validate_box_input(
                    [0, 0, 10, 10],
                    width=100,
                    height=100,
                    coordinate_system="pixel",
                    name="source_box",
                )

    def test_mock_backend_vla_execute(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sample = tmp_path / "sample.png"
            Image.new("RGB", (640, 480), (200, 200, 200)).save(sample)
            cfg = _server_config(
                backend="mock",
                workdir=tmp_path / "work",
                bridge_dir=None,
                http_url="",
                timeout_sec=1.0,
                sample_top=sample,
                sample_wrist=None,
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
            cfg = _server_config(
                backend="file",
                workdir=tmp_path / "work",
                bridge_dir=bridge_dir,
                http_url="",
                timeout_sec=2.0,
                sample_top=None,
                sample_wrist=None,
                live_top_image=sample,
                camera_file_max_age=0.0,
                camera_dark_frame_mean_threshold=0.0,
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
                self.assertEqual([json.loads(path.read_text())["operation"] for path in requests], ["vla_execute"])
            finally:
                stop_event.set()
                worker.join(timeout=1.0)
            if worker_error:
                raise worker_error[0]

    def test_real_bridge_algorithm_uses_drawn_box_json(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bridge_dir = tmp_path / "bridge"
            runs_dir = bridge_dir / "runs"
            top_image = tmp_path / "top.png"
            wrist_image = tmp_path / "wrist.png"
            policy_path = tmp_path / "policy"
            infer_script = tmp_path / "infer.sh"
            algorithm_script = tmp_path / "fake_pick_place.sh"
            algorithm_config = tmp_path / "site_config.json"
            Image.new("RGB", (640, 480), (200, 200, 200)).save(top_image)
            Image.new("RGB", (640, 480), (180, 180, 180)).save(wrist_image)
            policy_path.mkdir()
            infer_script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            infer_script.chmod(0o755)
            algorithm_config.write_text("{}\n", encoding="utf-8")
            algorithm_script.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$@\" > \"$(dirname \"$0\")/fake_pick_place_args.txt\"\n"
                "exit 0\n",
                encoding="utf-8",
            )
            algorithm_script.chmod(0o755)

            request = {
                "request_id": "req_test_algorithm",
                "operation": "vla_execute",
                "payload": {
                    "instruction": "move source to target",
                    "atomic_action": "pick_and_place",
                    "overlay": {
                        "box_overlay_id": "box_1",
                        "box_layer_id": "layer_1",
                        "camera_id": "top",
                        "observation_id": "obs_1",
                        "original_image_path": str(top_image),
                        "overlay_path": str(top_image),
                        "source_box": [20, 30, 100, 120],
                        "target_box": [300, 180, 430, 310],
                        "red_box": [20, 30, 100, 120],
                        "green_box": [300, 180, 430, 310],
                        "width": 640,
                        "height": 480,
                    },
                },
            }

            response = handle_request(
                request,
                bridge_dir=bridge_dir,
                runs_dir=runs_dir,
                top_image=top_image,
                wrist_image=wrist_image,
                policy_path=policy_path,
                infer_script=infer_script,
                task="default task",
                steps=1,
                camera_file_max_age=0.0,
                camera_dark_frame_mean_threshold=0.0,
                infer_timeout_sec=5.0,
                execution_backend="algorithm",
                arm_target="arm2",
                algorithm_script=algorithm_script,
                algorithm_config=algorithm_config,
                algorithm_mode="dry-run",
                yes_i_checked_workspace=False,
            )

            self.assertTrue(response["success"], response)
            result = response["result"]
            self.assertTrue(result["success"])
            self.assertFalse(result["applied"])
            self.assertEqual(result["metadata"]["execution_backend"], "algorithm")
            overlay_json_path = Path(result["metadata"]["overlay_json_path"])
            overlay_payload = json.loads(overlay_json_path.read_text(encoding="utf-8"))
            self.assertEqual(overlay_payload["source_box"], [20, 30, 100, 120])
            self.assertEqual(overlay_payload["target_box"], [300, 180, 430, 310])
            command = result["metadata"]["command"]
            self.assertIn("--config", command)
            self.assertEqual(command[command.index("--config") + 1], str(algorithm_config))
            self.assertIn("--allow-uncalibrated", command)
            self.assertIn("--allow-placeholder-adapter", command)
            self.assertIn("--bbox-json", command)
            self.assertEqual(command[command.index("--bbox-json") + 1], str(overlay_json_path))
            self.assertNotIn("--bbox", command)
            self.assertNotIn("--place-bbox", command)
            self.assertTrue((tmp_path / "fake_pick_place_args.txt").is_file())

    def test_real_bridge_worker_requires_explicit_arm_target(self) -> None:
        parser = build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["--bridge-dir", "/tmp/bridge"])


    def test_real_bridge_claim_request_is_atomic(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            requests_dir = tmp_path / "requests"
            requests_dir.mkdir()
            request_path = requests_dir / "req_once.json"
            request_path.write_text(json.dumps({"request_id": "req_once"}), encoding="utf-8")

            first_claim = _claim_request(request_path, None)
            second_claim = _claim_request(request_path, None)

            self.assertIsNotNone(first_claim)
            self.assertTrue(first_claim.is_file())
            self.assertIsNone(second_claim)
            self.assertFalse(request_path.exists())

    def test_real_bridge_worker_lock_rejects_second_owner(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "worker.lock"
            with _exclusive_worker_lock(lock_path, None):
                with self.assertRaises(SystemExit) as raised:
                    with _exclusive_worker_lock(lock_path, None):
                        pass
                self.assertEqual(raised.exception.code, 75)

    def test_real_bridge_writes_audit_events(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bridge_dir = tmp_path / "bridge"
            runs_dir = bridge_dir / "runs"
            top_image = tmp_path / "top.png"
            wrist_image = tmp_path / "wrist.png"
            policy_path = tmp_path / "policy"
            infer_script = tmp_path / "infer.sh"
            algorithm_script = tmp_path / "fake_pick_place.sh"
            algorithm_config = tmp_path / "site_config.json"
            audit_log = bridge_dir / "events.jsonl"
            Image.new("RGB", (640, 480), (200, 200, 200)).save(top_image)
            Image.new("RGB", (640, 480), (180, 180, 180)).save(wrist_image)
            policy_path.mkdir()
            infer_script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            infer_script.chmod(0o755)
            algorithm_config.write_text("{}\n", encoding="utf-8")
            algorithm_script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            algorithm_script.chmod(0o755)

            request = {
                "request_id": "req_audit",
                "operation": "vla_execute",
                "payload": {
                    "instruction": "move source to target",
                    "atomic_action": "pick_and_place",
                    "overlay": {
                        "box_overlay_id": "box_audit",
                        "box_layer_id": "layer_audit",
                        "camera_id": "top",
                        "observation_id": "obs_audit",
                        "original_image_path": str(top_image),
                        "overlay_path": str(top_image),
                        "source_box": [10, 20, 90, 110],
                        "target_box": [300, 190, 430, 320],
                        "width": 640,
                        "height": 480,
                    },
                },
            }

            response = handle_request(
                request,
                bridge_dir=bridge_dir,
                runs_dir=runs_dir,
                top_image=top_image,
                wrist_image=wrist_image,
                policy_path=policy_path,
                infer_script=infer_script,
                task="default task",
                steps=1,
                camera_file_max_age=0.0,
                camera_dark_frame_mean_threshold=0.0,
                infer_timeout_sec=5.0,
                execution_backend="algorithm",
                arm_target="arm2",
                algorithm_script=algorithm_script,
                algorithm_config=algorithm_config,
                algorithm_mode="dry-run",
                yes_i_checked_workspace=False,
                audit_log_path=audit_log,
            )

            self.assertTrue(response["success"], response)
            events = [json.loads(line) for line in audit_log.read_text(encoding="utf-8").splitlines()]
            event_names = [event["event"] for event in events]
            self.assertIn("bridge_request_start", event_names)
            self.assertIn("bridge_algorithm_subprocess_start", event_names)
            self.assertIn("bridge_algorithm_subprocess_end", event_names)
            self.assertIn("bridge_request_end", event_names)
            request_event = next(event for event in events if event["event"] == "bridge_request_start")
            self.assertEqual(request_event["request_id"], "req_audit")
            self.assertEqual(request_event["payload_summary"]["overlay"]["source_box"], [10, 20, 90, 110])
            subprocess_event = next(event for event in events if event["event"] == "bridge_algorithm_subprocess_start")
            self.assertEqual(subprocess_event["target_box"], [300, 190, 430, 320])
            self.assertEqual(subprocess_event["algorithm_mode"], "dry-run")

    def test_box_layer_to_algorithm_flow_preserves_prior_draw_boxes(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bridge_dir = tmp_path / "bridge"
            runs_dir = bridge_dir / "runs"
            top_reference = tmp_path / "top_reference.png"
            top_current = tmp_path / "top_current.png"
            wrist_image = tmp_path / "wrist.png"
            policy_path = tmp_path / "policy"
            infer_script = tmp_path / "infer.sh"
            algorithm_script = tmp_path / "fake_pick_place.sh"
            algorithm_config = tmp_path / "site_config.json"
            Image.new("RGB", (640, 480), (200, 200, 200)).save(top_reference)
            Image.new("RGB", (640, 480), (210, 210, 220)).save(top_current)
            Image.new("RGB", (640, 480), (180, 180, 180)).save(wrist_image)
            policy_path.mkdir()
            infer_script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            infer_script.chmod(0o755)
            algorithm_config.write_text("{}\n", encoding="utf-8")
            algorithm_script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            algorithm_script.chmod(0o755)

            store = ArtifactStore(tmp_path / "work")
            obs_reference = store.import_observation(top_reference, camera_id="top", observation_id="obs_reference")
            layer = store.save_box_layer(
                camera_id="top",
                observation_id=obs_reference.observation_id,
                red_box=[40, 50, 110, 130],
                green_box=[320, 190, 450, 330],
            )
            obs_current = store.import_observation(top_current, camera_id="top", observation_id="obs_current")
            overlay = store.render_layer_on_observation(
                box_layer_id=layer.box_layer_id,
                observation_id=obs_current.observation_id,
            )

            request = {
                "request_id": "req_layer_flow",
                "operation": "vla_execute",
                "payload": {
                    "instruction": "move source to target",
                    "atomic_action": "pick_and_place",
                    "overlay": overlay.model_dump(),
                },
            }
            response = handle_request(
                request,
                bridge_dir=bridge_dir,
                runs_dir=runs_dir,
                top_image=top_current,
                wrist_image=wrist_image,
                policy_path=policy_path,
                infer_script=infer_script,
                task="default task",
                steps=1,
                camera_file_max_age=0.0,
                camera_dark_frame_mean_threshold=0.0,
                infer_timeout_sec=5.0,
                execution_backend="algorithm",
                arm_target="arm2",
                algorithm_script=algorithm_script,
                algorithm_config=algorithm_config,
                algorithm_mode="dry-run",
                yes_i_checked_workspace=False,
            )

            self.assertTrue(response["success"], response)
            result = response["result"]
            overlay_json_path = Path(result["metadata"]["overlay_json_path"])
            overlay_payload = json.loads(overlay_json_path.read_text(encoding="utf-8"))
            self.assertEqual(result["box_layer_id"], layer.box_layer_id)
            self.assertEqual(overlay_payload["source_box"], [40, 50, 110, 130])
            self.assertEqual(overlay_payload["target_box"], [320, 190, 450, 330])
            self.assertEqual(overlay_payload["observation_id"], obs_current.observation_id)
            self.assertTrue(Path(overlay_payload["original_image_path"]).is_file())


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


def _server_config(**kwargs: object) -> ServerConfig:
    values: dict[str, object] = {
        "repo_root": Path.cwd(),
        "arm_id": "arm1",
        "arm_target": "arm1",
        "arm_label": "RoboClaw arm1 top-camera left-side ROS/MoveIt arm",
        "arm_side": "left",
        "arm_camera_side": "top-camera-left",
        "live_top_image": Path.cwd() / ".data" / "camera" / "latest_rgb.png",
        "live_wrist_image": Path.cwd() / ".data" / "camera" / "latest_ugreen_rgb.png",
        "camera_file_max_age": 5.0,
        "camera_dark_frame_mean_threshold": 0.08,
    }
    values.update(kwargs)
    return ServerConfig(**values)


if __name__ == "__main__":
    unittest.main()
