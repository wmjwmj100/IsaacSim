# Panthera-HT 80x80 Table Scene

This standalone example imports the Panthera-HT arm from the official ROS2 description package and places it on an 80 cm x 80 cm table in Isaac Sim.

## Assets

`Panthera-HT_Main` is a documentation hub. The importable URDF/xacro and STL meshes live in the linked ROS2 repository:

```bash
mkdir -p external
git clone https://github.com/HighTorque-Robotics/Panthera-HT-ROS2.git external/Panthera-HT-ROS2
```

The launcher defaults to:

```text
external/Panthera-HT-ROS2/src/panthera_ht_description_with_finger/urdf/Panthera-HT_description_with_finger.urdf.xacro
```

You can override it with `PANTHERA_URDF=/absolute/path/to/Panthera-HT_description_with_finger.urdf.xacro`.

## Materials

The official Panthera ROS2 description only exposes one generic grey URDF material and STL meshes. It does not include texture files, MTL files, OBJ/DAE material sidecars, or USD materials for Isaac Sim.

The scene therefore applies Isaac Sim material overrides after URDF import, binding a polished silver metal material to all Panthera links. To keep the original grey URDF material, run with `--use-official-grey` or set `PANTHERA_USE_OFFICIAL_GREY=1`.

## Lighting

The scene uses a natural daylight setup: soft blue sky fill, warm low-intensity sunlight, and a large window-style area light for everyday indoor daylight rather than strong studio lighting.

## Realism Mode

Realism mode is enabled by default to reduce visual sim-to-real gap for VLA experiments. It adds darker matte room panels, a floor, warmer laminate table detail, cable-like props, calibration/marker blocks, lower-intensity daylight, brushed-metal Panthera/mount materials, explicit VLA camera focus/aperture settings, and optional screenshot brightness/contrast/noise jitter. Use `--disable-realism` or `PANTHERA_DISABLE_REALISM=1` for the older minimal table scene. Use `--randomize-realism --realism-seed 42` or `PANTHERA_RANDOMIZE_REALISM=1 PANTHERA_REALISM_SEED=42` to deterministically randomize light intensity/color, material roughness/color, small prop offsets, and VLA screenshot post-processing.

## Layout Cameras and VLA Export

The scene creates layout cameras that match the current real-world setup: `left_wrist_rgb` and `right_wrist_rgb` are wrist-mounted under each arm's `link6`, while `d435i_rgb` and `d435i_depth` model the Intel RealSense D435i mounted at the opposite table edge. D435i RGB/depth intrinsics are recorded from the supplied `640x480@30` profile. The UGREEN wrist cameras are intentionally marked approximate/uncalibrated because the real UVC camera does not expose `fx/fy/cx/cy/distortion`.

Startup and after-motion PNG checks are written to `outputs/panthera_ht/layout_cameras/`. A synchronized VLA-style sequence is written to `outputs/panthera_ht/layout_cameras/sequence/`, with per-frame RGB PNGs, D435i distance-to-image-plane `.npy` arrays, frame metadata, camera poses, joint targets, and an `episode_manifest.json`.

In non-headless GUI mode, the one-click launcher opens continuously updating live viewport windows for `left_wrist_rgb`, `right_wrist_rgb`, and `d435i_rgb`. Use `--disable-layout-cameras` to skip camera creation, `--skip-layout-screenshots` to keep cameras but skip PNG/sequence export, or `--disable-layout-camera-viewports` to keep GUI mode without live camera viewport windows.

## Run

For the portable one-click path, launch directly from the Isaac Sim repo root:

```bash
./run_panthera_ht_one_click.sh
```

The one-click launcher clones Panthera ROS2 assets if missing and builds Isaac Sim release Python if needed. Advanced/manual users can still build first and use the lower-level launcher:

```bash
./build.sh --release
./run_panthera_ht_table.sh
```

Useful options:

```bash
./run_panthera_ht_one_click.sh
PANTHERA_HEADLESS=1 PANTHERA_MAX_FRAMES=120 PANTHERA_DISABLE_LAYOUT_CAMERA_VIEWPORTS=1 ./run_panthera_ht_one_click.sh
./run_panthera_ht_table.sh --no-motion
./run_panthera_ht_table.sh --randomize-realism --realism-seed 42
./run_panthera_ht_table.sh --disable-realism
PANTHERA_TABLE_SIZE=0.80 ./run_panthera_ht_table.sh
```

The script writes a sanitized import URDF to `outputs/panthera_ht/Panthera-HT_description_with_finger_isaacsim.urdf`, resolving `package://.../meshes` URLs to absolute STL paths for Isaac Sim.
