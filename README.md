# Semantic Digital Twin from 2D Video

Constructs a semantic digital twin from 2D scene imagery using VLM-based object classification and ontology-driven annotation.

## Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/)
- PostgreSQL (for world persistence)
- Access to the [cognitive_robot_abstract_machine](git@github.com:Sanic/cognitive_robot_abstract_machine.git) repository

## Installation

### 1. Install the CRAM dependency

This project depends on packages from the CRAM monorepo (`semantic_digital_twin`, `krrood`, etc.). Install it first:

```bash
git clone git@github.com:Sanic/cognitive_robot_abstract_machine.git
cd cognitive_robot_abstract_machine
git checkout semdt-creation-from-video

python3 -m venv cram-env
source cram-env/bin/activate
pip install poetry
poetry install
```

### 2. Install this package

With the same virtual environment activated:

```bash
cd semdt-from-2d-video
pip install -e .
```

### 3. Configure environment variables

The scripts require database credentials and an API key:

```bash
export PGDATABASE=<your_database>
export PGUSER=<your_user>
export PGPASSWORD=<your_password>
export OPENROUTER_API_KEY=<your_key>  # for VLM queries
```

## Dataset: HM3D Semantics v0.2

This project uses the [Habitat-Matterport 3D (HM3D)](https://aihabitat.org/datasets/hm3d-semantics/) dataset. Each scene is provided as three parallel asset bundles:

| Directory | Files | Purpose |
|---|---|---|
| `hm3d-minival-glb-v0.2/` | `<scene>.glb` | Visual mesh (textured) |
| `hm3d-minival-habitat-v0.2/` | `<scene>.basis.glb` + `<scene>.basis.navmesh` | Habitat simulator mesh + navigation mesh |
| `hm3d-minival-semantic-annots-v0.2/` | `<scene>.semantic.glb` + `<scene>.semantic.txt` | Semantic mesh + label lookup table |

### How semantic annotations work

The **`.semantic.glb`** contains the same geometry as the scene mesh, but every face is painted a flat color encoding which object it belongs to. The **`.semantic.txt`** is the lookup table mapping those colors to labels:

```
object_id, hex_color, label,      room_id
1,         97C517,    "ceiling",   1
16,        1C9E7F,    "bed",       1
51,        067FB0,    "toilet",    2
```

### Extracting per-object meshes

Load the semantic GLB, read per-face colors, and map them through the txt file:

```python
import trimesh
from pathlib import Path

scene_id = "TEEsavR23oF"
base = Path("datasets/matterport3d")

# 1. Parse the lookup table
annotations = {}
txt_path = base / f"hm3d-minival-semantic-annots-v0.2/00800-{scene_id}/{scene_id}.semantic.txt"
for line in txt_path.read_text().splitlines()[1:]:  # skip header
    parts = line.split(",")
    obj_id, hex_color, label, room_id = int(parts[0]), parts[1], parts[2].strip('"'), int(parts[3])
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    annotations[(r, g, b)] = {"id": obj_id, "label": label, "room_id": room_id}

# 2. Load the semantic GLB
semantic_glb = base / f"hm3d-minival-semantic-annots-v0.2/00800-{scene_id}/{scene_id}.semantic.glb"
scene = trimesh.load(str(semantic_glb))

# 3. Map face colors to labels
for name, geom in scene.geometry.items():
    if not isinstance(geom, trimesh.Trimesh):
        continue
    face_colors = geom.visual.face_colors[:, :3]
    for face_idx, color in enumerate(face_colors):
        key = tuple(color)
        if key in annotations:
            ann = annotations[key]
            # ann["label"] -> "bed", "wall", "toilet", ...
            # ann["id"]    -> object instance id
            # ann["room_id"] -> room it belongs to
```

Faces sharing the same annotation can be grouped to extract individual object submeshes.

Note: only a subset of minival scenes have semantic annotations (00800, 00802, 00803, 00808). The remaining scenes have visual/navigation meshes but no annotation layer.

## Usage

**Extract class structure** from a scene directory (renders images, queries a VLM, persists the world to the database):

```bash
python scripts/extract_class_structure.py <obj_dir> <output.json> --export-dir <export_dir>
```

**Refine annotations** by resolving constructor field dependencies and instantiating semantic annotations:

```bash
python scripts/refine_class_structure.py <summary.json> <world_database_id>
```

**Inspect and render** a persisted world:

```bash
python scripts/load_and_render_scene.py <world_name>
python scripts/inspect_camera_pose.py <obj_dir>
```
