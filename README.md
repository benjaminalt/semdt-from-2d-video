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
