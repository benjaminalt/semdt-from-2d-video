#!/usr/bin/env python3
"""
Batch runner for the semantic digital twin pipeline.

Discovers all HM3D scenes with semantic annotations and runs
extract_class_structure.py + refine_class_structure.py on each,
with per-scene output isolation, automatic DB ID tracking,
and resumability via a persistent state file.

Output structure:
    batch_output/
      <scene_id>/
        taxonomy_export/
        room<N>/                    (or "all/" when --num-rooms is not used)
          images/
            scene_orig_<pose>.png
            scene_<group>_<pose>.png
          vlm_responses.json
          vlm_summary.json
          resolution_log.json       (from refine)
          pending_annotations.json  (from refine)
          instantiation_results.json
      batch_state.json

Usage:
    python scripts/run_batch.py                       # run all scenes
    python scripts/run_batch.py --scenes 00800 00803  # run specific scenes
    python scripts/run_batch.py --resume              # resume from last state
    python scripts/run_batch.py --extract-only        # skip refinement
    python scripts/run_batch.py --dry-run             # show what would run
"""

import argparse
import json
import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
DATASET_DIR = REPO_ROOT / "datasets" / "matterport3d"
SEMANTIC_ANNOTS_DIR = DATASET_DIR / "hm3d-minival-semantic-annots-v0.2"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "batch_output"


def discover_scenes(base_dir: Path) -> list[str]:
    """Return sorted list of scene directory names that have semantic annotations."""
    if not base_dir.exists():
        return []
    return sorted(
        d.name for d in base_dir.iterdir()
        if d.is_dir() and (d / f"{d.name.split('-', 1)[0]}.semantic.glb").exists()
        or list(d.glob("*.semantic.glb"))
    )


def load_state(state_file: Path) -> dict:
    if state_file.exists():
        return json.loads(state_file.read_text())
    return {"scenes": {}}


def save_state(state_file: Path, state: dict):
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2))


def reset_taxonomy() -> bool:
    """Reset generated_classes.py and regenerate ORM to a clean baseline.

    This ensures each scene starts with the hand-authored taxonomy only,
    with no leftover classes from prior scenes.
    """
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "reset_class_taxonomy.py"),
        "--simple",
    ]
    log.info("Resetting taxonomy: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("Taxonomy reset failed:\nSTDERR: %s", result.stderr[-2000:])
        return False

    # Regenerate ORM
    cram_dir = REPO_ROOT.parent / "cognitive_robot_abstract_machine"
    generate_orm = cram_dir / "semantic_digital_twin" / "scripts" / "generate_orm.py"
    if generate_orm.exists():
        log.info("Regenerating ORM...")
        result = subprocess.run(
            [sys.executable, str(generate_orm)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            log.error("ORM regeneration failed:\nSTDERR: %s", result.stderr[-2000:])
            return False
    else:
        log.warning("ORM generation script not found at %s, skipping", generate_orm)

    log.info("Taxonomy reset complete")
    return True


def discover_room_dirs(scene_out: Path) -> list[Path]:
    """Find room subdirectories that contain a vlm_summary.json."""
    room_dirs = []
    for d in sorted(scene_out.iterdir()):
        if d.is_dir() and (d / "vlm_summary.json").exists():
            room_dirs.append(d)
    return room_dirs


def run_extract(
    scene_dir: Path,
    scene_out: Path,
    extra_args: list[str],
) -> tuple[bool, int | None]:
    """
    Run extract_class_structure.py for one scene.
    Returns (success, world_database_id).
    """
    # The combined output file goes at the scene level;
    # per-room files are written by extract into room subdirs.
    output_json = scene_out / "vlm_responses_combined.json"

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "extract_class_structure.py"),
        str(scene_dir),
        str(output_json),
        "--dataset", "hm3d",
        "--output-dir", str(scene_out),
        "--headless",
        *extra_args,
    ]

    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.error("Extract failed for %s:\nSTDOUT: %s\nSTDERR: %s",
                  scene_dir.name, result.stdout[-2000:], result.stderr[-2000:])
        return False, None

    # Parse the database_id from stdout
    db_id = None
    for line in result.stdout.splitlines():
        m = re.search(r"World persisted with database_id:\s*(\d+)", line)
        if m:
            db_id = int(m.group(1))
            break

    if db_id is None:
        log.warning("Could not parse database_id from extract output for %s", scene_dir.name)

    log.info("Extract complete for %s (database_id=%s)", scene_dir.name, db_id)
    return True, db_id


def run_refine(
    summary_json: Path,
    world_db_id: int,
    extra_args: list[str],
) -> bool:
    """Run refine_class_structure.py for one room's summary."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "refine_class_structure.py"),
        str(summary_json),
        str(world_db_id),
        "--dataset", "hm3d",
        *extra_args,
    ]

    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.error("Refine failed for %s:\nSTDOUT: %s\nSTDERR: %s",
                  summary_json.parent.name, result.stdout[-2000:], result.stderr[-2000:])
        return False

    log.info("Refine complete for %s (world_database_id=%d)",
             summary_json.parent.name, world_db_id)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for the semantic digital twin pipeline"
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        help="Scene IDs to process (e.g. 00800 00803). Default: all available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base output directory (default: batch_output/)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved state, skipping already-completed stages",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only run extraction, skip refinement",
    )
    parser.add_argument(
        "--refine-only",
        action="store_true",
        help="Only run refinement (requires prior extraction state)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="Objects per VLM group (passed to extract)",
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Pass --skip_vlm to extract (reparse existing output)",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Pass --render-only to extract",
    )
    parser.add_argument(
        "--num-rooms",
        type=int,
        default=None,
        help="Pass --num-rooms N to extract (HM3D room batching)",
    )
    args = parser.parse_args()

    # Discover scenes
    all_scenes = discover_scenes(SEMANTIC_ANNOTS_DIR)
    if not all_scenes:
        log.error("No scenes found in %s", SEMANTIC_ANNOTS_DIR)
        sys.exit(1)

    if args.scenes:
        # Filter to requested scenes (match by prefix)
        selected = []
        for req in args.scenes:
            matches = [s for s in all_scenes if s.startswith(req)]
            if not matches:
                log.error("No scene matching '%s' found. Available: %s", req, all_scenes)
                sys.exit(1)
            selected.extend(matches)
        scenes = sorted(set(selected))
    else:
        scenes = all_scenes

    log.info("Scenes to process: %s", scenes)

    # State file for resumability
    state_file = args.output_dir / "batch_state.json"
    state = load_state(state_file) if args.resume else {"scenes": {}}

    # Build extra args for extract
    extract_extra = ["--group-size", str(args.group_size)]
    if args.skip_vlm:
        extract_extra.append("--skip_vlm")
    if args.render_only:
        extract_extra.append("--render-only")
    if args.num_rooms is not None:
        extract_extra.extend(["--num-rooms", str(args.num_rooms)])

    # Build extra args for refine
    refine_extra = []

    results = {"total": len(scenes), "extracted": 0, "refined": 0, "failed": []}

    for scene_name in scenes:
        scene_dir = SEMANTIC_ANNOTS_DIR / scene_name
        scene_id = scene_name.split("-")[0]  # e.g. "00800"
        scene_out = args.output_dir / scene_id
        scene_out.mkdir(parents=True, exist_ok=True)

        scene_state = state["scenes"].get(scene_id, {})

        log.info("=" * 60)
        log.info("Processing scene: %s", scene_name)
        log.info("=" * 60)

        if args.dry_run:
            log.info("[DRY RUN] Would reset taxonomy")
            log.info("[DRY RUN] Would extract %s -> %s", scene_dir, scene_out)
            if not args.extract_only:
                log.info("[DRY RUN] Would refine each room in %s", scene_out)
            continue

        # --- Reset taxonomy for scene isolation ---
        if not reset_taxonomy():
            log.error("Failed to reset taxonomy before scene %s, skipping", scene_id)
            results["failed"].append({"scene": scene_id, "stage": "reset"})
            continue

        # --- Extraction ---
        if not args.refine_only:
            if args.resume and scene_state.get("extract_status") == "done":
                log.info("Skipping extraction (already done, db_id=%s)",
                         scene_state.get("database_id"))
            else:
                ok, db_id = run_extract(scene_dir, scene_out, extract_extra)

                scene_state["extract_status"] = "done" if ok else "failed"
                scene_state["database_id"] = db_id
                scene_state["extract_time"] = datetime.now().isoformat()

                state["scenes"][scene_id] = scene_state
                save_state(state_file, state)

                if not ok:
                    results["failed"].append({"scene": scene_id, "stage": "extract"})
                    continue

                results["extracted"] += 1
        else:
            if scene_state.get("extract_status") != "done":
                log.error("Cannot refine %s: extraction not completed", scene_id)
                results["failed"].append(
                    {"scene": scene_id, "stage": "refine", "reason": "no extraction"})
                continue

        # --- Refinement (per room) ---
        if not args.extract_only:
            db_id = scene_state.get("database_id")
            if db_id is None:
                log.error("No database_id for scene %s, cannot refine", scene_id)
                results["failed"].append(
                    {"scene": scene_id, "stage": "refine", "reason": "no db_id"})
                continue

            room_dirs = discover_room_dirs(scene_out)
            if not room_dirs:
                log.error("No room directories with vlm_summary.json found in %s", scene_out)
                results["failed"].append(
                    {"scene": scene_id, "stage": "refine", "reason": "no room summaries"})
                continue

            # Track per-room refine status
            room_states = scene_state.get("rooms", {})
            all_rooms_ok = True

            for room_dir in room_dirs:
                room_name = room_dir.name
                room_state = room_states.get(room_name, {})

                if args.resume and room_state.get("refine_status") == "done":
                    log.info("Skipping refinement for %s/%s (already done)",
                             scene_id, room_name)
                    continue

                summary_json = room_dir / "vlm_summary.json"
                ok = run_refine(summary_json, db_id, refine_extra)

                room_state["refine_status"] = "done" if ok else "failed"
                room_state["refine_time"] = datetime.now().isoformat()
                room_states[room_name] = room_state

                if not ok:
                    all_rooms_ok = False

            scene_state["rooms"] = room_states
            scene_state["refine_status"] = "done" if all_rooms_ok else "partial"
            state["scenes"][scene_id] = scene_state
            save_state(state_file, state)

            if all_rooms_ok:
                results["refined"] += 1
            else:
                results["failed"].append({"scene": scene_id, "stage": "refine"})

    # Final summary
    log.info("=" * 60)
    log.info("BATCH COMPLETE")
    log.info("=" * 60)
    log.info("Total scenes: %d", results["total"])
    log.info("Extracted: %d", results["extracted"])
    log.info("Refined: %d", results["refined"])
    if results["failed"]:
        log.warning("Failed: %s", results["failed"])

    # Save final results
    if not args.dry_run:
        results_file = args.output_dir / "batch_results.json"
        results_file.write_text(json.dumps(results, indent=2))
        log.info("Results saved to %s", results_file)


if __name__ == "__main__":
    main()