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
        taxonomy_snapshot/
          generated_classes.py      (scene-specific generated classes)
          ormatic_interface.py      (scene-specific ORM)
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
    python scripts/run_batch.py --resume              # resume from batch_state.json
    python scripts/run_batch.py --continue-from batch_output/  # resume from disk state
    python scripts/run_batch.py --extract-only        # skip refinement
    python scripts/run_batch.py --dry-run             # show what would run
"""

import argparse
import json
import logging
import re
import shutil
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

CRAM_DIR = REPO_ROOT.parent / "cognitive_robot_abstract_machine"
SDT_DIR = CRAM_DIR / "semantic_digital_twin" / "src" / "semantic_digital_twin"
GENERATED_CLASSES_FILE = (
    SDT_DIR / "semantic_annotations" / "generated_classes.py"
)
ORMATIC_INTERFACE_FILE = SDT_DIR / "orm" / "ormatic_interface.py"


def save_taxonomy_snapshot(scene_out: Path):
    """Copy generated_classes.py and ormatic_interface.py to the scene output.

    This preserves the taxonomy state for this scene so that the persisted
    DB world can be loaded later with all generated classes available.
    """
    taxonomy_dir = scene_out / "taxonomy_snapshot"
    taxonomy_dir.mkdir(parents=True, exist_ok=True)

    for src in (GENERATED_CLASSES_FILE, ORMATIC_INTERFACE_FILE):
        if src.exists():
            dst = taxonomy_dir / src.name
            shutil.copy2(src, dst)
            log.info("Saved %s -> %s", src.name, dst)


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
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        log.error("Taxonomy reset failed:\nSTDERR: %s", result.stderr[-2000:])
        return False

    # Regenerate ORM
    generate_orm = CRAM_DIR / "semantic_digital_twin" / "scripts" / "generate_orm.py"
    if generate_orm.exists():
        log.info("Regenerating ORM...")
        result = subprocess.run(
            [sys.executable, str(generate_orm)],
            stdout=sys.stdout, stderr=subprocess.PIPE, text=True,
        )
        if result.returncode != 0:
            log.error("ORM regeneration failed:\nSTDERR: %s", result.stderr[-2000:])
            return False
    else:
        log.warning("ORM generation script not found at %s, skipping", generate_orm)

    log.info("Taxonomy reset complete")
    return True


def reconstruct_state_from_disk(output_dir: Path) -> dict:
    """Rebuild batch state by inspecting the filesystem.

    Used by --continue-from when batch_state.json is missing or incomplete
    (e.g. after a Ctrl-C).  For each scene directory found, checks:
      - Whether rooms have vlm_summary.json  -> extract VLM queries completed
      - Whether a combined vlm_responses file exists
      - Whether rooms have instantiation_results.json -> refine completed
    """
    state: dict = {"scenes": {}}
    if not output_dir.exists():
        return state

    for scene_dir in sorted(output_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_id = scene_dir.name
        # Skip non-scene directories
        if not scene_id.isdigit() and not scene_id[0].isdigit():
            continue

        scene_state: dict = {}

        # Check rooms for extract output
        room_dirs = []
        for d in sorted(scene_dir.iterdir()):
            if d.is_dir() and d.name not in ("taxonomy_export",):
                if (d / "vlm_summary.json").exists():
                    room_dirs.append(d)

        if room_dirs:
            scene_state["extract_vlm_done"] = True
            # Check if DB persistence completed (combined file is written
            # after VLM but before DB persist, so its presence is a hint
            # but not proof of DB persist)
            combined = scene_dir / "vlm_responses_combined.json"
            scene_state["has_combined_json"] = combined.exists()

            # Check per-room refine status
            room_states = {}
            for rd in room_dirs:
                room_name = rd.name
                if (rd / "instantiation_results.json").exists():
                    room_states[room_name] = {"refine_status": "done"}
                else:
                    room_states[room_name] = {"refine_status": "pending"}
            scene_state["rooms"] = room_states

        state["scenes"][scene_id] = scene_state
        log.info("Disk state for %s: vlm_done=%s, combined=%s, rooms=%s",
                 scene_id,
                 scene_state.get("extract_vlm_done", False),
                 scene_state.get("has_combined_json", False),
                 {k: v["refine_status"] for k, v in scene_state.get("rooms", {}).items()})

    return state


def reconstruct_combined_json(scene_out: Path) -> Path:
    """Rebuild the combined vlm_responses JSON from per-room files.

    Needed when extract was interrupted after writing per-room output
    but before writing the combined file.
    """
    combined_path = scene_out / "vlm_responses_combined.json"
    all_responses = []

    for room_dir in sorted(scene_out.iterdir()):
        if not room_dir.is_dir() or room_dir.name == "taxonomy_export":
            continue
        room_responses_file = room_dir / "vlm_responses.json"
        if room_responses_file.exists():
            with open(room_responses_file) as f:
                all_responses.extend(json.load(f))

    with open(combined_path, "w") as f:
        json.dump(all_responses, f, indent=2)

    log.info("Reconstructed combined JSON with %d groups at %s",
             len(all_responses), combined_path)
    return combined_path


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

    # Stream stdout to terminal while capturing it to parse the database_id
    captured_stdout = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
        captured_stdout.append(line)
    proc.wait()
    stderr = proc.stderr.read()

    if proc.returncode != 0:
        log.error("Extract failed for %s:\nSTDERR: %s",
                  scene_dir.name, stderr[-2000:])
        return False, None

    # Parse the database_id from stdout
    db_id = None
    for line in captured_stdout:
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

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    stderr = proc.stderr.read()

    if proc.returncode != 0:
        log.error("Refine failed for %s:\nSTDERR: %s",
                  summary_json.parent.name, stderr[-2000:])
        return False

    log.info("Refine complete for %s (world_database_id=%d)",
             summary_json.parent.name, world_db_id)
    return True


def run_persist(
    pending_jsons: list[Path],
    world_db_id: int,
    dataset: str = "hm3d",
) -> bool:
    """Run persist_annotations.py once with all pending annotation files.

    This regenerates the ORM (once) and persists all annotations in a
    single DB transaction.
    """
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "persist_annotations.py"),
        *[str(p) for p in pending_jsons],
        "--world-db-id", str(world_db_id),
        "--dataset", dataset,
    ]

    log.info("Running: %s", " ".join(cmd))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    stderr = proc.stderr.read()

    if proc.returncode != 0:
        log.error("Persist failed:\nSTDERR: %s", stderr[-2000:])
        return False

    log.info("Persist complete (world_database_id=%d)", world_db_id)
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
        help="Resume from saved batch_state.json, skipping already-completed stages",
    )
    parser.add_argument(
        "--continue-from",
        type=Path,
        default=None,
        metavar="DIR",
        help="Resume from a previous output directory by inspecting what exists "
             "on disk. Unlike --resume, does not require batch_state.json. "
             "Reconstructs state from per-room output files.",
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
    if args.continue_from:
        # Reconstruct state from disk, use that dir as output
        args.output_dir = args.continue_from
        state_file = args.output_dir / "batch_state.json"
        state = reconstruct_state_from_disk(args.continue_from)
        log.info("Reconstructed state from %s: %d scene(s)",
                 args.continue_from, len(state["scenes"]))
    elif args.resume:
        state = load_state(state_file)
    else:
        state = {"scenes": {}}

    # Build extra args for extract
    extract_extra = ["--group-size", str(args.group_size)]
    if args.skip_vlm:
        extract_extra.append("--skip_vlm")
    if args.render_only:
        extract_extra.append("--render-only")
    if args.num_rooms is not None:
        extract_extra.extend(["--num-rooms", str(args.num_rooms)])

    # Build extra args for refine
    # Always skip in-process persistence; we do one combined persist after
    # all rooms finish to avoid regenerating the ORM per room.
    refine_extra = ["--skip-persist"]

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
        # Skip reset in refine-only mode: the generated classes from extraction
        # must remain in place so SQLAlchemy can load the persisted world.
        if not args.refine_only:
            if not reset_taxonomy():
                log.error("Failed to reset taxonomy before scene %s, skipping", scene_id)
                results["failed"].append({"scene": scene_id, "stage": "reset"})
                continue

        # --- Extraction ---
        if not args.refine_only:
            already_done = (
                (args.resume or args.continue_from)
                and scene_state.get("extract_status") == "done"
                and scene_state.get("database_id") is not None
            )
            vlm_done_but_no_db = (
                args.continue_from
                and scene_state.get("extract_vlm_done")
                and scene_state.get("database_id") is None
            )

            if already_done:
                log.info("Skipping extraction (already done, db_id=%s)",
                         scene_state.get("database_id"))
            elif vlm_done_but_no_db:
                # VLM output exists on disk but world was never persisted
                # (e.g. Ctrl-C during DB write). Reconstruct combined JSON
                # if needed, then re-run extract with --skip_vlm to persist.
                log.info("VLM output exists for %s but no DB ID — "
                         "re-running extract with --skip_vlm to persist world",
                         scene_id)
                if not scene_state.get("has_combined_json"):
                    reconstruct_combined_json(scene_out)

                replay_args = [
                    a for a in extract_extra
                    if a not in ("--skip_vlm", "--render-only")
                ]
                replay_args.append("--skip_vlm")
                ok, db_id = run_extract(scene_dir, scene_out, replay_args)

                scene_state["extract_status"] = "done" if ok else "failed"
                scene_state["database_id"] = db_id
                scene_state["extract_time"] = datetime.now().isoformat()
                # Clear disk-only keys now that we have proper state
                scene_state.pop("extract_vlm_done", None)
                scene_state.pop("has_combined_json", None)

                state["scenes"][scene_id] = scene_state
                save_state(state_file, state)

                if not ok:
                    results["failed"].append({"scene": scene_id, "stage": "extract"})
                    continue

                results["extracted"] += 1
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

            # Save taxonomy snapshot after extraction so --refine-only can
            # restore it later without needing to re-run extraction.
            save_taxonomy_snapshot(scene_out)
        else:
            extract_done = (
                scene_state.get("extract_status") == "done"
                or scene_state.get("extract_vlm_done")
            )
            if not extract_done:
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

                if (args.resume or args.continue_from) and room_state.get("refine_status") == "done":
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

            if not all_rooms_ok:
                results["failed"].append({"scene": scene_id, "stage": "refine"})
                continue

            # --- Combined persistence (once per scene, after all rooms) ---
            # Collect pending_annotations.json from all rooms that succeeded
            pending_jsons = []
            for room_dir in room_dirs:
                pa = room_dir / "pending_annotations.json"
                if pa.exists():
                    pending_jsons.append(pa)

            if pending_jsons:
                log.info("Persisting annotations from %d rooms for scene %s",
                         len(pending_jsons), scene_id)
                ok = run_persist(pending_jsons, db_id)
                if ok:
                    results["refined"] += 1
                    scene_state["persist_status"] = "done"
                else:
                    results["failed"].append({"scene": scene_id, "stage": "persist"})
                    scene_state["persist_status"] = "failed"
                state["scenes"][scene_id] = scene_state
                save_state(state_file, state)
            else:
                log.warning("No pending_annotations.json files found for %s", scene_id)

            # Save the accumulated taxonomy (generated_classes.py + ORM)
            # so this scene's DB world can be loaded independently later
            save_taxonomy_snapshot(scene_out)

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