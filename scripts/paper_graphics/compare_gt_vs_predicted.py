#!/usr/bin/env python3
"""
Compare HM3D ground truth labels against pipeline-predicted semantic classes.

Loads the HM3D scene to map body UUIDs to ground truth labels (embedded in
body names as ``<label>_<object_id>``), then cross-references with the VLM
summary JSON produced by the extraction pipeline.

Usage:
    python compare_gt_vs_predicted.py batch_output/00800 \
        datasets/matterport3d/hm3d-minival-semantic-annots-v0.2/00800-TEEsavR23oF

    # Only specific rooms
    python compare_gt_vs_predicted.py batch_output/00800 \
        datasets/matterport3d/hm3d-minival-semantic-annots-v0.2/00800-TEEsavR23oF \
        --rooms 1 2

    # Save to CSV
    python compare_gt_vs_predicted.py batch_output/00800 \
        datasets/matterport3d/hm3d-minival-semantic-annots-v0.2/00800-TEEsavR23oF \
        --csv comparison.csv
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from semdt_2d_video.hm3d_world_loader import HM3DWorldLoader


def normalize_label(label: str) -> str:
    """Normalize a label for comparison.

    Converts HM3D snake_case/spaces (``door frame``) and pipeline CamelCase
    (``DoorFrame``) to a common lowercase-no-separator form (``doorframe``).
    """
    # CamelCase -> space-separated
    label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label)
    # Remove all non-alphanumeric, lowercase
    return re.sub(r"[^a-z0-9]", "", label.lower())


def extract_gt_label(body_name: str) -> str:
    """Extract the HM3D ground truth label from a body name like ``bed_16``."""
    # Body names are <label>_<object_id>.  Labels may themselves contain
    # underscores (e.g. ``door_frame_35``), so split on the *last* underscore
    # followed by digits only.
    match = re.match(r"^(.+)_(\d+)$", body_name)
    if match:
        return match.group(1).replace("_", " ")
    return body_name


def build_uuid_to_gt(
    scene_dir: Path, batch_room_dir: Path, room_id: int,
) -> dict[str, str]:
    """Map original pipeline UUIDs to ground truth labels.

    Re-loads the HM3D scene (which produces *new* UUIDs) and recovers the
    original UUID mapping via ``vlm_responses.json``.  The pipeline stored
    ``body_ids`` per group in the same order as ``loader.object_bodies``, so
    zipping by position gives us ``original_uuid -> body_name -> gt_label``.
    """
    # Collect original UUIDs in pipeline order from vlm_responses.json
    responses_file = batch_room_dir / "vlm_responses.json"
    if not responses_file.exists():
        return {}
    with open(responses_file) as f:
        responses = json.load(f)
    original_uuids = []
    for group in sorted(responses, key=lambda g: g["group_index"]):
        original_uuids.extend(group["body_ids"])

    # Re-load scene — object_bodies come back in the same deterministic order
    loader = HM3DWorldLoader(scene_dir=scene_dir, room_id=room_id)
    bodies = loader.object_bodies

    if len(bodies) != len(original_uuids):
        print(f"  Warning: body count mismatch: scene has {len(bodies)}, "
              f"vlm_responses has {len(original_uuids)} UUIDs")

    mapping = {}
    for orig_uuid, body in zip(original_uuids, bodies):
        mapping[orig_uuid] = extract_gt_label(body.name.name)
    return mapping


def load_predictions(room_dir: Path) -> list[dict]:
    """Load the VLM summary for a single room."""
    summary_file = room_dir / "vlm_summary.json"
    if not summary_file.exists():
        return []
    with open(summary_file) as f:
        return json.load(f)


def compare_room(
    scene_dir: Path, batch_room_dir: Path, room_id: int,
) -> list[dict]:
    """Compare GT vs predicted for one room, return list of row dicts."""
    print(f"Loading HM3D scene for room {room_id}...")
    uuid_to_gt = build_uuid_to_gt(scene_dir, batch_room_dir, room_id)

    predictions = load_predictions(batch_room_dir)
    if not predictions:
        print(f"  No predictions found in {batch_room_dir}")
        return []

    rows = []
    for pred in predictions:
        body_id = pred["body_id"]
        gt_label = uuid_to_gt.get(body_id, "?")
        predicted_class = pred.get("class", "?")
        superclass = pred.get("superclass", "?")
        confidence = pred.get("confidence")
        match = normalize_label(gt_label) == normalize_label(predicted_class)

        rows.append({
            "room": room_id,
            "body_id": body_id,
            "gt_label": gt_label,
            "predicted_class": predicted_class,
            "superclass": superclass,
            "match": match,
            "confidence": confidence,
        })

    return rows


def print_table(rows: list[dict]) -> None:
    """Pretty-print the comparison as a terminal table."""
    if not rows:
        print("No data to display.")
        return

    # Column widths
    gt_w = max(len(r["gt_label"]) for r in rows)
    pred_w = max(len(r["predicted_class"]) for r in rows)
    super_w = max(len(r["superclass"]) for r in rows)
    gt_w = max(gt_w, len("Ground Truth"))
    pred_w = max(pred_w, len("Predicted"))
    super_w = max(super_w, len("Superclass"))

    header = (
        f"{'Room':>4}  {'Ground Truth':<{gt_w}}  {'Predicted':<{pred_w}}  "
        f"{'Superclass':<{super_w}}  {'Conf':>5}  {'Match'}"
    )
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for r in rows:
        mark = "OK" if r["match"] else "MISS"
        conf_str = f"{r['confidence']:.2f}" if r["confidence"] is not None else "  -  "
        line = (
            f"{r['room']:>4}  {r['gt_label']:<{gt_w}}  "
            f"{r['predicted_class']:<{pred_w}}  {r['superclass']:<{super_w}}  "
            f"{conf_str:>5}  {mark}"
        )
        # ANSI color: green for match, red for miss
        if r["match"]:
            print(f"\033[32m{line}\033[0m")
        else:
            print(f"\033[31m{line}\033[0m")

    print(sep)

    total = len(rows)
    correct = sum(1 for r in rows if r["match"])
    print(f"\nAccuracy: {correct}/{total} ({100 * correct / total:.1f}%)")

    # Per-class breakdown for mismatches
    misses = [r for r in rows if not r["match"]]
    if misses:
        print(f"\nMismatches ({len(misses)}):")
        for r in misses:
            print(f"  {r['gt_label']!r} -> {r['predicted_class']!r} "
                  f"[{r['superclass']}]  (conf={r['confidence']})")


def save_csv(rows: list[dict], path: Path) -> None:
    """Write comparison rows to a CSV file."""
    fieldnames = ["room", "body_id", "gt_label", "predicted_class", "superclass",
                  "match", "confidence"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare HM3D ground truth labels against pipeline predictions."
    )
    parser.add_argument(
        "batch_dir", type=Path,
        help="Batch output directory for one scene (e.g. batch_output/00800).",
    )
    parser.add_argument(
        "scene_dir", type=Path,
        help="HM3D semantic annotation directory "
             "(e.g. datasets/matterport3d/hm3d-minival-semantic-annots-v0.2/00800-TEEsavR23oF).",
    )
    parser.add_argument(
        "--rooms", type=int, nargs="*", default=None,
        help="Room IDs to compare. If omitted, discovers rooms from batch_dir.",
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Save comparison table to a CSV file.",
    )
    args = parser.parse_args()

    # Discover rooms from batch output directory
    if args.rooms:
        room_ids = args.rooms
    else:
        room_dirs = sorted(args.batch_dir.glob("room*"))
        room_ids = []
        for d in room_dirs:
            match = re.match(r"room(\d+)", d.name)
            if match and (d / "vlm_summary.json").exists():
                room_ids.append(int(match.group(1)))
        if not room_ids:
            print(f"No room*/vlm_summary.json found in {args.batch_dir}",
                  file=sys.stderr)
            sys.exit(1)

    print(f"Comparing rooms: {room_ids}")
    all_rows = []
    for rid in room_ids:
        batch_room_dir = args.batch_dir / f"room{rid}"
        rows = compare_room(args.scene_dir, batch_room_dir, rid)
        all_rows.extend(rows)

    print_table(all_rows)

    if args.csv:
        save_csv(all_rows, args.csv)


if __name__ == "__main__":
    main()
