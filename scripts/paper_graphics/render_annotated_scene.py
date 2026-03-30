#!/usr/bin/env python3
"""
Render an HM3D scene with objects colored by their pipeline-predicted semantic
class.  Produces publication-quality images with a color legend, suitable for
the qualitative evaluation section of a paper.

Each unique semantic class gets a distinct color.  All objects of the same
class share a color, making it easy to see how the pipeline groups objects.

Usage:
    # Render room 1 of scene 00800, headless (no display), save to directory
    python render_annotated_scene.py \
        batch_output/00800 \
        datasets/matterport3d/hm3d-minival-semantic-annots-v0.2/00800-TEEsavR23oF \
        --rooms 1 \
        --headless \
        --output-dir paper_output/00800

    # Render all rooms, include ground truth comparison in legend
    python render_annotated_scene.py \
        batch_output/00800 \
        datasets/matterport3d/hm3d-minival-semantic-annots-v0.2/00800-TEEsavR23oF \
        --output-dir paper_output/00800 \
        --show-gt \
        --headless

    # Also render with ground truth coloring for side-by-side comparison
    python render_annotated_scene.py \
        batch_output/00800 \
        datasets/matterport3d/hm3d-minival-semantic-annots-v0.2/00800-TEEsavR23oF \
        --rooms 1 --headless --output-dir paper_output/00800 \
        --render-gt

    # Custom resolution for high-quality paper figures
    python render_annotated_scene.py \
        batch_output/00800 \
        datasets/matterport3d/hm3d-minival-semantic-annots-v0.2/00800-TEEsavR23oF \
        --rooms 1 --headless --output-dir paper_output/00800 \
        --resolution 1920 1080
"""

import argparse
import colorsys
import io
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from semdt_2d_video.hm3d_world_loader import HM3DWorldLoader


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------

def generate_class_palette(class_names: List[str]) -> Dict[str, Tuple[int, int, int]]:
    """Assign a visually distinct RGB color to each class name.

    Uses golden-ratio hue spacing in HSV with varied saturation/value to
    maximise perceptual distance between colors.
    """
    n = len(class_names)
    golden_ratio = 0.618033988749895
    palette: Dict[str, Tuple[int, int, int]] = {}
    sat_val = [(0.75, 0.95), (0.90, 0.85), (0.65, 1.0), (0.95, 0.70), (0.80, 0.90)]

    for i, name in enumerate(sorted(class_names)):
        hue = (i * golden_ratio) % 1.0
        s, v = sat_val[i % len(sat_val)]
        r, g, b = colorsys.hsv_to_rgb(hue, s, v)
        palette[name] = (int(r * 255), int(g * 255), int(b * 255))
    return palette


def extract_gt_label(body_name: str) -> str:
    """Extract ground-truth label from body name ``<label>_<object_id>``."""
    match = re.match(r"^(.+)_(\d+)$", body_name)
    return match.group(1).replace("_", " ") if match else body_name


# ---------------------------------------------------------------------------
# Legend rendering (PIL-based, standalone image)
# ---------------------------------------------------------------------------

def render_legend_image(
    palette: Dict[str, Tuple[int, int, int]],
    output_path: Path,
    title: str = "Semantic Classes",
    gt_mapping: Optional[Dict[str, List[str]]] = None,
) -> None:
    """Render a standalone legend image mapping colors to class names.

    If *gt_mapping* is provided (predicted_class -> [gt_labels]), the legend
    shows ground truth labels next to each predicted class for comparison.
    """
    from PIL import Image, ImageDraw, ImageFont

    font_size = 20
    swatch_size = 22
    padding = 16
    line_height = swatch_size + 8
    col_gap = 40

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size + 4)
    except OSError:
        font = ImageFont.load_default()
        title_font = font

    entries = sorted(palette.keys())

    # Measure text widths to determine layout
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)

    class_texts = []
    for cls in entries:
        text = cls
        if gt_mapping and cls in gt_mapping:
            gt_labels = sorted(set(gt_mapping[cls]))
            text += f"  (GT: {', '.join(gt_labels)})"
        class_texts.append(text)

    max_text_w = max(
        draw.textlength(t, font=font) for t in class_texts
    ) if class_texts else 100
    title_w = draw.textlength(title, font=title_font)

    entry_w = swatch_size + 10 + int(max_text_w)
    img_w = max(int(title_w), entry_w) + 2 * padding
    img_h = padding + line_height + padding + len(entries) * line_height + padding

    img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((padding, padding), title, fill=(0, 0, 0), font=title_font)

    y = padding + line_height + padding // 2
    for cls, text in zip(entries, class_texts):
        rgb = palette[cls]
        # Color swatch
        draw.rectangle(
            [padding, y, padding + swatch_size, y + swatch_size],
            fill=rgb, outline=(0, 0, 0),
        )
        # Class name
        draw.text((padding + swatch_size + 10, y), text, fill=(0, 0, 0), font=font)
        y += line_height

    img.save(output_path)


# ---------------------------------------------------------------------------
# Scene coloring & rendering
# ---------------------------------------------------------------------------

def build_body_name_to_class(
    batch_room_dir: Path, loader: HM3DWorldLoader,
) -> Dict[str, str]:
    """Map body *names* to predicted classes using group order from vlm_responses.

    The pipeline stored ``body_ids`` (original UUIDs) per group in the same
    order as ``loader.object_bodies``.  Re-loading produces new UUIDs, so we
    zip by position to recover ``body_name -> predicted_class``.
    """
    responses_file = batch_room_dir / "vlm_responses.json"
    summary_file = batch_room_dir / "vlm_summary.json"
    if not responses_file.exists() or not summary_file.exists():
        return {}

    with open(responses_file) as f:
        responses = json.load(f)
    with open(summary_file) as f:
        predictions = json.load(f)

    # original UUID -> predicted class
    uuid_to_class = {p["body_id"]: p["class"] for p in predictions}

    # Collect original UUIDs in pipeline order
    original_uuids: List[str] = []
    for group in sorted(responses, key=lambda g: g["group_index"]):
        original_uuids.extend(group["body_ids"])

    # Zip with re-loaded bodies (same deterministic order)
    bodies = loader.object_bodies
    name_to_class: Dict[str, str] = {}
    for orig_uuid, body in zip(original_uuids, bodies):
        cls = uuid_to_class.get(orig_uuid)
        if cls:
            name_to_class[body.name.name] = cls
    return name_to_class


def colorize_bodies(
    loader: HM3DWorldLoader,
    body_class_map: Dict[str, str],
    palette: Dict[str, Tuple[int, int, int]],
) -> None:
    """Paint each body according to its predicted class color.

    *body_class_map* maps body **names** (not UUIDs) to class names.
    """
    for body in loader.object_bodies:
        cls = body_class_map.get(body.name.name)
        if cls and cls in palette:
            rgb = palette[cls]
        else:
            rgb = (180, 180, 180)  # gray for unmapped

        mesh = body.collision[0].mesh
        mesh.visual.face_colors = np.array(
            [rgb[0], rgb[1], rgb[2], 255], dtype=np.uint8,
        )


def colorize_bodies_by_gt(
    loader: HM3DWorldLoader,
    palette: Dict[str, Tuple[int, int, int]],
) -> None:
    """Paint each body according to its ground-truth label color."""
    for body in loader.object_bodies:
        gt = extract_gt_label(body.name.name)
        if gt in palette:
            rgb = palette[gt]
        else:
            rgb = (180, 180, 180)
        mesh = body.collision[0].mesh
        mesh.visual.face_colors = np.array(
            [rgb[0], rgb[1], rgb[2], 255], dtype=np.uint8,
        )


def render_room(
    scene_dir: Path,
    batch_room_dir: Path,
    room_id: int,
    output_dir: Path,
    headless: bool,
    resolution: Tuple[int, int],
    show_gt: bool,
    render_gt: bool,
    visual_glb_path: Optional[Path],
) -> None:
    """Render one room with predicted-class coloring (and optionally GT)."""
    print(f"\n--- Room {room_id} ---")

    # Load scene first (needed to build the name-based mapping)
    print(f"  Loading HM3D scene (room {room_id})...")
    loader = HM3DWorldLoader(
        scene_dir=scene_dir, visual_glb_path=visual_glb_path, room_id=room_id,
    )

    # Build body_name -> predicted_class using order-based UUID recovery
    body_class_map = build_body_name_to_class(batch_room_dir, loader)
    if not body_class_map:
        print(f"  No predictions found in {batch_room_dir}, skipping.")
        return

    predicted_classes = sorted(set(body_class_map.values()))
    palette = generate_class_palette(predicted_classes)

    # Build GT mapping for legend: predicted_class -> [gt_labels]
    gt_mapping: Dict[str, List[str]] = {}
    if show_gt or render_gt:
        for body in loader.object_bodies:
            gt = extract_gt_label(body.name.name)
            pred = body_class_map.get(body.name.name)
            if pred:
                gt_mapping.setdefault(pred, []).append(gt)

    room_out = output_dir / f"room{room_id}"
    room_out.mkdir(parents=True, exist_ok=True)

    # Compute camera poses
    poses = loader.compute_camera_poses()

    # --- Render original textured scene ---
    print("  Rendering original scene...")
    for pose_name, cam in poses.items():
        loader._reset_body_colors()
        png = loader.render_scene_from_camera_pose(
            cam, room_out / f"original_{pose_name}.png",
            headless=headless, use_visual_mesh=True,
        )

    # --- Render predicted-class coloring ---
    print("  Rendering predicted-class coloring...")
    colorize_bodies(loader, body_class_map, palette)
    for pose_name, cam in poses.items():
        png = loader.render_scene_from_camera_pose(
            cam, room_out / f"predicted_{pose_name}.png",
            headless=headless,
        )

    # Save legend
    render_legend_image(
        palette, room_out / "legend_predicted.png",
        title=f"Predicted Classes (Room {room_id})",
        gt_mapping=gt_mapping if show_gt else None,
    )

    # --- Optionally render GT coloring ---
    if render_gt:
        gt_labels = sorted({
            extract_gt_label(b.name.name) for b in loader.object_bodies
        })
        gt_palette = generate_class_palette(gt_labels)
        loader._reset_body_colors()
        colorize_bodies_by_gt(loader, gt_palette)
        for pose_name, cam in poses.items():
            png = loader.render_scene_from_camera_pose(
                cam, room_out / f"gt_{pose_name}.png",
                headless=headless,
            )
        render_legend_image(
            gt_palette, room_out / "legend_gt.png",
            title=f"Ground Truth Labels (Room {room_id})",
        )

    print(f"  Saved to {room_out}/")


def main():
    parser = argparse.ArgumentParser(
        description="Render HM3D scene with pipeline-predicted semantic "
                    "annotations for paper figures.",
    )
    parser.add_argument(
        "batch_dir", type=Path,
        help="Batch output directory for one scene (e.g. batch_output/00800).",
    )
    parser.add_argument(
        "scene_dir", type=Path,
        help="HM3D semantic annotation directory.",
    )
    parser.add_argument(
        "--rooms", type=int, nargs="*", default=None,
        help="Room IDs to render. If omitted, discovers from batch_dir.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for rendered images. Default: <batch_dir>/paper_renders.",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Use offscreen rendering (no display required).",
    )
    parser.add_argument(
        "--resolution", type=int, nargs=2, default=[1024, 768],
        metavar=("W", "H"),
        help="Render resolution (default: 1024 768).",
    )
    parser.add_argument(
        "--show-gt", action="store_true",
        help="Show ground truth labels alongside predicted classes in the legend.",
    )
    parser.add_argument(
        "--render-gt", action="store_true",
        help="Also render the scene colored by ground truth labels (for "
             "side-by-side comparison in the paper).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (args.batch_dir / "paper_renders")

    # Discover rooms
    if args.rooms:
        room_ids = args.rooms
    else:
        room_dirs = sorted(args.batch_dir.glob("room*"))
        room_ids = []
        for d in room_dirs:
            m = re.match(r"room(\d+)", d.name)
            if m and (d / "vlm_summary.json").exists():
                room_ids.append(int(m.group(1)))
        if not room_ids:
            print(f"No room*/vlm_summary.json found in {args.batch_dir}",
                  file=sys.stderr)
            sys.exit(1)

    # Derive visual GLB path
    visual_glb_path = None
    glb_dir = (
        args.scene_dir.parent.parent
        / "hm3d-minival-glb-v0.2"
        / args.scene_dir.name
    )
    if glb_dir.exists():
        glb_files = list(glb_dir.glob("*.glb"))
        if glb_files:
            visual_glb_path = glb_files[0]
            print(f"Visual GLB: {visual_glb_path}")

    resolution = tuple(args.resolution)

    for rid in room_ids:
        batch_room_dir = args.batch_dir / f"room{rid}"
        render_room(
            scene_dir=args.scene_dir,
            batch_room_dir=batch_room_dir,
            room_id=rid,
            output_dir=output_dir,
            headless=args.headless,
            resolution=resolution,
            show_gt=args.show_gt,
            render_gt=args.render_gt,
            visual_glb_path=visual_glb_path,
        )

    print(f"\nAll renders saved to {output_dir}/")


if __name__ == "__main__":
    main()
