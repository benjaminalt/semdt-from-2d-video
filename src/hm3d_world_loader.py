"""
World loader for HM3D (Habitat-Matterport 3D) Semantics v0.2 scenes.

Parses the semantic annotation GLB + TXT files to produce a World where
each semantically annotated object is a separate Body with a TriangleMesh.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
import trimesh

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import TransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Color, TriangleMesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


def _look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = np.array([0.0, 1.0, 0.0]),
) -> np.ndarray:
    """Compute a camera-to-world 4x4 transform (OpenGL: camera looks along -Z)."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    cam_up = np.cross(right, forward)

    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = cam_up
    mat[:3, 2] = -forward
    mat[:3, 3] = eye
    return mat


@dataclass
class SemanticObject:
    """A single semantically annotated object parsed from HM3D annotations."""

    object_id: int
    label: str
    room_id: int
    hex_color: str
    rgb: Tuple[int, int, int]


@dataclass
class HM3DWorldLoader:
    """
    Load an HM3D scene with semantic annotations into a World.

    Expects a scene directory containing:
      - <scene_id>.semantic.glb  (geometry with per-face annotation colors)
      - <scene_id>.semantic.txt  (color -> label lookup table)

    Optionally accepts a path to the visual GLB for textured rendering.
    """

    scene_dir: Path
    """Directory containing the semantic GLB and TXT files for one scene."""

    visual_glb_path: Optional[Path] = None
    """Optional path to the textured visual GLB (from hm3d-minival-glb-v0.2/)."""

    world: World = field(init=False, default=None)
    """The constructed World."""

    annotations: Dict[Tuple[int, int, int], SemanticObject] = field(
        init=False, default_factory=dict
    )
    """Mapping from RGB color tuple to SemanticObject."""

    _scene_id: str = field(init=False, default="")
    _original_visuals: Dict[UUID, object] = field(init=False, default_factory=dict)
    """Saved copy of each body's mesh visual for reset after highlighting."""

    def __post_init__(self):
        self.scene_dir = Path(self.scene_dir)
        self._scene_id = self._detect_scene_id()
        self.annotations = self._parse_semantic_txt()
        self.world = self._build_world()
        self._save_original_state()

    def _detect_scene_id(self) -> str:
        """Infer the scene ID from the .semantic.txt file in the directory."""
        txt_files = list(self.scene_dir.glob("*.semantic.txt"))
        if not txt_files:
            raise FileNotFoundError(
                f"No .semantic.txt file found in {self.scene_dir}"
            )
        # e.g. "TEEsavR23oF.semantic.txt" -> "TEEsavR23oF"
        return txt_files[0].stem.replace(".semantic", "")

    @property
    def semantic_glb_path(self) -> Path:
        return self.scene_dir / f"{self._scene_id}.semantic.glb"

    @property
    def semantic_txt_path(self) -> Path:
        return self.scene_dir / f"{self._scene_id}.semantic.txt"

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_semantic_txt(self) -> Dict[Tuple[int, int, int], SemanticObject]:
        """Parse the .semantic.txt lookup table."""
        annotations: Dict[Tuple[int, int, int], SemanticObject] = {}
        lines = self.semantic_txt_path.read_text().splitlines()

        for line in lines[1:]:  # skip header ("HM3D Semantic Annotations")
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            object_id = int(parts[0])
            hex_color = parts[1]
            label = parts[2].strip('"')
            room_id = int(parts[3])

            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)

            annotations[(r, g, b)] = SemanticObject(
                object_id=object_id,
                label=label,
                room_id=room_id,
                hex_color=hex_color,
                rgb=(r, g, b),
            )
        return annotations

    # ------------------------------------------------------------------
    # World construction
    # ------------------------------------------------------------------

    def _build_world(self) -> World:
        """Load the semantic GLB, split by annotation color, and build a World."""
        scene = trimesh.load(str(self.semantic_glb_path))

        # Collect all trimesh geometries into a single mesh so we can split uniformly
        meshes: List[trimesh.Trimesh] = []
        if isinstance(scene, trimesh.Scene):
            for geom in scene.geometry.values():
                if isinstance(geom, trimesh.Trimesh):
                    meshes.append(geom)
        elif isinstance(scene, trimesh.Trimesh):
            meshes.append(scene)
        else:
            raise ValueError(
                f"Unexpected type from trimesh.load: {type(scene)}"
            )

        # Group faces across all geometries by their annotation color
        # Key: (r,g,b) -> list of (vertices, faces) ready to be concatenated
        color_groups: Dict[Tuple[int, int, int], List[trimesh.Trimesh]] = {}

        for mesh in meshes:
            # Semantic GLBs may use TextureVisuals (UV-mapped flat color texture)
            # instead of ColorVisuals — convert so we can read per-face colors.
            if hasattr(mesh.visual, "to_color"):
                mesh.visual = mesh.visual.to_color()
            face_colors = mesh.visual.face_colors[:, :3]  # (N, 3) uint8 RGB
            unique_colors = np.unique(face_colors, axis=0)

            for color in unique_colors:
                key = tuple(int(c) for c in color)
                mask = np.all(face_colors == color, axis=1)
                submesh = mesh.submesh([mask], only_watertight=False, append=True)
                if submesh is None or len(submesh.faces) == 0:
                    continue
                color_groups.setdefault(key, []).append(submesh)

        # Build the World
        world = World(name=f"hm3d_{self._scene_id}")
        root = Body(name=PrefixedName("root"))

        with world.modify_world():
            world.add_body(root)

            for rgb, submeshes in color_groups.items():
                combined = trimesh.util.concatenate(submeshes)
                annotation = self.annotations.get(rgb)

                if annotation is None:
                    # Face color not in the lookup table -- skip
                    continue

                body_name = f"{annotation.label}_{annotation.object_id}"

                # Paint the submesh with its annotation color for visual identity
                r, g, b = rgb
                combined.visual.face_colors = np.array(
                    [r, g, b, 255], dtype=np.uint8
                )

                triangle_mesh = TriangleMesh(
                    mesh=combined,
                    origin=TransformationMatrix(),
                    color=Color(R=r / 255.0, G=g / 255.0, B=b / 255.0),
                )
                shape_collection = ShapeCollection([triangle_mesh])

                body = Body(
                    name=PrefixedName(body_name),
                    collision=shape_collection,
                    visual=shape_collection,
                )

                connection = FixedConnection(
                    parent=root,
                    child=body,
                    name=PrefixedName(f"root_to_{body_name}"),
                )
                world.add_connection(connection)

        return world

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_bodies_by_label(self, label: str) -> List[Body]:
        """Return all bodies whose name contains the given semantic label."""
        return [
            body for body in self.world.bodies
            if label in body.name.name
        ]

    def get_bodies_in_room(self, room_id: int) -> List[Body]:
        """Return all bodies belonging to the given room."""
        room_object_names = {
            f"{ann.label}_{ann.object_id}"
            for ann in self.annotations.values()
            if ann.room_id == room_id
        }
        return [
            body for body in self.world.bodies
            if body.name.name in room_object_names
        ]

    @property
    def labels(self) -> List[str]:
        """Return the sorted list of unique semantic labels in this scene."""
        return sorted({ann.label for ann in self.annotations.values()})

    @property
    def room_ids(self) -> List[int]:
        """Return the sorted list of room IDs in this scene."""
        return sorted({ann.room_id for ann in self.annotations.values()})

    @property
    def object_bodies(self) -> List[Body]:
        """Return all bodies except the root (i.e. all semantic objects)."""
        return [b for b in self.world.bodies if b.name.name != "root"]

    # ------------------------------------------------------------------
    # Rendering & highlighting (mirrors WarsawWorldLoader interface)
    # ------------------------------------------------------------------

    def _save_original_state(self) -> None:
        """Snapshot each body's mesh visual so we can restore after highlighting."""
        for body in self.object_bodies:
            mesh = body.collision[0].mesh
            self._original_visuals[body.id] = mesh.visual.copy()

    def _reset_body_colors(self) -> None:
        """Restore all bodies to their original visual state."""
        for body in self.object_bodies:
            body.collision[0].mesh.visual = self._original_visuals[body.id].copy()

    @staticmethod
    def _apply_highlight_to_group(bodies: List[Body]) -> Dict[UUID, Color]:
        """Apply distinct highlight colors to a group of bodies.

        Returns a mapping from body id to the Color that was applied.
        """
        colors = Color.distinct_html_colors(len(bodies))
        for body, color in zip(bodies, colors):
            body_mesh = body.collision[0]
            body_mesh.override_mesh_with_color(color)
        return {body.id: color for body, color in zip(bodies, colors)}

    def render_scene_from_camera_pose(
        self, camera_transform, output_filepath=None
    ) -> bytes:
        """Render the world from a single camera pose, return PNG bytes."""
        scene = trimesh.Scene()
        for body in self.object_bodies:
            mesh = body.collision[0].mesh
            if mesh is not None:
                scene.add_geometry(mesh, node_name=body.name.name)

        scene.graph[scene.camera.name] = camera_transform
        png = scene.save_image(resolution=(1024, 768), visible=True)

        if output_filepath:
            with open(output_filepath, "wb") as f:
                f.write(png)
        return png

    def compute_camera_poses(self) -> Dict[str, np.ndarray]:
        """Compute three camera poses that provide good coverage of the scene.

        Places cameras at 120-degree azimuth intervals around the scene
        centroid, elevated above center.  Uses Y-up (glTF / HM3D convention).
        """
        all_vertices = [
            body.collision[0].mesh.vertices for body in self.object_bodies
        ]
        vertices = np.vstack(all_vertices)
        centroid = vertices.mean(axis=0)

        extent = vertices.max(axis=0) - vertices.min(axis=0)
        max_extent = extent.max()

        distance = max_extent * 1.5
        height_offset = max_extent * 0.5

        up = np.array([0.0, 1.0, 0.0])

        azimuth_angles = {
            "back": np.radians(180),
            "front_right": np.radians(-45),
            "front_left": np.radians(45),
        }

        poses: Dict[str, np.ndarray] = {}
        for name, azimuth in azimuth_angles.items():
            eye = centroid + np.array(
                [
                    distance * np.sin(azimuth),
                    height_offset,
                    distance * np.cos(azimuth),
                ]
            )
            poses[name] = _look_at(eye, centroid, up)

        return poses

    def export_semantic_annotation_inheritance_structure(
        self, output_directory: Path
    ) -> None:
        """Export the kinematic structure and SemanticAnnotation taxonomy to JSON."""
        from semantic_digital_twin.semantic_annotation import SemanticAnnotation
        from krrood.class_diagram.utils.inheritance_structure_exporter import (
            InheritanceStructureExporter,
        )

        output_directory.mkdir(parents=True, exist_ok=True)

        self.world.export_kinematic_structure_tree_to_json(
            output_directory / "kinematic_structure.json",
            include_connections=False,
        )
        InheritanceStructureExporter(
            SemanticAnnotation, output_directory / "semantic_annotations.json"
        ).export()
