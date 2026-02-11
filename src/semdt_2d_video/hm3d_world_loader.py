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

    scene_dir: Optional[Path] = None
    """Directory containing the semantic GLB and TXT files for one scene."""

    visual_glb_path: Optional[Path] = None
    """Optional path to the textured visual GLB (from hm3d-minival-glb-v0.2/)."""

    room_id: Optional[int] = None
    """If set, only load objects belonging to this room."""

    world: World = field(default=None)
    """The constructed World (built from files, or provided directly via from_world)."""

    annotations: Dict[Tuple[int, int, int], SemanticObject] = field(
        init=False, default_factory=dict
    )
    """Mapping from RGB color tuple to SemanticObject."""

    _scene_id: str = field(init=False, default="")
    _original_visuals: Dict[UUID, object] = field(init=False, default_factory=dict)
    """Saved copy of each body's mesh visual for reset after highlighting."""

    _visual_scene: Optional[trimesh.Scene] = field(init=False, default=None)
    """Pre-loaded textured visual scene from the visual GLB (if available)."""

    def __post_init__(self):
        if self.world is not None:
            # World provided directly (e.g. loaded from DB) — skip file loading
            self._scene_id = self.world.name or ""
            self.annotations = {}
            self._original_visuals = {}
            self._visual_scene = None
            self._save_original_state()
            return
        self.scene_dir = Path(self.scene_dir)
        self._scene_id = self._detect_scene_id()
        self.annotations = self._parse_semantic_txt()
        self.world = self._build_world()
        self._save_original_state()
        self._load_visual_scene()

    @classmethod
    def from_world(cls, world: World) -> "HM3DWorldLoader":
        """Create an HM3DWorldLoader wrapping an existing World (e.g. loaded from DB).

        Bypasses all file-based loading (GLB parsing, semantic TXT parsing).
        The resulting loader supports rendering, highlighting, and camera pose
        computation but will have an empty ``annotations`` dict.
        """
        loader = object.__new__(cls)
        loader.scene_dir = None
        loader.visual_glb_path = None
        loader.room_id = None
        loader.world = world
        loader.annotations = {}
        loader._scene_id = world.name or ""
        loader._original_visuals = {}
        loader._visual_scene = None
        loader._save_original_state()
        return loader

    def _detect_scene_id(self) -> str:
        """Infer the scene ID from the .semantic.txt file in the directory."""
        txt_files = list(self.scene_dir.glob("*.semantic.txt"))
        if not txt_files:
            raise FileNotFoundError(
                f"No .semantic.txt file found in {self.scene_dir}"
            )
        # e.g. "TEEsavR23oF.semantic.txt" -> "TEEsavR23oF"
        return txt_files[0].stem.replace(".semantic", "")

    @staticmethod
    def discover_room_ids(scene_dir: Path) -> List[int]:
        """Return sorted room IDs from the semantic TXT without loading any GLB."""
        scene_dir = Path(scene_dir)
        txt_files = list(scene_dir.glob("*.semantic.txt"))
        if not txt_files:
            raise FileNotFoundError(
                f"No .semantic.txt file found in {scene_dir}"
            )
        room_ids: set = set()
        for line in txt_files[0].read_text().splitlines()[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            room_ids.add(int(parts[3]))
        return sorted(room_ids)

    @property
    def semantic_glb_path(self) -> Path:
        return self.scene_dir / f"{self._scene_id}.semantic.glb"

    @property
    def semantic_txt_path(self) -> Path:
        return self.scene_dir / f"{self._scene_id}.semantic.txt"

    def _load_visual_scene(self) -> None:
        """Load the textured visual GLB if a path was provided.

        When *room_id* is set, crops the visual scene to the bounding box of
        the room's semantic meshes (with a small margin) so that textured
        renders show only the room.
        """
        if self.visual_glb_path is None:
            return
        path = Path(self.visual_glb_path)
        if not path.exists():
            print(f"Warning: visual GLB not found at {path}, "
                  f"original renders will use semantic colors")
            return
        self._visual_scene = trimesh.load(str(path))

        if self.room_id is not None and self.object_bodies:
            self._crop_visual_scene_to_room()

    def _crop_visual_scene_to_room(self, margin: float = 0.5) -> None:
        """Crop the visual scene to the AABB of the room's semantic meshes."""
        all_verts = np.vstack(
            [body.collision[0].mesh.vertices for body in self.object_bodies]
        )
        bbox_min = all_verts.min(axis=0) - margin
        bbox_max = all_verts.max(axis=0) + margin

        if not isinstance(self._visual_scene, trimesh.Scene):
            return

        to_remove = []
        for name, geom in list(self._visual_scene.geometry.items()):
            if not isinstance(geom, trimesh.Trimesh) or len(geom.vertices) == 0:
                to_remove.append(name)
                continue
            # Keep only vertices inside the AABB
            inside = np.all(
                (geom.vertices >= bbox_min) & (geom.vertices <= bbox_max),
                axis=1,
            )
            if not inside.any():
                to_remove.append(name)
                continue
            # Keep faces where all three vertices are inside
            face_mask = inside[geom.faces].all(axis=1)
            if not face_mask.any():
                to_remove.append(name)
                continue
            geom.update_faces(face_mask)
            geom.remove_unreferenced_vertices()

        for name in to_remove:
            self._visual_scene.delete_geometry(name)

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

                # Early room filter: skip colors not belonging to the target room
                if self.room_id is not None:
                    annotation = self.annotations.get(key)
                    if annotation is None or annotation.room_id != self.room_id:
                        continue

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
        """Snapshot each body's mesh face colors so we can restore after highlighting."""
        for body in self.object_bodies:
            mesh = body.collision[0].mesh
            if hasattr(mesh.visual, "to_color"):
                mesh.visual = mesh.visual.to_color()
            self._original_visuals[body.id] = mesh.visual.face_colors.copy()

    def _reset_body_colors(self) -> None:
        """Restore all bodies to their original face colors."""
        for body in self.object_bodies:
            mesh = body.collision[0].mesh
            mesh.visual.face_colors = self._original_visuals[body.id]

    def _neutralize_body_colors(self) -> None:
        """Set all bodies to a uniform neutral gray.

        Call this before ``_apply_highlight_to_group`` so that only the
        highlighted objects carry distinct colors in the rendered image.
        """
        gray = np.array([180, 180, 180, 255], dtype=np.uint8)
        for body in self.object_bodies:
            mesh = body.collision[0].mesh
            mesh.visual.face_colors = gray

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
        self, camera_transform, output_filepath=None, headless=False,
        use_visual_mesh=False,
    ) -> bytes:
        """Render the world from a single camera pose, return PNG bytes.

        If *use_visual_mesh* is True and a visual GLB was loaded, render
        the textured visual scene instead of the semantic-colored bodies.
        """
        resolution = (1024, 768)

        if use_visual_mesh and self._visual_scene is not None:
            scene = self._visual_scene.copy()
        else:
            scene = trimesh.Scene()
            for body in self.object_bodies:
                mesh = body.collision[0].mesh
                if mesh is not None:
                    scene.add_geometry(mesh, node_name=body.name.name)

        if headless:
            png = self._render_offscreen(scene, camera_transform, resolution)
        else:
            scene.graph[scene.camera.name] = camera_transform
            png = scene.save_image(resolution=resolution, visible=True)

        png = self._autocrop_png(png)

        if output_filepath:
            with open(output_filepath, "wb") as f:
                f.write(png)
        return png

    @staticmethod
    def _autocrop_png(png_bytes: bytes, margin: int = 10) -> bytes:
        """Crop whitespace borders from a rendered PNG image."""
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        arr = np.array(img)

        # Mask of non-white pixels (any channel < 255)
        non_white = np.any(arr < 250, axis=2)
        if not non_white.any():
            return png_bytes

        rows = np.any(non_white, axis=1)
        cols = np.any(non_white, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Add margin, clamped to image bounds
        h, w = arr.shape[:2]
        rmin = max(0, rmin - margin)
        rmax = min(h - 1, rmax + margin)
        cmin = max(0, cmin - margin)
        cmax = min(w - 1, cmax + margin)

        cropped = img.crop((cmin, rmin, cmax + 1, rmax + 1))
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _render_offscreen(
        trimesh_scene: trimesh.Scene,
        camera_transform: np.ndarray,
        resolution: Tuple[int, int],
    ) -> bytes:
        """Render a trimesh scene offscreen using pyrender + EGL."""
        import os
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

        import pyrender
        from PIL import Image
        import io

        # Prepare meshes for pyrender:
        # 1. Convert TextureVisuals to ColorVisuals — avoids GL texture
        #    uploads which fail under EGL with PyOpenGL_accelerate.
        # 2. Unmerge vertices so face colors become per-vertex (pyrender
        #    rejects face colors on smooth meshes).
        for geom in trimesh_scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                if hasattr(geom.visual, "to_color"):
                    geom.visual = geom.visual.to_color()
                geom.unmerge_vertices()

        pr_scene = pyrender.Scene()
        for name, geom in trimesh_scene.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                pr_mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
                pr_scene.add(pr_mesh, name=name)

        # Add a camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        pr_scene.add(camera, pose=camera_transform)

        # Add a directional light so the scene is visible
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        pr_scene.add(light, pose=camera_transform)

        renderer = pyrender.OffscreenRenderer(*resolution)
        color, _ = renderer.render(pr_scene)
        renderer.delete()

        img = Image.fromarray(color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def compute_camera_poses(
        self, bodies: Optional[List[Body]] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute three camera poses that provide good coverage of the scene.

        Auto-detects the up axis from the bounding box (the axis with the
        smallest extent corresponds to floor-to-ceiling height in indoor
        scenes).  Places cameras at two azimuth angles around the centroid
        plus a top-down view.  If *bodies* is None, uses all object bodies.
        """
        if bodies is None:
            bodies = self.object_bodies
        all_vertices = [
            body.collision[0].mesh.vertices for body in bodies
        ]
        vertices = np.vstack(all_vertices)
        centroid = vertices.mean(axis=0)

        extent = vertices.max(axis=0) - vertices.min(axis=0)
        max_extent = extent.max()

        # Detect up axis: smallest bbox extent = floor-to-ceiling in indoor scenes
        up_idx = int(np.argmin(extent))
        ground = [i for i in range(3) if i != up_idx]

        distance = max_extent * 1.5
        height_offset = max_extent * 0.5

        up = np.zeros(3)
        up[up_idx] = 1.0

        azimuth_angles = {
            "front_right": np.radians(-45),
            "front_left": np.radians(45),
        }

        poses: Dict[str, np.ndarray] = {}
        for name, azimuth in azimuth_angles.items():
            offset = np.zeros(3)
            offset[ground[0]] = distance * np.sin(azimuth)
            offset[ground[1]] = distance * np.cos(azimuth)
            offset[up_idx] = height_offset
            eye = centroid + offset
            poses[name] = _look_at(eye, centroid, up)

        # Top-down view: camera above, looking down along the up axis
        top_offset = np.zeros(3)
        top_offset[up_idx] = distance
        top_eye = centroid + top_offset
        top_up = np.zeros(3)
        top_up[ground[1]] = -1.0
        poses["top"] = _look_at(top_eye, centroid, up=top_up)

        return poses

    def export_semantic_annotation_inheritance_structure(
        self, output_directory: Path
    ) -> None:
        """Export the kinematic structure and SemanticAnnotation taxonomy to JSON."""
        from semantic_digital_twin.semantic_annotations.semantic_annotations import SemanticAnnotation
        from semantic_digital_twin.utils import InheritanceStructureExporter

        output_directory.mkdir(parents=True, exist_ok=True)

        self.world.export_kinematic_structure_tree_to_json(
            output_directory / "kinematic_structure.json",
            include_connections=False,
        )
        InheritanceStructureExporter(
            SemanticAnnotation, output_directory / "semantic_annotations.json"
        ).export()
