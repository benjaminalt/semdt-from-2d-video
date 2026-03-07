import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmark.predicted_loader import load_predicted_labels
from benchmark.hm3d_ground_truth_loader import load_ground_truth_labels
from benchmark.label_normalizer import normalize_labels
from benchmark.matcher import evaluate

from sqlalchemy.orm import Session
from sqlalchemy import select
from krrood.ormatic.utils import create_engine
from semantic_digital_twin.orm.ormatic_interface import WorldMappingDAO

import os


def get_engine():

    DB_NAME = os.getenv("PGDATABASE")
    DB_USER = os.getenv("PGUSER")
    DB_PASSWORD = os.getenv("PGPASSWORD")

    return create_engine(
        f"postgresql://{DB_USER}:{DB_PASSWORD}@localhost:5432/{DB_NAME}"
    )


def list_worlds():

    engine = get_engine()

    session = Session(engine)

    query = select(WorldMappingDAO.name)

    worlds = [r[0] for r in session.execute(query).fetchall()]

    session.close()

    return worlds


def find_scene_directory(scene_id):

    base = Path("datasets/matterport3d/hm3d-minival-semantic-annots-v0.2")

    for folder in base.iterdir():

        if scene_id in folder.name:
            return folder

    raise RuntimeError("Dataset folder not found")


def main():

    worlds = list_worlds()

    print("\nAvailable worlds:\n")

    for i, w in enumerate(worlds):
        print(f"[{i}] {w}")

    idx = int(input("\nSelect world: "))

    world_name = worlds[idx]

    scene_id = world_name.replace("hm3d_", "")

    scene_dir = find_scene_directory(scene_id)

    predicted = load_predicted_labels(world_name)

    ground_truth = load_ground_truth_labels(scene_dir)

    predicted_norm, ground_truth_norm = normalize_labels(
        predicted,
        ground_truth
    )

    evaluate(predicted_norm, ground_truth_norm)


if __name__ == "__main__":
    main()