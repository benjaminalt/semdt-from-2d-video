import os
from sqlalchemy.orm import Session
from sqlalchemy import select
from krrood.ormatic.utils import create_engine

from semantic_digital_twin.orm.ormatic_interface import WorldMappingDAO


def get_engine():

    DB_NAME = os.getenv("PGDATABASE")
    DB_USER = os.getenv("PGUSER")
    DB_PASSWORD = os.getenv("PGPASSWORD")
    DB_HOST = os.getenv("PGHOST", "localhost")
    DB_PORT = os.getenv("PGPORT", "5432")

    connection = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    return create_engine(connection)


def load_predicted_labels(world_name):

    print("\n==============================")
    print("LOADING PREDICTED LABELS")
    print("==============================")

    engine = get_engine()
    session = Session(engine)

    query = select(WorldMappingDAO).where(WorldMappingDAO.name == world_name)

    world_dao = session.scalars(query).first()

    if world_dao is None:
        raise RuntimeError("World not found")

    world = world_dao.from_dao()

    labels = []

    for ann in world.semantic_annotations:

        label = ann.__class__.__name__.lower().replace("_", " ")

        labels.append(label)

    session.close()

    print("\nPredicted objects:\n")

    for l in labels:
        print(" ", l)

    print("\nTotal predicted:", len(labels))

    return labels