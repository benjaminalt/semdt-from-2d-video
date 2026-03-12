
# Semantic Digital Twin Benchmark

This branch contains a small benchmarking pipeline for evaluating the
semantic objects predicted by the **Semantic Digital Twin system**
against **HM3D ground truth annotations**.

The goal is simple: given a reconstructed scene stored in the database,
compare the predicted objects with the objects annotated in the HM3D
dataset and compute some standard metrics.

This is mainly used to get a quick idea of **how well the semantic
reconstruction matches the real scene**.

------------------------------------------------------------------------

## What the benchmark does

The pipeline compares two things:

1.  **Predicted objects**\
    extracted from the Semantic Digital Twin world stored in PostgreSQL.

2.  **Ground truth objects**\
    taken from the HM3D semantic annotation files.

Since labels are often slightly different (for example *television* vs
*tv*), the system performs a **semantic normalization step using
sentence embeddings** before computing the metrics.

------------------------------------------------------------------------

## Pipeline overview

    database world
    ↓
    load predicted objects
    ↓
    load HM3D ground truth
    ↓
    detect rooms automatically
    ↓
    choose room (or whole apartment)
    ↓
    semantic normalization (embedding similarity)
    ↓
    object matching
    ↓
    metrics (Precision, Recall, F1, IoU, mIoU)

This allows benchmarking either: - a **single room** - or the **entire
apartment**

------------------------------------------------------------------------

## Project structure

    benchmark/
        predicted_loader.py          # loads predicted objects from DB
        hm3d_ground_truth_loader.py  # loads HM3D annotations and detects rooms
        embedding_matcher.py         # semantic similarity using sentence embeddings
        label_normalizer.py          # normalizes labels between prediction and GT
        matcher.py                   # computes evaluation metrics

    scripts/
        evaluate_scene.py            # main script that runs the pipeline

------------------------------------------------------------------------

## Metrics

The following metrics are computed:

**Precision**\
How many predicted objects were correct.

**Recall**\
How many ground truth objects were successfully detected.

**F1 Score**\
Balanced measure between precision and recall.

**IoU (Intersection over Union)**\
Overlap between predicted and ground truth objects for each category.

**mIoU (mean IoU)**\
Average IoU across all categories.

------------------------------------------------------------------------

## Running the benchmark

First activate the environment and make sure the database credentials
are set:

    export PGDATABASE=...
    export PGUSER=...
    export PGPASSWORD=...

Then run:

    python scripts/evaluate_scene.py

The script will:

1.  list the available worlds in the database
2.  ask you to select one
3.  detect all rooms in the scene
4.  let you choose a room or the whole apartment
5.  run the benchmark

Example:

    Available worlds:

    [0] hm3d_TEEsavR23oF

    Select world: 0

    Detected rooms:

    [0] bedroom (35 objects)
    [1] bathroom (29 objects)
    [2] living_room (133 objects)
    ...
    [13] whole_apartment