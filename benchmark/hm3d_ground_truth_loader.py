from pathlib import Path
from collections import defaultdict


def detect_room_type(labels):

    labels = set(labels)

    if {"bed", "pillow", "wardrobe"} & labels:
        return "bedroom"

    if {"toilet", "bath", "sink", "shower"} & labels:
        return "bathroom"

    if {"oven", "microwave", "fridge"} & labels:
        return "kitchen"

    if {"tv", "sofa", "couch"} & labels:
        return "living_room"

    if {"table", "chair"} & labels:
        return "dining_room"

    return "unknown"


def load_ground_truth_labels(scene_dir):

    print("\n==============================")
    print("LOADING HM3D GROUND TRUTH")
    print("==============================")

    txt_file = list(Path(scene_dir).glob("*.semantic.txt"))[0]

    rooms = defaultdict(list)

    with open(txt_file) as f:

        lines = f.readlines()[1:]

        for line in lines:

            parts = line.strip().split(",")

            label = parts[2].strip('"').lower()
            room_id = int(parts[3])

            rooms[room_id].append(label)

    room_ids = sorted(rooms.keys())

    # detect semantic room types
    room_types = []

    for r in room_ids:

        labels = rooms[r]

        room_type = detect_room_type(labels)

        room_types.append(room_type)

    # number duplicates (bedroom_2 etc.)
    type_counter = {}

    final_names = []

    for t in room_types:

        if t not in type_counter:
            type_counter[t] = 1
            final_names.append(t)

        else:
            type_counter[t] += 1
            final_names.append(f"{t}_{type_counter[t]}")

    print("\nDetected rooms:\n")

    for i, name in enumerate(final_names):

        room_id = room_ids[i]

        print(f"[{i}] {name} ({len(rooms[room_id])} objects)")

    print(f"[{len(room_ids)}] whole_apartment")

    idx = int(input("\nSelect room: "))

    if idx == len(room_ids):

        labels = []

        for r in room_ids:
            labels.extend(rooms[r])

        print("\nSelected: whole_apartment")

    else:

        room_id = room_ids[idx]

        labels = rooms[room_id]

        print(f"\nSelected room: {final_names[idx]}")

    print("\nGround truth labels:\n")

    for l in labels:
        print(" ", l)

    print("\nTotal ground truth objects:", len(labels))

    return labels