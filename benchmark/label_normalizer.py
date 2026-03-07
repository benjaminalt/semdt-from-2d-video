from benchmark.embedding_matcher import similarity


def normalize_labels(predicted, ground_truth, threshold=0.75):

    print("\n==============================")
    print("SEMANTIC NORMALIZATION")
    print("==============================")

    gt_unique = list(set(ground_truth))

    normalized = []

    for p in predicted:

        if p in gt_unique:
            normalized.append(p)
            continue

        best_score = 0
        best_label = None

        for g in gt_unique:

            s = similarity(p, g)

            if s > best_score:

                best_score = s
                best_label = g

        print(f"{p} -> {best_label} similarity={best_score:.3f}")

        if best_score > threshold:
            normalized.append(best_label)
        else:
            normalized.append(p)

    return normalized, ground_truth