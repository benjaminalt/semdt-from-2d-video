from collections import Counter


def evaluate(predicted, ground_truth):

    print("\n==============================")
    print("SEMANTIC MATCHING EVALUATION")
    print("==============================")

    pred_counter = Counter(predicted)
    gt_counter = Counter(ground_truth)

    true_positives = 0

    for label in pred_counter:

        true_positives += min(
            pred_counter[label],
            gt_counter[label]
        )

    false_positives = sum(pred_counter.values()) - true_positives
    false_negatives = sum(gt_counter.values()) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0

    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0

    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("\nTrue Positives:", true_positives)
    print("False Positives:", false_positives)
    print("False Negatives:", false_negatives)

    print("\nPrecision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1 Score:", round(f1_score, 3))

    print("\nIntersection over Union (IoU) per class:\n")

    labels = set(predicted).union(set(ground_truth))

    ious = []

    for l in sorted(labels):

        intersection = min(pred_counter[l], gt_counter[l])
        union = pred_counter[l] + gt_counter[l] - intersection

        score = intersection / union if union else 0

        ious.append(score)

        print(f"{l:<20} {score:.3f}")

    miou = sum(ious) / len(ious) if ious else 0

    print("\nMean Intersection over Union (mIoU):", round(miou, 3))