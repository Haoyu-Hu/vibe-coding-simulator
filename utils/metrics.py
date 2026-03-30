from __future__ import annotations


def succ_at_k(success_flags: list[bool]) -> float:
    if not success_flags:
        return 0.0
    return 1.0 if any(success_flags) else 0.0


def avg_turns_to_success(turn_results: list[bool]) -> float | None:
    for idx, ok in enumerate(turn_results, start=1):
        if ok:
            return float(idx)
    return None


def trapezoid_auc(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) < 2:
        return 0.0
    area = 0.0
    for i in range(1, len(xs)):
        width = xs[i] - xs[i - 1]
        height = (ys[i] + ys[i - 1]) / 2.0
        area += width * height
    return area


def precision_recall_auc(labels: list[int], scores: list[float]) -> float:
    if len(labels) != len(scores):
        raise ValueError("labels and scores length mismatch")
    if not labels:
        return 0.0

    paired = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    total_pos = sum(labels)
    if total_pos == 0:
        return 0.0

    tp = 0
    fp = 0
    recalls = [0.0]
    precisions = [1.0]
    for score, label in paired:
        if label:
            tp += 1
        else:
            fp += 1
        recall = tp / total_pos
        precision = tp / (tp + fp)
        recalls.append(recall)
        precisions.append(precision)

    # Integrate precision over recall by step-wise trapezoid.
    area = 0.0
    for i in range(1, len(recalls)):
        delta = recalls[i] - recalls[i - 1]
        area += delta * (precisions[i] + precisions[i - 1]) / 2.0
    return area
