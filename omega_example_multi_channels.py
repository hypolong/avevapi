import csv
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

from archetypeai.api_client import ArchetypeAI

MODEL = "OmegaEncoder::omega_embeddings_1_4"
WINDOW = 1024      # timesteps per window (the encoder handles 16-1024)
# All four bearings are embedded together as one multi-channel window;
# bearing_1 is the one that fails in this experiment.
CHANNELS = ["bearing_1", "bearing_2", "bearing_3", "bearing_4"]
TEST_WINDOWS = 20  # held-out windows to classify (~1000 available)


def make_client():
    # Every surface lives under one /api prefix, so a single client serves
    # both uploads (/api/files) and queries (/api/lens/query).
    endpoint = os.environ.get(
        "ATAI_API_ENDPOINT", "https://platform.u1.archetypeai.app").rstrip("/")
    return ArchetypeAI(os.environ["ATAI_API_KEY"], api_endpoint=f"{endpoint}/api")


def read_column(csv_path, column, cast=float):
    """Read one column from a CSV into a list."""
    with open(csv_path, newline="") as csv_file:
        reader = csv.reader(csv_file)
        index = next(reader).index(column)
        return [cast(row[index]) for row in reader]


def read_columns(csv_path, columns, cast=float):
    """Read several columns from a CSV into a list of per-column lists."""
    with open(csv_path, newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        indices = [header.index(column) for column in columns]
        rows = list(reader)
    return [[cast(row[index]) for row in rows] for index in indices]


def window(channels, start):
    """Slice the same WINDOW timesteps out of every channel."""
    return [values[start:start + WINDOW] for values in channels]


def embed(client, channels):
    """Embed one multi-channel window (channels x 16-1024 timesteps) into a flat vector."""
    body = {
        "model": MODEL,
        "query": "",
        "normalize_input": False,
        "events": [
            {"type": "data.numeric_array", "event_data": {"contents": channels}}
        ],
    }
    result = client.requests_post(
        f"{client.api_endpoint}/lens/query",
        data_payload=json.dumps(body),
        additional_headers={"Content-Type": "application/json"},
    )
    # A multi-channel window comes back as one embedding per channel (nested
    # lists); concatenate them into a single flat vector for k-NN distances.
    return flatten(result["response"]["response"])


def flatten(values):
    """Flatten arbitrarily nested lists of numbers into one flat list."""
    if not isinstance(values, list):
        return [values]
    return [number for value in values for number in flatten(value)]


def knn_predict(embedding, library, k=3):
    """Majority vote among the k nearest labeled library embeddings."""
    distances = sorted(
        (math.dist(embedding, lib_embedding), label)
        for lib_embedding, label in library
    )
    votes = [label for _, label in distances[:k]]
    return max(set(votes), key=votes.count)


def main():
    client = make_client()

    # 1. Build a small n-shot library from the labeled shot files.
    #    Normalize every window with ONE scaler per channel fit on the library
    #    pool — per-window normalization would erase the fault's amplitude signal.
    healthy = read_columns("bearing_healthy.csv", CHANNELS)
    degraded = read_columns("bearing_degraded.csv", CHANNELS)
    stats = []
    for healthy_values, degraded_values in zip(healthy, degraded):
        pool = healthy_values + degraded_values
        mean = sum(pool) / len(pool)
        std = (sum((value - mean) ** 2 for value in pool) / len(pool)) ** 0.5
        stats.append((mean, std))

    def scale(channels):
        return [[(value - mean) / std for value in values]
                for values, (mean, std) in zip(channels, stats)]

    starts = range(0, len(healthy[0]) - WINDOW + 1, 256)  # overlapping shots
    shots = [(scale(window(healthy, s)), "healthy") for s in starts]
    shots += [(scale(window(degraded, s)), "degraded") for s in starts]

    with ThreadPoolExecutor(max_workers=8) as pool_executor:
        vectors = list(pool_executor.map(
            lambda shot: embed(client, shot[0]), shots))
    library = [(vector, label) for vector, (_, label) in zip(vectors, shots)]
    print(f"n-shot library: {len(library)} labeled windows")

    # 2. Classify held-out windows the library has never seen.
    #    bearing_inference.csv has no label column, so leakage is impossible.
    timestamps = read_column("bearing_inference.csv", "timestamp", cast=int)
    series = read_columns("bearing_inference.csv", CHANNELS)

    # Non-overlapping windows that don't span a gap in the recording,
    # evenly spread across the timeline.
    all_starts = [
        start for start in range(0, len(timestamps) - WINDOW + 1, WINDOW)
        if timestamps[start + WINDOW - 1] - timestamps[start] == WINDOW - 1
    ]
    stride = max(1, len(all_starts) // TEST_WINDOWS)
    test_starts = all_starts[::stride][:TEST_WINDOWS]

    with ThreadPoolExecutor(max_workers=8) as pool_executor:
        predictions = list(pool_executor.map(
            lambda start: knn_predict(
                embed(client, scale(window(series, start))), library),
            test_starts))

    # 3. Score against the ground-truth labels (used ONLY for scoring).
    truth_by_timestamp = dict(zip(
        read_column("bearing_labels.csv", "timestamp", cast=int),
        read_column("bearing_labels.csv", "label", cast=str)))
    truths = [truth_by_timestamp[timestamps[start]] for start in test_starts]

    for start, pred, truth in zip(test_starts, predictions, truths):
        marker = "ok" if pred == truth else "MISS"
        print(f"  ts={timestamps[start]:>8}  predicted={pred:<8} truth={truth:<8} {marker}")

    # Metrics with "degraded" as the positive class: did we catch the fault?
    pairs = list(zip(predictions, truths))
    correct = sum(pred == truth for pred, truth in pairs)
    true_pos = sum(pred == truth == "degraded" for pred, truth in pairs)
    true_neg = sum(pred == truth == "healthy" for pred, truth in pairs)
    false_pos = sum(pred == "degraded" and truth == "healthy" for pred, truth in pairs)
    false_neg = sum(pred == "healthy" and truth == "degraded" for pred, truth in pairs)
    precision = true_pos / (true_pos + false_pos) if true_pos + false_pos else 0.0
    recall = true_pos / (true_pos + false_neg) if true_pos + false_neg else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print("confusion matrix (rows: truth, cols: predicted)")
    print(f"{'':>16}{'healthy':>9}{'degraded':>10}")
    print(f"{'healthy':>16}{true_neg:>9}{false_pos:>10}")
    print(f"{'degraded':>16}{false_neg:>9}{true_pos:>10}")
    print(f"accuracy : {correct}/{len(pairs)} = {correct / len(pairs):.0%}")
    print(f"precision: {precision:.3f}  recall: {recall:.3f}  f1: {f1:.3f}"
          f"  (positive class: degraded)")


if __name__ == "__main__":
    main()