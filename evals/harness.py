
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_goldens(path):

    path = Path(path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with open(path, "r", encoding="utf-8") as file:
        goldens = json.load(file)

    required = {"query", "ideal_answer", "reference_context"}

    for index, golden in enumerate(goldens, 1):
        missing = required - golden.keys()

        if missing:
            raise ValueError(
                f"Golden case {index} missing fields: {missing}"
            )

    return goldens


def summarize_by_metric(result):

    summary = {}

    for test in result.test_results:
        for metric in test.metrics_data:

            if metric.score is None:
                continue

            name = metric.name

            if name not in summary:
                summary[name] = {
                    "scores": [],
                    "passed": 0,
                    "total": 0,
                }

            summary[name]["scores"].append(metric.score)
            summary[name]["total"] += 1
            summary[name]["passed"] += int(metric.success)

    for values in summary.values():
        values["average_score"] = (
            sum(values["scores"]) / len(values["scores"])
        )
        del values["scores"]

    return summary


def print_summary(title, summary):

    print(f"\n========== {title.upper()} EVALUATION ==========")

    if not summary:
        print("No evaluation results available.")
        return

    for name, values in summary.items():
        print(f"\n{name}")
        print(f"Average score: {values['average_score']:.4f}")
        print(f"Passed: {values['passed']}/{values['total']}")

    print("========================================")
