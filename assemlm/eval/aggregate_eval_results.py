#!/usr/bin/env python3
"""

 ``eval_assemlm_v2.py``  split
 JSON
 DataLoader
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


COUNT_KEYS = (
    "Total Dataset Samples",
    "Selected Samples",
    "Evaluated Samples",
    "Not Selected Samples",
    "Dropped Samples",
    "Unevaluated Samples",
    "Valid Samples",
    "Invalid Samples",
)

PERFORMANCE_KEYS = (
    "Avg Pose Loss",
    "Avg GD",
    "RMSE(T)",
    "CD",
    "PA(0.01)",
    "PA(0.008)",
    "PA(0.005)",
    "CD(R)",
)


def _numeric_values(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            values.append(float(value))
    return values


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_finite_numbers(value: Any, path: tuple[str, ...] = ()) -> None:
    """Reject JSON NaN/Infinity values before they reach aggregate arithmetic."""
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_finite_numbers(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite_numbers(child, path + (str(index),))
    elif isinstance(value, float) and not math.isfinite(value):
        location = ".".join(path) or "<root>"
        raise ValueError(f"non-finite numeric value at {location}: {value!r}")


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _metric_summary(details: list[dict[str, Any]]) -> dict[str, float | int]:
    """"""
    valid_details = [row for row in details if row.get("valid", False)]
    summary: dict[str, float | int] = {
        "Evaluated Samples": len(details),
        "Valid Samples": len(valid_details),
        "Invalid Samples": len(details) - len(valid_details),
    }

    loss_values = _numeric_values(details, "pose_loss")
    if loss_values:
        summary["Avg Pose Loss"] = float(_mean(loss_values))

    field_to_summary = {
        "GD": "Avg GD",
        "RMSE(T)": "RMSE(T)",
        "CD": "CD",
        "CD(R)": "CD(R)",
        "PA(0.01)": "PA(0.01)",
        "PA(0.008)": "PA(0.008)",
        "PA(0.005)": "PA(0.005)",
    }
    for detail_key, summary_key in field_to_summary.items():
        values = _numeric_values(valid_details, detail_key)
        if values:
            summary[summary_key] = float(_mean(values))

    return summary


def _aggregate_sources(sources: list[dict[str, Any]]) -> dict[str, Any]:
    details = [row for source in sources for row in source["payload"]["details"]]
    micro_summary = _metric_summary(details)

    # num_samples  drop_last
    for key in COUNT_KEYS:
        values = [source["payload"].get("summary", {}).get(key) for source in sources]
        if all(isinstance(value, (int, float)) for value in values):
            micro_summary[key] = int(sum(values))

    #
    macro_summary = {}
    for key in PERFORMANCE_KEYS:
        values = [
            float(source["payload"]["summary"][key])
            for source in sources
            if _is_finite_number(source["payload"].get("summary", {}).get(key))
        ]
        if values:
            macro_summary[key] = float(_mean(values))

    category_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        category_rows[str(row.get("category", "unknown"))].append(row)
    per_category = {
        category: _metric_summary(rows)
        for category, rows in sorted(category_rows.items())
    }

    return {
        "micro_average_by_sample": micro_summary,
        "macro_average_by_dataset": macro_summary,
        "per_category": per_category,
    }


def _atomic_write_json(output_file: Path, payload: dict[str, Any]) -> None:
    """Publish an aggregate only after its complete JSON has been written."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_name(output_file.name + ".partial")
    temporary_file.unlink(missing_ok=True)
    try:
        with temporary_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        temporary_file.replace(output_file)
    except Exception:
        temporary_file.unlink(missing_ok=True)
        raise


def _parse_result_spec(spec: str) -> tuple[str, str, Path]:
    parts = spec.split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            f"Invalid --result {spec!r}; expected DATASET:SPLIT:/path/to/eval_results.json."
        )
    dataset, split, result_path = parts
    if split not in {"train", "test"}:
        raise ValueError(f"Invalid split {split!r} in --result {spec!r}; expected train or test.")
    return dataset, split, Path(result_path).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        metavar="DATASET:SPLIT:JSON",
        help="One independently generated eval_results.json. Repeat for every dataset/split.",
    )
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--run_timestamp", type=str, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Expected evaluation seed. When omitted, infer it from the source result files.",
    )
    args = parser.parse_args()

    sources = []
    seen_pairs = set()
    for spec in args.result:
        dataset, split, result_path = _parse_result_spec(spec)
        pair = (dataset, split)
        if pair in seen_pairs:
            parser.error(f"Duplicate dataset/split input: {dataset}:{split}")
        seen_pairs.add(pair)
        if not result_path.is_file():
            parser.error(f"Result file does not exist: {result_path}")

        try:
            with result_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"Unable to read valid JSON from {result_path}: {exc}")
        if not isinstance(payload, dict):
            parser.error(f"Malformed evaluation result (top-level JSON must be an object): {result_path}")
        try:
            _validate_finite_numbers(payload)
        except ValueError as exc:
            parser.error(f"Malformed evaluation result: {result_path}: {exc}")
        if not isinstance(payload.get("summary"), dict) or not isinstance(payload.get("details"), list):
            parser.error(f"Malformed evaluation result (summary/details missing): {result_path}")
        if any(not isinstance(row, dict) for row in payload["details"]):
            parser.error(f"Malformed evaluation result (details must contain objects): {result_path}")
        if any("valid" in row and not isinstance(row["valid"], bool) for row in payload["details"]):
            parser.error(f"Malformed evaluation result (details.valid must be boolean): {result_path}")
        for count_key in COUNT_KEYS:
            count_value = payload["summary"].get(count_key)
            if count_value is not None and (
                not _is_finite_number(count_value)
                or int(count_value) != count_value
                or int(count_value) < 0
            ):
                parser.error(
                    f"Malformed evaluation result ({count_key} must be a non-negative integer): {result_path}"
                )
        checkpoint = payload.get("ckpt_path")
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            parser.error(f"Malformed evaluation result (ckpt_path missing): {result_path}")
        source_seed = payload.get("seed")
        if (
            not isinstance(source_seed, int)
            or isinstance(source_seed, bool)
            or source_seed < 0
        ):
            parser.error(
                f"Malformed evaluation result (seed must be a non-negative integer): {result_path}"
            )
        if payload.get("dataset_name") not in (None, dataset):
            parser.error(
                f"Dataset label mismatch for {result_path}: "
                f"file={payload.get('dataset_name')!r}, argument={dataset!r}."
            )
        if payload.get("split") not in (None, split):
            parser.error(
                f"Split mismatch for {result_path}: file={payload.get('split')!r}, argument={split!r}."
            )

        evaluated = payload["summary"].get("Evaluated Samples")
        if evaluated is not None and (
            not _is_finite_number(evaluated)
            or int(evaluated) != evaluated
            or int(evaluated) != len(payload["details"])
        ):
            parser.error(
                f"Evaluated Samples={evaluated} but details has {len(payload['details'])} rows: {result_path}"
            )
        sources.append(
            {
                "dataset": dataset,
                "split": split,
                "path": str(result_path),
                "payload": payload,
            }
        )

    ckpt_paths = sorted({str(source["payload"].get("ckpt_path")) for source in sources})
    if len(ckpt_paths) != 1:
        parser.error(f"All result files must use the same checkpoint, got: {ckpt_paths}")

    source_seeds = {int(source["payload"]["seed"]) for source in sources}
    if len(source_seeds) > 1:
        parser.error(f"All result files must use the same seed, got: {sorted(source_seeds)}")
    inferred_seed = next(iter(source_seeds), None)
    if args.seed is not None and inferred_seed is not None and args.seed != inferred_seed:
        parser.error(
            f"--seed={args.seed} does not match source result seed={inferred_seed}."
        )
    output_seed = args.seed if args.seed is not None else inferred_seed

    by_dataset: dict[str, dict[str, Any]] = defaultdict(dict)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        by_dataset[source["dataset"]][source["split"]] = {
            "result_file": source["path"],
            "summary": source["payload"]["summary"],
            "per_category": source["payload"].get("per_category", {}),
        }
        by_split[source["split"]].append(source)

    output = {
        "format_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_timestamp": args.run_timestamp,
        "seed": output_seed,
        "ckpt_path": ckpt_paths[0],
        "aggregation_note": (
            "Each dataset was evaluated independently. micro_average_by_sample recomputes metrics "
            "from all valid per-sample rows in the same split; macro_average_by_dataset gives each "
            "dataset equal weight. Train and test are intentionally not mixed."
        ),
        "datasets": {dataset: splits for dataset, splits in sorted(by_dataset.items())},
        "overall_by_split": {
            split: _aggregate_sources(split_sources)
            for split, split_sources in sorted(by_split.items())
        },
        "source_results": [
            {
                "dataset": source["dataset"],
                "split": source["split"],
                "path": source["path"],
            }
            for source in sources
        ],
    }

    output_file = args.output_file.expanduser().resolve()
    _atomic_write_json(output_file, output)
    print(f"Aggregated evaluation results saved to {output_file}")


if __name__ == "__main__":
    main()
