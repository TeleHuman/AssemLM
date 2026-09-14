#!/usr/bin/env python3
"""AssemLM web GUI for inspecting datasets and multimodal assembly results.

Run from the AssemLM project with the assemlm_open environment, for example:
    python scripts/AssemLM_GUI.py --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import argparse
import atexit
import base64
import io
import json
import logging
import os
import re
import select
import shutil
import subprocess
import sys
import threading
import traceback
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
from flask import Flask, jsonify, render_template_string, request


def _runtime_project_root() -> Path:
    """Resolve the workspace root for both source checkouts and wheel installs."""
    configured = os.environ.get("ASSEMLM_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    source_root = Path(__file__).resolve().parents[1]
    if (source_root / "config" / "assemlm_v2.yaml").is_file():
        return source_root
    return Path.cwd().resolve()


PROJECT_ROOT = _runtime_project_root()
LOGS_ROOT = PROJECT_ROOT / "logs"
GENERATED_SAMPLES_ROOT = PROJECT_ROOT / "generated_samples"
INFERENCE_WORKER_PATH = PROJECT_ROOT / "main" / "assemlm_inference_worker.py"
DEFAULT_DATASET_PATH = ""
DEFAULT_CHECKPOINT_PATH = ""
DEFAULT_CONFIG_PATH = ""
INFERENCE_JOBS: dict[str, dict[str, Any]] = {}
INFERENCE_JOBS_LOCK = threading.Lock()
MAX_INFERENCE_JOBS = 256
DATASET_LOAD_JOBS: dict[str, dict[str, Any]] = {}
DATASET_LOAD_JOBS_LOCK = threading.Lock()
MAX_DATASET_LOAD_JOBS = 16
SAMPLE_PREVIEWS: dict[str, dict[str, Any]] = {}
SAMPLE_PREVIEWS_LOCK = threading.Lock()
MAX_SAMPLE_PREVIEWS = 128
LOADED_MODEL: dict[str, str] | None = None
ACTIVE_MODEL_LOAD_JOB_ID: str | None = None
LOADED_MODEL_LOCK = threading.Lock()
WORKER_LOCK = threading.Lock()
WORKER_PROCESS: subprocess.Popen[str] | None = None
WORKER_LOG_HANDLE: Any = None
POSE_QUERY_TOKEN = "<pose_query>"
MANUAL_TYPE_LINEART = "Lineart"
MANUAL_TYPE_FREESTYLE = "Freestyle"
MAX_SEED = 2**32 - 2


class JobCapacityError(RuntimeError):
    """Raised when all in-memory inference job slots are still active."""


def _prune_terminal_jobs_locked() -> None:
    """Bound in-memory job state while retaining active and recent results."""
    if len(INFERENCE_JOBS) < MAX_INFERENCE_JOBS:
        return
    for job_id, job in list(INFERENCE_JOBS.items()):
        if len(INFERENCE_JOBS) <= MAX_INFERENCE_JOBS:
            break
        if job.get("state") in {"done", "error"}:
            INFERENCE_JOBS.pop(job_id, None)


def _reserve_job_slot_locked() -> None:
    """Make room for one new job or fail without growing state unboundedly."""
    _prune_terminal_jobs_locked()
    if len(INFERENCE_JOBS) < MAX_INFERENCE_JOBS:
        return
    # ``_prune_terminal_jobs_locked`` intentionally retains one terminal item
    # at the limit so callers can still inspect the most recent result.  A new
    # job needs an additional slot, so evict the oldest terminal item now.
    for job_id, job in list(INFERENCE_JOBS.items()):
        if job.get("state") in {"done", "error"}:
            INFERENCE_JOBS.pop(job_id, None)
            if len(INFERENCE_JOBS) < MAX_INFERENCE_JOBS:
                return
    raise JobCapacityError(
        f"The GUI already has {MAX_INFERENCE_JOBS} active inference jobs; "
        "wait for one to finish before retrying."
    )


def _prune_dataset_load_jobs_locked() -> None:
    """Keep only a bounded number of completed dataset-load jobs."""
    while len(DATASET_LOAD_JOBS) >= MAX_DATASET_LOAD_JOBS:
        terminal_job = next(
            (
                job_id
                for job_id, job in DATASET_LOAD_JOBS.items()
                if job.get("state") in {"done", "error"}
            ),
            None,
        )
        if terminal_job is None:
            break
        DATASET_LOAD_JOBS.pop(terminal_job, None)


def _prune_sample_previews_locked() -> None:
    """Bound unsaved randomization previews created by abandoned browser tabs."""
    while len(SAMPLE_PREVIEWS) > MAX_SAMPLE_PREVIEWS:
        SAMPLE_PREVIEWS.pop(next(iter(SAMPLE_PREVIEWS)))


def _create_run_directory() -> tuple[Path, Path]:
    """Create an isolated, timestamped directory for one GUI process."""
    started_at = datetime.now().astimezone()
    run_name = f"GUI_{started_at.strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir = LOGS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    pid_path = run_dir / "pid"
    pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "entrypoint": str(Path(__file__).resolve()),
                "started_at": started_at.isoformat(),
                "command": sys.argv,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir, pid_path


logger = logging.getLogger("assemlm.gui")
RUN_DIR: Path | None = None
PID_PATH: Path | None = None
LOG_PATH: Path | None = None
RUNTIME_INIT_LOCK = threading.Lock()


def _initialize_runtime_logging() -> Path:
    """Create process artifacts only when the GUI is actually started.

    Keeping this out of module import makes the help command, unit tests, and
    library imports side-effect free.
    """
    global RUN_DIR, PID_PATH, LOG_PATH
    with RUNTIME_INIT_LOCK:
        if RUN_DIR is not None:
            return RUN_DIR
        run_dir, pid_path = _create_run_directory()
        log_path = run_dir / "app.log"
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        if not any(
            type(handler) is logging.StreamHandler
            for handler in root_logger.handlers
        ):
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            root_logger.addHandler(stream_handler)
        RUN_DIR = run_dir
        PID_PATH = pid_path
        LOG_PATH = log_path
        return run_dir

app = Flask(__name__)


@app.after_request
def _disable_response_cache(response):
    """Prevent browsers from reusing stale GUI state after a restart."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def _decode_scalar(value: Any) -> str:
    """Convert HDF5 scalar/array values to clean text."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    size = getattr(value, "size", None)
    if size == 1:
        try:
            return _decode_scalar(value.reshape(-1)[0])
        except Exception:
            try:
                return _decode_scalar(value.item())
            except Exception:
                pass
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        return _decode_scalar(value[0])
    return str(value).strip()


def _split_values(node: h5py.Dataset | h5py.Group) -> list[tuple[str, str | None]]:
    """Read split entries as (asset name, optional inline category)."""
    if isinstance(node, h5py.Group):
        entries: list[tuple[str, str | None]] = []
        for name in node.keys():
            child = node[name]
            entries.append((str(name), _category_from_node(child)))
        return entries

    raw = node[()]
    field_names = getattr(getattr(raw, "dtype", None), "names", None)
    if field_names:
        asset_field = next(
            (x for x in ("asset", "asset_name", "name", "id") if x in field_names),
            field_names[0],
        )
        category_field = next((x for x in ("category", "class", "label") if x in field_names), None)
        rows = raw.reshape(-1) if getattr(raw, "shape", ()) else [raw]
        return [
            (
                _decode_scalar(row[asset_field]),
                _decode_scalar(row[category_field]) if category_field else None,
            )
            for row in rows
        ]

    if isinstance(raw, (str, bytes)):
        return [(_decode_scalar(raw), None)]
    try:
        values: Iterable[Any] = raw.reshape(-1)
    except Exception:
        try:
            values = list(raw)
        except Exception:
            values = [raw]
    return [(_decode_scalar(value), None) for value in values]


def _category_from_node(node: Any) -> str | None:
    if isinstance(node, h5py.Group):
        if "category" in node:
            return _decode_scalar(node["category"][()]) or None
        for key in ("category", "class", "label"):
            if key in node.attrs:
                return _decode_scalar(node.attrs[key]) or None
    elif isinstance(node, h5py.Dataset):
        return _decode_scalar(node[()]) or None
    return None


def _find_split(h5: h5py.File, split: str) -> h5py.Dataset | h5py.Group | None:
    candidates = (
        f"split/{split}",
        f"splits/{split}",
        split,
        f"data/{split}",
        f"datasets/{split}",
    )
    for candidate in candidates:
        if candidate in h5 and isinstance(h5[candidate], (h5py.Dataset, h5py.Group)):
            return h5[candidate]
    return None


def _find_objects(h5: h5py.File) -> h5py.Group | None:
    for candidate in ("objs", "objects", "assets", "data/objs", "data/objects"):
        if candidate in h5 and isinstance(h5[candidate], h5py.Group):
            return h5[candidate]
    return None


def _find_object(objects: h5py.Group | None, asset_name: str) -> Any:
    if objects is None:
        return None
    candidates = [asset_name, asset_name.replace("/", "_")]
    basename = Path(asset_name).name
    if basename not in candidates:
        candidates.append(basename)
    for candidate in candidates:
        if candidate in objects:
            return objects[candidate]
    return None


def _read_group_text(group: h5py.Group, key: str, fallback: str = "") -> str:
    if key not in group:
        return fallback
    return _decode_scalar(group[key][()])


def _read_group_points(group: h5py.Group, key: str, asset_name: str) -> np.ndarray:
    """Read one point cloud using the release's explicit ``[N, 3]`` contract."""
    if key not in group:
        raise KeyError(f"Asset {asset_name!r} is missing {key!r}.")
    points = np.asarray(group[key][()])
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(
            f"Asset {asset_name!r} has invalid {key!r} shape {points.shape}; "
            "expected a non-empty [N, 3] array."
        )
    points = points.astype(np.float32, copy=False)
    if not np.isfinite(points).all():
        raise ValueError(f"Asset {asset_name!r} contains non-finite values in {key!r}.")
    return points


def _read_group_image(group: h5py.Group, key: str) -> np.ndarray | None:
    if key not in group:
        return None
    value = np.asarray(group[key][()])
    if (
        value.ndim != 3
        or value.shape[-1] not in (3, 4)
        or value.shape[0] == 0
        or value.shape[1] == 0
        or value.dtype.kind not in "uif"
    ):
        return None
    numeric = value.astype(np.float64, copy=False)
    if not np.isfinite(numeric).all() or numeric.min() < 0 or numeric.max() > 255:
        return None
    return value.astype(np.uint8)


def _model_instruction(category: str) -> str:
    """Return the internal prompt assembled by AssemLM2._prepare_inputs."""
    return f"Assemble the {category} object {POSE_QUERY_TOKEN}"


def _display_instruction(category: str, raw_instruction: str = "") -> str:
    """Return the natural-language instruction shown to GUI users.

    ``<pose_query>`` is an internal model token and must not be exposed in
    sample metadata or saved instruction files.
    """
    instruction = str(raw_instruction or "").replace(POSE_QUERY_TOKEN, "")
    instruction = re.sub(r"\s+", " ", instruction).strip()
    instruction = re.sub(r"\s+([,.!?;:])", r"\1", instruction)
    if not instruction or not re.search(r"\w", instruction, flags=re.UNICODE):
        return f"Assemble the {category} object."
    return instruction


def _select_manual_images(
    group: h5py.Group,
    file_path: Path | None = None,
) -> tuple[str | None, str | None, str]:
    """Select the image pair that the dataset/model should see.

    twobytwo samples contain both render styles and use lineart.  For every
    other dataset the preferred style is freestyle; if only one pair exists,
    that pair is selected regardless of the file name.  The final return value
    is a display label used by the GUI and saved sample metadata.
    """
    pairs = {
        MANUAL_TYPE_LINEART: ("image_base_lineart", "image_assemble_lineart"),
        MANUAL_TYPE_FREESTYLE: ("image_base_freestyle", "image_assemble_freestyle"),
    }
    available = {
        label: all(key in group for key in pair)
        for label, pair in pairs.items()
    }
    is_twobytwo = bool(file_path and "twobytwo" in file_path.stem.lower())
    if is_twobytwo and available[MANUAL_TYPE_LINEART]:
        label = MANUAL_TYPE_LINEART
    elif available[MANUAL_TYPE_FREESTYLE] and not (
        available[MANUAL_TYPE_LINEART] and is_twobytwo
    ):
        label = MANUAL_TYPE_FREESTYLE
    elif available[MANUAL_TYPE_LINEART]:
        label = MANUAL_TYPE_LINEART
    elif available[MANUAL_TYPE_FREESTYLE]:
        label = MANUAL_TYPE_FREESTYLE
    else:
        return None, None, "Unavailable"
    return (*pairs[label], label)


def _image_data_uri(image: np.ndarray | Image.Image | None) -> str | None:
    if image is None:
        return None
    buffer = io.BytesIO()
    pil_image = image if isinstance(image, Image.Image) else Image.fromarray(image)
    pil_image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _asset_record(
    h5: h5py.File,
    asset_name: str,
    split: str,
    inline_category: str | None = None,
) -> dict[str, Any]:
    objects = _find_objects(h5)
    group = _find_object(objects, asset_name)
    if group is None or not isinstance(group, h5py.Group):
        raise KeyError(f"Asset not found in the objects group: {asset_name}")
    category = (inline_category or _read_group_text(group, "category", "Unlabeled")).strip() or "Unlabeled"
    source_instruction = _read_group_text(group, "instruction")
    display_instruction = _display_instruction(category, source_instruction)
    base_image_key, assemble_image_key, manual_type = _select_manual_images(group, Path(h5.filename))
    key = group.name.rsplit("/", 1)[-1]
    return {
        "asset_name": asset_name,
        "hdf5_key": key,
        "split": split,
        "category": category,
        "instruction": display_instruction,
        "source_instruction": source_instruction,
        "model_instruction": _model_instruction(category),
        "manual_type": manual_type,
        "image_keys": {"base": base_image_key, "assemble": assemble_image_key},
        "has_partA": "partA-pc" in group,
        "has_partB": "base_partB-pc" in group,
        "has_images": base_image_key is not None and assemble_image_key is not None,
    }


def _load_dataset_index(
    file_path: Path,
    split: str | None = None,
    progress_callback=None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with h5py.File(file_path, "r") as h5:
        split_names = [split] if split else [name for name in ("train", "test") if _find_split(h5, name) is not None]
        split_entries: list[tuple[str, list[tuple[str, str | None]]]] = []
        total_entries = 0
        for split_name in split_names:
            node = _find_split(h5, split_name)
            if node is None:
                continue
            entries = _split_values(node)
            split_entries.append((split_name, entries))
            total_entries += len(entries)

        processed = 0
        for split_name, entries in split_entries:
            for asset_name, inline_category in entries:
                records.append(_asset_record(h5, asset_name, split_name, inline_category))
                processed += 1
                if progress_callback is not None:
                    progress_callback(processed, total_entries, split_name)
    return records


def _summarize_hdf5(file_path: Path, progress_callback=None) -> dict[str, Any]:
    file_path = Path(file_path).expanduser().resolve()
    if progress_callback is not None:
        progress_callback(0, 1, "validating dataset")

    def _scan_progress(processed: int, total: int, split_name: str) -> None:
        fraction = processed / max(total, 1)
        percent = 5.0 + 90.0 * fraction
        progress_callback(percent, 100, f"reading {split_name} split")

    records = _load_dataset_index(
        file_path,
        progress_callback=_scan_progress if progress_callback is not None else None,
    )
    if progress_callback is not None:
        progress_callback(97, 100, "summarizing categories")

    splits: dict[str, Any] = {}
    for split in ("train", "test"):
        split_records = [record for record in records if record["split"] == split]
        counts = Counter(record["category"] for record in split_records)
        categories = [
            {"category": category, "count": count}
            for category, count in sorted(counts.items(), key=lambda item: (-item[1], item[0].lower()))
        ]
        splits[split] = {
            "total": len(split_records),
            "categories": categories,
            "samples": split_records,
        }
    category_names = sorted(
        {record["category"] for record in records},
        key=lambda category: (
            -sum(1 for record in records if record["category"] == category),
            category.lower(),
        ),
    )
    result = {
        "file": str(file_path),
        "file_size_mb": file_path.stat().st_size / (1024 * 1024),
        "total": len(records),
        "category_count": len(category_names),
        "categories": [
            {
                "category": category,
                "train": sum(1 for record in records if record["split"] == "train" and record["category"] == category),
                "test": sum(1 for record in records if record["split"] == "test" and record["category"] == category),
                "total": sum(1 for record in records if record["category"] == category),
            }
            for category in category_names
        ],
        "splits": splits,
    }
    if progress_callback is not None:
        progress_callback(100, 100, "dataset ready")
    return result


def _load_hdf5_sample(file_path: Path, split: str, asset_name: str) -> dict[str, Any]:
    with h5py.File(file_path, "r") as h5:
        node = _find_split(h5, split)
        inline_category = None
        if node is not None:
            for candidate_name, candidate_category in _split_values(node):
                if candidate_name == asset_name:
                    inline_category = candidate_category
                    break
        record = _asset_record(h5, asset_name, split, inline_category)
        group = _find_object(_find_objects(h5), asset_name)
        if group is None:
            raise KeyError(f"Asset not found: {asset_name}")
        moving = _read_group_points(group, "partA-pc", asset_name)
        fixed = _read_group_points(group, "base_partB-pc", asset_name)
        image_keys = record.get("image_keys", {})
        base_image = _read_group_image(group, image_keys.get("base")) if image_keys.get("base") else None
        assemble_image = _read_group_image(group, image_keys.get("assemble")) if image_keys.get("assemble") else None
        record.update(
            {
                "images": {
                    "base": _image_data_uri(base_image),
                    "assemble": _image_data_uri(assemble_image),
                },
                "point_clouds": {
                    "moving": moving.tolist(),
                    "fixed": fixed.tolist(),
                },
                "point_counts": {"moving": int(len(moving)), "fixed": int(len(fixed))},
            }
        )
        return record


def _safe_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return safe or "sample"


def _random_rotation(seed: int) -> np.ndarray:
    """Match the reference torch 6D rotation draw for a fixed seed."""
    import torch
    import torch.nn.functional as F

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    raw = torch.rand((1, 6), generator=generator, dtype=torch.float32)
    raw = raw.reshape(-1, 2, 3).permute(0, 2, 1)
    first = F.normalize(raw[:, :, 0], p=2, dim=1)
    second_raw = raw[:, :, 1]
    second = F.normalize(
        second_raw - (first * second_raw).sum(dim=1, keepdim=True) * first,
        p=2,
        dim=1,
    )
    third = torch.cross(first, second, dim=1)
    return torch.stack((first, second, third), dim=1).permute(0, 2, 1)[0].numpy().astype(np.float32)


def _rotation_to_6d(rotation: np.ndarray) -> np.ndarray:
    return rotation[:, :2].T.reshape(6).astype(np.float32)


def _save_image_data_uri(data_uri: str | None, path: Path) -> str | None:
    if not data_uri or "," not in data_uri:
        return None
    encoded = data_uri.split(",", 1)[1]
    path.write_bytes(base64.b64decode(encoded))
    return str(path)


def _resolve_sample_file(sample_dir: Path, raw_path: str | os.PathLike[str], label: str) -> Path:
    """Resolve saved-sample files while keeping metadata paths inside the sample."""
    root = sample_dir.expanduser().resolve()
    candidate = Path(raw_path)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} path must stay inside the sample directory: {raw_path!r}") from exc
    return resolved


def _saved_sample_payload(sample_dir: Path) -> dict[str, Any]:
    sample_dir = sample_dir.expanduser().resolve()
    meta_path = sample_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Sample metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    files = meta.get("files", {})

    def saved_manual_spec() -> tuple[str, str, str, str]:
        requested = str(meta.get("manual_type", "")).strip().lower()
        candidates = [MANUAL_TYPE_LINEART, MANUAL_TYPE_FREESTYLE]
        if requested.startswith("line"):
            candidates = [MANUAL_TYPE_LINEART, MANUAL_TYPE_FREESTYLE]
        elif requested.startswith("free"):
            candidates = [MANUAL_TYPE_FREESTYLE, MANUAL_TYPE_LINEART]
        else:
            # Old samples did not record a type.  Infer it from their files.
            has_lineart = any(
                key in files or (sample_dir / filename).is_file()
                for key, filename in (
                    ("image_base_lineart", "image_base_lineart.png"),
                    ("image_assemble_lineart", "image_assemble_lineart.png"),
                )
            )
            if has_lineart:
                candidates = [MANUAL_TYPE_LINEART, MANUAL_TYPE_FREESTYLE]
        for label in candidates:
            base_key, assemble_key = (
                ("image_base_lineart", "image_assemble_lineart")
                if label == MANUAL_TYPE_LINEART
                else ("image_base_freestyle", "image_assemble_freestyle")
            )
            base_default = f"{base_key}.png"
            assemble_default = f"{assemble_key}.png"
            base_path = _resolve_sample_file(
                sample_dir, files.get(base_key, base_default), "Base image"
            )
            assemble_path = _resolve_sample_file(
                sample_dir, files.get(assemble_key, assemble_default), "Assembly image"
            )
            if base_path.is_file() and assemble_path.is_file():
                return label, base_key, assemble_key, base_default
        return MANUAL_TYPE_FREESTYLE, "image_base_freestyle", "image_assemble_freestyle", "image_base_freestyle.png"

    manual_type, base_image_key, assemble_image_key, _ = saved_manual_spec()

    def load_points(key: str, fallback: str) -> np.ndarray:
        path = _resolve_sample_file(sample_dir, files.get(key, fallback), key)
        points = np.load(path, allow_pickle=False)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError(
                f"{key} point cloud has invalid shape {points.shape}; expected a non-empty [N, 3] array."
            )
        points = points.astype(np.float32, copy=False)
        if not np.isfinite(points).all():
            raise ValueError(f"{key} point cloud contains non-finite values: {path}")
        return points

    def load_uri(key: str, fallback: str) -> str | None:
        path = _resolve_sample_file(sample_dir, files.get(key, fallback), key)
        if not path.is_file():
            return None
        with Image.open(path) as image:
            return _image_data_uri(image.convert("RGB"))

    moving = load_points("partA_pc", "partA_pc.npy")
    fixed = load_points("base_partB_pc", "base_partB_pc.npy")
    category = str(meta.get("category", "Unlabeled"))
    source_instruction = str(
        meta.get("source_instruction") or meta.get("instruction", "") or ""
    )
    display_instruction = _display_instruction(category, source_instruction)
    model_instruction = str(meta.get("model_instruction") or _model_instruction(category))

    def image_fallback(key: str) -> str:
        return f"{key}.png"

    return {
        "asset_name": meta.get("asset_name", sample_dir.name),
        "hdf5_key": meta.get("hdf5_key", ""),
        "split": meta.get("split", ""),
        "category": category,
        "instruction": display_instruction,
        "source_instruction": source_instruction,
        "model_instruction": model_instruction,
        "manual_type": manual_type,
        "image_keys": {"base": base_image_key, "assemble": assemble_image_key},
        "sample_dir": str(sample_dir),
        "images": {
            "base": load_uri(base_image_key, image_fallback(base_image_key)),
            "assemble": load_uri(assemble_image_key, image_fallback(assemble_image_key)),
        },
        "point_clouds": {"moving": moving.tolist(), "fixed": fixed.tolist()},
        "point_counts": {"moving": int(len(moving)), "fixed": int(len(fixed))},
        "meta": meta,
    }


def _preview_payload(
    sample: dict[str, Any],
    randomized_moving: np.ndarray,
    fixed: np.ndarray,
    preview_id: str,
    rotation: np.ndarray,
    center: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    manual_type = str(sample.get("manual_type", "Unavailable"))
    image_keys = sample.get("image_keys", {})
    category = str(sample.get("category", "Unlabeled"))
    display_instruction = _display_instruction(
        category,
        sample.get("source_instruction") or sample.get("instruction", ""),
    )
    model_instruction = str(sample.get("model_instruction") or _model_instruction(category))
    return {
        **{
            key: sample[key]
            for key in ("asset_name", "hdf5_key", "split", "images")
        },
        "category": category,
        "instruction": display_instruction,
        "source_instruction": sample.get("source_instruction", ""),
        "model_instruction": model_instruction,
        "manual_type": manual_type,
        "image_keys": image_keys,
        "preview_id": preview_id,
        "saved": False,
        "sample_dir": None,
        "point_clouds": {"moving": randomized_moving.tolist(), "fixed": fixed.tolist()},
        "point_counts": {"moving": int(len(randomized_moving)), "fixed": int(len(fixed))},
        "meta": {
            "asset_name": sample["asset_name"],
            "hdf5_key": sample["hdf5_key"],
            "split": sample["split"],
            "category": category,
            "instruction": display_instruction,
            "model_instruction": model_instruction,
            "source_instruction": sample.get("source_instruction", ""),
            "manual_type": manual_type,
            "image_keys": image_keys,
            "randomization": {
                "seed": int(seed),
                "centering_mode": "mean",
                "import_rotation": True,
            },
            "ground_truth_pose": {
                "translation": center.astype(np.float32).tolist(),
                "rotation_6d": _rotation_to_6d(rotation).tolist(),
                "rotation_matrix": rotation.tolist(),
            },
        },
    }


def _randomize_sample(payload: dict[str, Any]) -> dict[str, Any]:
    hdf5_path = Path(payload["hdf5_path"]).expanduser().resolve()
    split = str(payload.get("split", "test"))
    asset_name = str(payload["asset_name"])
    # Keep the GUI's seed contract aligned with the seed input's default value.
    # The browser sends the current value explicitly, but using zero here also
    # makes direct API calls deterministic when no seed is supplied.
    seed = int(payload.get("seed", 0))
    if seed < 0 or seed > MAX_SEED:
        raise ValueError(f"Seed must be an integer in the range 0..{MAX_SEED}.")
    centering_mode = str(payload.get("centering_mode", "mean")).lower()
    if centering_mode != "mean":
        raise ValueError("The GUI uses mean centering only.")
    sample = _load_hdf5_sample(hdf5_path, split, asset_name)
    if not sample.get("has_images"):
        raise ValueError(
            f"Asset {asset_name!r} has no complete image_base/image_assemble pair "
            "for the selected manual type."
        )
    moving = np.asarray(sample["point_clouds"]["moving"], dtype=np.float32)
    fixed = np.asarray(sample["point_clouds"]["fixed"], dtype=np.float32)
    center = moving.mean(axis=0).astype(np.float32)
    rotation = _random_rotation(seed)
    randomized_moving = (rotation @ (moving - center).T).T
    preview_id = uuid.uuid4().hex
    with SAMPLE_PREVIEWS_LOCK:
        SAMPLE_PREVIEWS[preview_id] = {
            "sample": sample,
            "moving": moving,
            "fixed": fixed,
            "randomized_moving": randomized_moving,
            "rotation": rotation,
            "center": center,
            "seed": seed,
        }
        _prune_sample_previews_locked()
    return _preview_payload(sample, randomized_moving, fixed, preview_id, rotation, center, seed)


def _save_sample_preview(preview_id: str) -> dict[str, Any]:
    with SAMPLE_PREVIEWS_LOCK:
        preview = SAMPLE_PREVIEWS.get(preview_id)
    if preview is None:
        raise KeyError("Randomization preview not found. Generate a new preview before saving.")
    sample = preview["sample"]
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    sample_dir = GENERATED_SAMPLES_ROOT / (
        f"{_safe_component(sample['asset_name'])}_{timestamp}_{uuid.uuid4().hex[:8]}"
    )
    temporary_sample_dir = sample_dir.with_name(f".{sample_dir.name}.partial")
    GENERATED_SAMPLES_ROOT.mkdir(parents=True, exist_ok=True)
    if sample_dir.exists():
        raise FileExistsError(f"Generated sample directory already exists: {sample_dir}")
    shutil.rmtree(temporary_sample_dir, ignore_errors=True)
    temporary_sample_dir.mkdir(parents=True, exist_ok=False)
    manual_type = str(sample.get("manual_type", MANUAL_TYPE_FREESTYLE))
    if manual_type.lower().startswith("line"):
        base_image_key, assemble_image_key = "image_base_lineart", "image_assemble_lineart"
    else:
        manual_type = MANUAL_TYPE_FREESTYLE
        base_image_key, assemble_image_key = "image_base_freestyle", "image_assemble_freestyle"
    files = {
        "partA_pc": "partA_pc.npy",
        "base_partB_pc": "base_partB_pc.npy",
        "original_partA_pc": "original_partA_pc.npy",
        base_image_key: f"{base_image_key}.png",
        assemble_image_key: f"{assemble_image_key}.png",
        "instruction": "instruction.txt",
        "category": "category.txt",
    }
    try:
        np.save(temporary_sample_dir / files["partA_pc"], preview["randomized_moving"])
        np.save(temporary_sample_dir / files["base_partB_pc"], preview["fixed"])
        np.save(temporary_sample_dir / files["original_partA_pc"], preview["moving"])
        if not _save_image_data_uri(
            sample["images"]["base"], temporary_sample_dir / files[base_image_key]
        ):
            raise ValueError(f"Unable to save {manual_type} base image.")
        if not _save_image_data_uri(
            sample["images"]["assemble"], temporary_sample_dir / files[assemble_image_key]
        ):
            raise ValueError(f"Unable to save {manual_type} assembly image.")
        (temporary_sample_dir / files["instruction"]).write_text(
            sample["instruction"] + "\n", encoding="utf-8"
        )
        (temporary_sample_dir / files["category"]).write_text(
            sample["category"] + "\n", encoding="utf-8"
        )
        meta = _preview_payload(
            sample,
            preview["randomized_moving"],
            preview["fixed"],
            preview_id,
            preview["rotation"],
            preview["center"],
            preview["seed"],
        )["meta"]
        meta.update({"created_at": datetime.now().astimezone().isoformat(), "files": files})
        (temporary_sample_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        temporary_sample_dir.replace(sample_dir)
    except Exception:
        shutil.rmtree(temporary_sample_dir, ignore_errors=True)
        raise
    with SAMPLE_PREVIEWS_LOCK:
        SAMPLE_PREVIEWS.pop(preview_id, None)
    return _saved_sample_payload(sample_dir)


def _font_name() -> str | None:
    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for font_path in candidates:
        if os.path.exists(font_path):
            try:
                font_manager.fontManager.addfont(font_path)
                return font_manager.FontProperties(fname=font_path).get_name()
            except Exception:
                continue
    return None


def analyze_hdf5(file_name: str) -> dict[str, Any]:
    """Analyze train/test split sizes and category counts in one HDF5 file."""
    raw_path = os.path.expandvars(os.path.expanduser(file_name.strip()))
    file_path = Path(raw_path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")

    counters = {"train": Counter(), "test": Counter()}
    split_sizes = {"train": 0, "test": 0}
    missing_categories = {"train": 0, "test": 0}
    with h5py.File(file_path, "r") as h5:
        objects = _find_objects(h5)
        for split in ("train", "test"):
            node = _find_split(h5, split)
            if node is None:
                raise KeyError(f"Missing {split} split (tried split/{split}, {split}, and related paths)")
            entries = _split_values(node)
            split_sizes[split] = len(entries)
            for asset_name, inline_category in entries:
                category = inline_category or _category_from_node(_find_object(objects, asset_name))
                category = (category or "Unlabeled").strip() or "Unlabeled"
                if category == "Unlabeled":
                    missing_categories[split] += 1
                counters[split][category] += 1

    category_names = sorted(
        set(counters["train"]) | set(counters["test"]),
        key=lambda name: (-(counters["train"][name] + counters["test"][name]), name.lower()),
    )
    rows = [
        {
            "category": category,
            "train": counters["train"][category],
            "test": counters["test"][category],
            "total": counters["train"][category] + counters["test"][category],
        }
        for category in category_names
    ]
    return {
        "file": str(file_path),
        "file_size_mb": file_path.stat().st_size / (1024 * 1024),
        "train_total": split_sizes["train"],
        "test_total": split_sizes["test"],
        "total": split_sizes["train"] + split_sizes["test"],
        "category_count": len(category_names),
        "missing_categories": missing_categories,
        "rows": rows,
    }


def _make_chart(result: dict[str, Any]) -> str:
    rows = result["rows"]
    categories = [row["category"] for row in rows]
    train = [row["train"] for row in rows]
    test = [row["test"] for row in rows]
    font_name = _font_name()
    if font_name:
        plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    width = max(12.0, min(24.0, 7.0 + len(categories) * 0.62))
    fig, ax = plt.subplots(figsize=(width, 7.4), dpi=160)
    positions = list(range(len(categories)))
    bar_width = 0.38
    train_bars = ax.bar(
        [position - bar_width / 2 for position in positions],
        train,
        bar_width,
        label="Train",
        color="#3b82f6",
        edgecolor="#1d4ed8",
        linewidth=0.6,
    )
    test_bars = ax.bar(
        [position + bar_width / 2 for position in positions],
        test,
        bar_width,
        label="Test",
        color="#f59e0b",
        edgecolor="#b45309",
        linewidth=0.6,
    )
    ax.set_title("HDF5 Category Distribution", fontsize=18, fontweight="bold", pad=16)
    ax.set_xlabel("Category", fontsize=12, labelpad=10)
    ax.set_ylabel("Sample Count", fontsize=12, labelpad=10)
    ax.set_xticks(positions)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
    maximum = max([*train, *test, 1])
    ax.set_ylim(0, maximum * 1.2 + 1)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    for bars in (train_bars, test_bars):
        for bar in bars:
            value = int(bar.get_height())
            if value:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + max(0.12, maximum * 0.015),
                    str(value),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _project_path(raw_path: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(raw_path).strip()))).resolve()


def _ensure_worker() -> subprocess.Popen[str]:
    global WORKER_PROCESS, WORKER_LOG_HANDLE
    if WORKER_PROCESS is not None and WORKER_PROCESS.poll() is None:
        return WORKER_PROCESS
    if WORKER_PROCESS is not None:
        # The previous worker may have crashed.  Release its log handle before
        # creating a replacement so repeated inference attempts do not leak
        # descriptors or keep stale logs open.
        WORKER_PROCESS = None
        if WORKER_LOG_HANDLE is not None:
            WORKER_LOG_HANDLE.close()
            WORKER_LOG_HANDLE = None
    requested_python = os.environ.get("ASSEMLM_GUI_PYTHON", sys.executable)
    python_bin = requested_python
    if not Path(python_bin).is_file():
        python_bin = shutil.which(requested_python) or sys.executable
    worker_env = os.environ.copy()
    model_root = os.environ.get("ASSEMLM_MODEL_ROOT", str(PROJECT_ROOT))
    worker_env["PYTHONPATH"] = os.pathsep.join(
        [model_root, worker_env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    run_dir = _initialize_runtime_logging()
    worker_log_path = run_dir / "inference_worker.log"
    WORKER_LOG_HANDLE = worker_log_path.open("a", encoding="utf-8")
    WORKER_PROCESS = subprocess.Popen(
        [python_bin, str(INFERENCE_WORKER_PATH)],
        cwd=str(PROJECT_ROOT),
        env=worker_env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=WORKER_LOG_HANDLE,
        text=True,
        bufsize=1,
    )
    return WORKER_PROCESS


def _drop_worker_locked(process: subprocess.Popen[str] | None = None) -> None:
    """Terminate a broken worker and release its pipe/log resources.

    The caller must hold ``WORKER_LOCK`` when this is used from a request.
    Keeping this cleanup in one place prevents malformed protocol responses
    from poisoning the next inference request.
    """
    global WORKER_PROCESS, WORKER_LOG_HANDLE
    current = process or WORKER_PROCESS
    if current is WORKER_PROCESS:
        WORKER_PROCESS = None
    if current is not None and current.poll() is None:
        try:
            terminate = getattr(current, "terminate", None)
            if terminate is None:
                current.kill()
            else:
                terminate()
            current.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                current.kill()
                current.wait(timeout=3)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
        except ProcessLookupError:
            pass
    log_handle = WORKER_LOG_HANDLE
    WORKER_LOG_HANDLE = None
    if log_handle is not None:
        log_handle.close()


def _worker_request(payload: dict[str, Any]) -> dict[str, Any]:
    global WORKER_PROCESS, WORKER_LOG_HANDLE
    with WORKER_LOCK:
        timeout_raw = os.environ.get("ASSEMLM_GUI_WORKER_TIMEOUT_SECONDS", "1800")
        try:
            timeout = float(timeout_raw)
        except ValueError:
            logger.warning(
                "Invalid ASSEMLM_GUI_WORKER_TIMEOUT_SECONDS=%r; using 1800 seconds.",
                timeout_raw,
            )
            timeout = 1800.0
        if timeout < 0:
            raise ValueError("ASSEMLM_GUI_WORKER_TIMEOUT_SECONDS must be non-negative.")
        process = _ensure_worker()
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Inference worker pipes are not available.")
        process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        process.stdin.flush()
        if timeout > 0:
            ready, _, _ = select.select([process.stdout], [], [], timeout)
            if not ready:
                _drop_worker_locked(process)
                raise TimeoutError(
                    f"Inference worker did not respond within {timeout:g} seconds. "
                    "Check inference_worker.log and retry."
                )
        response_line = process.stdout.readline()
        if not response_line:
            _drop_worker_locked(process)
            raise RuntimeError("Inference worker exited without a response. Check inference_worker.log.")
        try:
            response = json.loads(response_line)
        except json.JSONDecodeError as exc:
            _drop_worker_locked(process)
            raise RuntimeError(
                "Inference worker returned invalid JSON. Check inference_worker.log and retry."
            ) from exc
        if not isinstance(response, dict):
            _drop_worker_locked(process)
            raise RuntimeError("Inference worker returned an invalid protocol response.")
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Unknown inference worker error."))
        result = response.get("result")
        if not isinstance(result, dict):
            _drop_worker_locked(process)
            raise RuntimeError("Inference worker response is missing a result object.")
        return result


def _shutdown_worker() -> None:
    """Stop the private worker when the GUI process exits."""
    global WORKER_PROCESS, WORKER_LOG_HANDLE
    _drop_worker_locked()


atexit.register(_shutdown_worker)


def _run_inference_job(job_id: str, payload: dict[str, Any]) -> None:
    with INFERENCE_JOBS_LOCK:
        INFERENCE_JOBS[job_id]["state"] = "running"
        INFERENCE_JOBS[job_id]["message"] = "Loading model and running inference..."
    try:
        result = _worker_request({**payload, "job_id": job_id})
        with INFERENCE_JOBS_LOCK:
            INFERENCE_JOBS[job_id].update(state="done", message="Inference completed.", result=result)
    except Exception as exc:
        logger.exception("Inference job failed: %s", job_id)
        with INFERENCE_JOBS_LOCK:
            INFERENCE_JOBS[job_id].update(
                state="error",
                message=str(exc),
                traceback=traceback.format_exc(),
            )


def _run_model_load_job(job_id: str, payload: dict[str, Any]) -> None:
    global LOADED_MODEL
    with INFERENCE_JOBS_LOCK:
        INFERENCE_JOBS[job_id]["state"] = "running"
        INFERENCE_JOBS[job_id]["message"] = "Loading model weights..."
    try:
        result = _worker_request({**payload, "action": "load_model", "job_id": job_id})
        with LOADED_MODEL_LOCK:
            is_current = ACTIVE_MODEL_LOAD_JOB_ID == job_id
            if is_current:
                LOADED_MODEL = {
                    "checkpoint_path": result["checkpoint_path"],
                    "config_path": result["config_path"],
                }
        if not is_current:
            result = {**result, "superseded": True}
        with INFERENCE_JOBS_LOCK:
            INFERENCE_JOBS[job_id].update(
                state="done",
                message=(
                    "Model load superseded by a newer request."
                    if not is_current
                    else "Model loaded and cached."
                ),
                result=result,
            )
    except Exception as exc:
        logger.exception("Model load job failed: %s", job_id)
        with INFERENCE_JOBS_LOCK:
            INFERENCE_JOBS[job_id].update(
                state="error",
                message=str(exc),
                traceback=traceback.format_exc(),
            )


def _run_dataset_load_job(job_id: str, payload: dict[str, Any]) -> None:
    def update_progress(percent: float, _total: float, message: str) -> None:
        bounded = max(0.0, min(100.0, float(percent)))
        with DATASET_LOAD_JOBS_LOCK:
            job = DATASET_LOAD_JOBS.get(job_id)
            if job is not None:
                job.update(state="running", percent=bounded, message=message)

    with DATASET_LOAD_JOBS_LOCK:
        DATASET_LOAD_JOBS[job_id]["state"] = "running"
        DATASET_LOAD_JOBS[job_id]["message"] = "Loading HDF5 dataset..."
    try:
        result = _summarize_hdf5(payload["path"], progress_callback=update_progress)
        with DATASET_LOAD_JOBS_LOCK:
            DATASET_LOAD_JOBS[job_id].update(
                state="done",
                percent=100.0,
                message="Dataset loaded.",
                result=result,
            )
    except Exception as exc:
        logger.exception("Dataset load job failed: %s", job_id)
        with DATASET_LOAD_JOBS_LOCK:
            DATASET_LOAD_JOBS[job_id].update(
                state="error",
                message=str(exc),
                traceback=traceback.format_exc(),
            )


@app.post("/api/dataset/load")
def api_dataset_load():
    try:
        payload = request.get_json(silent=True) or {}
        file_path = _project_path(payload.get("path", ""))
        if not file_path.is_file():
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")
        return jsonify({"ok": True, "data": _summarize_hdf5(file_path)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/dataset/load/start")
def api_dataset_load_start():
    try:
        payload = request.get_json(silent=True) or {}
        file_path = _project_path(payload.get("path", ""))
        if not file_path.is_file():
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")
        job_id = uuid.uuid4().hex
        with DATASET_LOAD_JOBS_LOCK:
            _prune_dataset_load_jobs_locked()
            DATASET_LOAD_JOBS[job_id] = {
                "state": "queued",
                "percent": 0.0,
                "message": "Queued for dataset loading.",
                "created_at": datetime.now().astimezone().isoformat(),
            }
        threading.Thread(
            target=_run_dataset_load_job,
            args=(job_id, {"path": str(file_path)}),
            daemon=True,
        ).start()
        return jsonify({"ok": True, "job_id": job_id})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.get("/api/dataset/load/<job_id>")
def api_dataset_load_status(job_id: str):
    with DATASET_LOAD_JOBS_LOCK:
        job = DATASET_LOAD_JOBS.get(job_id)
        if job is None:
            return jsonify({"ok": False, "error": "Dataset load job not found."}), 404
        return jsonify({"ok": True, "job": job})


@app.post("/api/dataset/sample")
def api_dataset_sample():
    try:
        payload = request.get_json(silent=True) or {}
        file_path = _project_path(payload.get("path", ""))
        sample = _load_hdf5_sample(file_path, str(payload.get("split", "test")), str(payload["asset_name"]))
        return jsonify({"ok": True, "data": sample})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/sample/randomize")
def api_sample_randomize():
    try:
        payload = request.get_json(silent=True) or {}
        result = _randomize_sample(payload)
        return jsonify({"ok": True, "data": result})
    except Exception as exc:
        logger.exception("Sample randomization failed")
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/sample/save")
def api_sample_save():
    try:
        payload = request.get_json(silent=True) or {}
        result = _save_sample_preview(str(payload["preview_id"]))
        return jsonify({"ok": True, "data": result})
    except Exception as exc:
        logger.exception("Sample save failed")
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/sample/load")
def api_saved_sample_load():
    try:
        payload = request.get_json(silent=True) or {}
        sample_dir = _project_path(payload.get("sample_dir", ""))
        return jsonify({"ok": True, "data": _saved_sample_payload(sample_dir)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/inference/start")
def api_inference_start():
    try:
        payload = request.get_json(silent=True) or {}
        sample_dir = _project_path(payload.get("sample_dir", ""))
        if not (sample_dir / "meta.json").is_file():
            raise FileNotFoundError(f"Stage-one sample folder is invalid: {sample_dir}")
        with LOADED_MODEL_LOCK:
            loaded_model = dict(LOADED_MODEL) if LOADED_MODEL else None
        if loaded_model is None:
            raise RuntimeError("Load a model checkpoint and config before running inference.")
        job_id = uuid.uuid4().hex
        job_payload = {
            "sample_dir": str(sample_dir),
            "checkpoint_path": loaded_model["checkpoint_path"],
            "config_path": loaded_model["config_path"],
        }
        with INFERENCE_JOBS_LOCK:
            _reserve_job_slot_locked()
            INFERENCE_JOBS[job_id] = {
                "state": "queued",
                "message": "Queued for inference.",
                "created_at": datetime.now().astimezone().isoformat(),
            }
        threading.Thread(target=_run_inference_job, args=(job_id, job_payload), daemon=True).start()
        return jsonify({"ok": True, "job_id": job_id})
    except JobCapacityError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 429
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/inference/model/load")
def api_inference_model_load():
    try:
        payload = request.get_json(silent=True) or {}
        checkpoint_path = _project_path(payload.get("checkpoint_path", ""))
        config_path = _project_path(payload.get("config_path", ""))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        job_id = uuid.uuid4().hex
        job_payload = {"checkpoint_path": str(checkpoint_path), "config_path": str(config_path)}
        with INFERENCE_JOBS_LOCK:
            _reserve_job_slot_locked()
            INFERENCE_JOBS[job_id] = {
                "state": "queued",
                "message": "Queued for model loading.",
                "created_at": datetime.now().astimezone().isoformat(),
            }
        # Reserve capacity before invalidating the previous model.  A rejected
        # 429 request must not make an otherwise usable model disappear.
        global LOADED_MODEL, ACTIVE_MODEL_LOAD_JOB_ID
        with LOADED_MODEL_LOCK:
            LOADED_MODEL = None
            ACTIVE_MODEL_LOAD_JOB_ID = job_id
        threading.Thread(target=_run_model_load_job, args=(job_id, job_payload), daemon=True).start()
        return jsonify({"ok": True, "job_id": job_id})
    except JobCapacityError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 429
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.get("/api/inference/model/status")
def api_inference_model_status():
    with LOADED_MODEL_LOCK:
        return jsonify({"ok": True, "loaded": dict(LOADED_MODEL) if LOADED_MODEL else None})


@app.get("/api/inference/<job_id>")
def api_inference_status(job_id: str):
    with INFERENCE_JOBS_LOCK:
        job = INFERENCE_JOBS.get(job_id)
        if job is None:
            return jsonify({"ok": False, "error": "Inference job not found."}), 404
        return jsonify({"ok": True, "job": job})


PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AssemLM GUI</title>
  <style>
    :root { --ink:#152238; --muted:#687890; --line:#dce5f0; --blue:#2563eb; --blue-dark:#12305d; --bg:#eef3f8; --panel:#fff; --green:#0f766e; --red:#b42318; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:var(--bg); font-family:Inter,Arial,sans-serif; }
    button,input,select { font:inherit; } button { cursor:pointer; border:0; }
    .shell { max-width:1640px; margin:0 auto; padding:22px 24px 44px; }
    .topbar { display:flex; align-items:center; justify-content:space-between; gap:20px; margin-bottom:18px; }
    .brand { display:flex; align-items:center; gap:12px; font-size:20px; font-weight:800; letter-spacing:.02em; }
    .brand-mark { display:grid; place-items:center; width:38px; height:38px; color:white; background:#172554; border-radius:10px; }
    .nav { display:flex; gap:6px; padding:5px; border:1px solid var(--line); border-radius:12px; background:#fff; }
    .nav button { padding:10px 16px; color:var(--muted); border-radius:8px; background:transparent; font-weight:700; }
    .nav button.active { color:white; background:var(--blue); }
    .view { display:none; } .view.active { display:block; }
    .hero { padding:25px 30px; color:white; border-radius:18px; background:linear-gradient(115deg,#102a56,#1d4ed8 62%,#0f766e); box-shadow:0 16px 34px #17255422; }
    .eyebrow { margin:0 0 8px; color:#bfdbfe; font-size:12px; letter-spacing:.13em; text-transform:uppercase; }
    h1 { margin:0; font-size:30px; } h2 { margin:0; font-size:17px; } h3 { margin:0; font-size:14px; }
    .hero p:last-child { max-width:900px; margin:10px 0 0; color:#dbeafe; line-height:1.6; }
    .toolbar { display:flex; flex-wrap:wrap; gap:10px; align-items:end; margin-top:18px; padding:14px; border:1px solid var(--line); border-radius:14px; background:#fff; }
    .field { display:flex; flex-direction:column; gap:6px; min-width:150px; } .field.grow { flex:1; min-width:280px; }
    .field label { color:var(--muted); font-size:11px; font-weight:800; letter-spacing:.06em; text-transform:uppercase; }
    .field-note { color:var(--muted); font-size:11px; line-height:1.4; overflow-wrap:anywhere; }
    input[type=text],input[type=number],select { width:100%; min-height:40px; padding:9px 11px; color:var(--ink); border:1px solid #cbd7e6; border-radius:8px; outline:0; background:#fff; }
    input:focus,select:focus { border-color:var(--blue); box-shadow:0 0 0 3px #2563eb1a; }
    .primary { min-height:40px; padding:0 17px; color:white; border-radius:8px; background:var(--blue); font-weight:800; }
    .secondary { min-height:40px; padding:0 15px; color:var(--blue-dark); border:1px solid #bfdbfe; border-radius:8px; background:#eff6ff; font-weight:800; }
    .layout { display:grid; grid-template-columns:245px minmax(270px,330px) minmax(0,1fr); gap:14px; margin-top:14px; align-items:start; }
    .panel { min-width:0; padding:16px; border:1px solid var(--line); border-radius:14px; background:var(--panel); box-shadow:0 8px 24px #23395d0b; }
    .panel-head { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:12px; }
    .panel-note,.meta { color:var(--muted); font-size:12px; line-height:1.5; }
    .category-list,.sample-list { display:flex; flex-direction:column; gap:5px; max-height:560px; overflow:auto; }
    .category-item,.sample-item { display:flex; align-items:center; justify-content:space-between; gap:8px; width:100%; padding:10px; color:var(--ink); border:1px solid transparent; border-radius:8px; background:#f7f9fc; text-align:left; }
    .category-item:hover,.sample-item:hover,.category-item.active,.sample-item.active { border-color:#93c5fd; background:#eff6ff; }
    .category-item strong { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; } .count { color:var(--muted); font-size:12px; }
    .sample-item { display:block; } .sample-item small { display:block; margin-top:4px; color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .detail { display:flex; flex-direction:column; gap:14px; }
    .detail-grid { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:12px; }
    .image-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; } .image-grid figure { margin:0; } .image-grid img { display:block; width:100%; aspect-ratio:1; object-fit:cover; border-radius:9px; background:#e9eef5; } figcaption { margin-top:5px; color:var(--muted); font-size:11px; }
    .viewer { position:relative; min-height:330px; overflow:hidden; border:1px solid #dbe4ef; border-radius:10px; background:#0c1424; } .viewer canvas { display:block; width:100%; height:330px; }
    .viewer-label { position:absolute; top:10px; left:10px; z-index:1; padding:5px 8px; color:#dbeafe; border-radius:6px; background:#0c1424c7; font-size:11px; }
    .info-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; } .info { padding:10px; border:1px solid var(--line); border-radius:8px; background:#f8fafc; } .info label { display:block; color:var(--muted); font-size:11px; } .info strong { display:block; margin-top:4px; overflow-wrap:anywhere; }
    .code-block { max-height:190px; overflow:auto; margin:0; padding:12px; color:#dbeafe; border-radius:9px; background:#0f172a; font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace; white-space:pre-wrap; }
    .split-title { display:flex; align-items:center; gap:10px; margin-bottom:10px; } .split-pill { padding:4px 7px; color:#075985; border-radius:5px; background:#e0f2fe; font-size:11px; font-weight:800; text-transform:uppercase; }
    .check { display:flex; align-items:center; gap:8px; min-height:40px; color:var(--ink); font-size:13px; } .check input { width:16px; height:16px; }
    .status { min-height:22px; margin-top:10px; color:var(--muted); font-size:12px; } .status.error { color:var(--red); } .status.success { color:var(--green); }
    .progress { display:flex; flex-direction:column; gap:6px; margin-top:12px; } .progress-track { height:9px; overflow:hidden; border-radius:999px; background:#dbe7f5; } .progress-fill { width:0%; height:100%; border-radius:999px; background:linear-gradient(90deg,#2563eb,#0f766e); transition:width .18s ease; } .progress-text { color:var(--muted); font-size:12px; }
    .hidden { display:none !important; }
    .overview { display:flex; flex-direction:column; gap:12px; margin-top:14px; }
    .overview-stats { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }
    .stat-card { padding:14px 16px; border:1px solid var(--line); border-radius:10px; background:#fff; }
    .stat-card label { display:block; color:var(--muted); font-size:11px; font-weight:800; letter-spacing:.05em; text-transform:uppercase; }
    .stat-card strong { display:block; margin-top:6px; color:var(--blue-dark); font-size:24px; line-height:1; }
    .chart-panel { min-width:0; padding:16px; border:1px solid var(--line); border-radius:14px; background:var(--panel); box-shadow:0 8px 24px #23395d0b; }
    .chart-scroll { overflow-x:auto; padding-bottom:4px; }
    .chart-scroll svg { display:block; min-width:760px; width:100%; height:320px; }
    .split-control { display:flex; align-items:end; gap:10px; margin-top:14px; padding-top:14px; border-top:1px solid var(--line); }
    .split-control .field { min-width:180px; }
    .chart-axis { fill:var(--muted); font-size:11px; }
    .chart-label { fill:var(--ink); font-size:11px; }
    .chart-grid { stroke:#e7edf5; stroke-width:1; }
    .chart-legend { display:flex; gap:16px; color:var(--muted); font-size:12px; }
    .legend-item { display:flex; align-items:center; gap:6px; }
    .legend-swatch { width:10px; height:10px; border-radius:3px; }
    .empty { padding:32px 18px; color:var(--muted); border:1px dashed #b8c7da; border-radius:10px; background:#f8fafc; text-align:center; line-height:1.6; }
    .metrics { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:9px; } .metric { padding:12px; border:1px solid var(--line); border-radius:9px; background:#f8fafc; } .metric label { color:var(--muted); font-size:11px; } .metric strong { display:block; margin-top:5px; font-size:20px; }
    .inference-layout { display:grid; grid-template-columns:360px minmax(0,1fr); gap:14px; margin-top:14px; align-items:start; } .stack { display:flex; flex-direction:column; gap:12px; } .stack .field { min-width:0; }
    @media (max-width:1180px) { .layout { grid-template-columns:220px 280px minmax(0,1fr); } .inference-layout { grid-template-columns:1fr; } }
    @media (max-width:900px) { .topbar { align-items:flex-start; flex-direction:column; } .layout { grid-template-columns:1fr 1fr; } .detail { grid-column:1 / -1; } .detail-grid { grid-template-columns:1fr; } }
    @media (max-width:620px) { .shell { padding:14px 10px 28px; } .layout { grid-template-columns:1fr; } .detail { grid-column:auto; } .hero { padding:22px 18px; } .toolbar { align-items:stretch; flex-direction:column; } .field.grow { min-width:0; } .metrics,.info-grid,.overview-stats { grid-template-columns:1fr 1fr; } }
  </style>
</head>
<body>
<main class="shell">
  <header class="topbar"><div class="brand"><span class="brand-mark">A</span><span>AssemLM GUI</span></div><nav class="nav"><button class="active" data-view="builder">Sample Builder</button><button data-view="inference">Model Inference</button></nav></header>
  <section id="builder" class="view active">
    <div class="hero"><p class="eyebrow">Stage 01 - Sample Construction</p><h1>Build a single randomized assembly sample</h1><p>Load an HDF5 dataset, inspect category counts, select one asset, view its multimodal data, and export a randomized moving-part pose for inference.</p></div>
    <div class="toolbar"><div class="field grow"><label for="datasetPath">HDF5 dataset path</label><input id="datasetPath" type="text" autocomplete="off" value="{{ default_dataset }}" placeholder="/path/to/assemlm_partnet.hdf5"><div class="field-note">Example: /path/to/assemlm_partnet.hdf5</div></div><button id="loadDataset" class="primary">Load Dataset</button></div>
    <div id="datasetProgress" class="progress hidden"><div class="progress-track"><div id="datasetProgressFill" class="progress-fill"></div></div><div id="datasetProgressText" class="progress-text"></div></div>
    <div id="builderStatus" class="status"></div>
    <section id="datasetOverview" class="overview hidden" aria-live="polite">
      <div class="overview-stats">
        <div class="stat-card"><label>Total samples</label><strong id="overviewTotal">0</strong></div>
        <div class="stat-card"><label>Train samples</label><strong id="overviewTrain">0</strong></div>
        <div class="stat-card"><label>Test samples</label><strong id="overviewTest">0</strong></div>
        <div class="stat-card"><label>Categories</label><strong id="overviewCategories">0</strong></div>
      </div>
      <section class="chart-panel">
        <div class="panel-head"><div><h2>Category distribution</h2><div class="panel-note">Sample counts across the loaded dataset</div></div><div class="chart-legend"><span class="legend-item"><i class="legend-swatch" style="background:#2563eb"></i>Train</span><span class="legend-item"><i class="legend-swatch" style="background:#f59e0b"></i>Test</span></div></div>
        <div class="chart-scroll"><svg id="categoryChart" role="img" aria-label="Train and test sample counts by category"></svg></div>
        <div class="split-control"><div class="field"><label for="datasetSplit">Browse split</label><select id="datasetSplit"><option value="test">Test</option><option value="train">Train</option></select></div><div class="panel-note">Choose a split below to browse its categories and samples.</div></div>
      </section>
    </section>
    <div id="datasetWorkspace" class="layout hidden"><section class="panel"><div class="panel-head"><h2>Categories</h2><span id="categoryTotal" class="panel-note">0</span></div><div id="categoryList" class="category-list"><div class="empty">Load a dataset to begin.</div></div></section><section class="panel"><div class="panel-head"><h2>Samples</h2><span id="sampleTotal" class="panel-note">0</span></div><div id="sampleList" class="sample-list"><div class="empty">Select a category.</div></div></section><section id="sampleDetail" class="detail"><div class="panel empty">Select a sample to inspect its data.</div></section></div>
  </section>
  <section id="inference" class="view">
    <div class="hero"><p class="eyebrow">Stage 02 - Model Inference</p><h1>Run the reference AssemLM model</h1><p>Provide a checkpoint, its training config, and a Stage 01 sample folder. The model worker loads each checkpoint once and reuses it for later inference jobs.</p></div>
    <div class="inference-layout"><section class="panel stack"><div class="panel-head"><div><h2>Model</h2><div class="panel-note">Load once, reuse across samples</div></div></div><div class="field"><label for="checkpointPath">Checkpoint file</label><input id="checkpointPath" type="text" autocomplete="off" value="{{ default_checkpoint }}" placeholder="/path/to/pytorch_model/mp_rank_00_model_states.pt"><div class="field-note">Example: /path/to/pytorch_model/mp_rank_00_model_states.pt</div></div><div class="field"><label for="configPath">Config YAML</label><input id="configPath" type="text" autocomplete="off" value="{{ default_config }}" placeholder="/path/to/config.yaml"><div class="field-note">Example: /path/to/config.yaml</div></div><div class="field-note">CLI prefill: --dataset PATH --checkpoint PATH --config PATH</div><button id="loadModel" class="primary">Load Model</button><div id="modelStatus" class="status"></div><div class="panel-head"><div><h2>Stage 01 sample</h2><div class="panel-note">Switch folders without reloading the model</div></div></div><div class="field"><label for="samplePath">Stage 01 sample folder</label><input id="samplePath" type="text" autocomplete="off" placeholder=".../generated_samples/object_timestamp"></div><button id="inspectStageSample" class="secondary">Inspect Stage 01 Sample</button><button id="runInference" class="primary">Run Inference</button><div id="inferenceStatus" class="status"></div></section><section id="inferenceDetail" class="detail"><div class="panel empty">Load a model and inspect a Stage 01 folder before running inference.</div></section></div>
  </section>
</main>
<script>
const state = {summary:null,selectedCategory:null,selectedSample:null,sample:null,stageSample:null,previewId:null,pollTimer:null,modelPollTimer:null,datasetLoadTimer:null,modelLoaded:null,modelLoadToken:0,sampleLoadToken:0,randomizeToken:0,randomSeed:0,inferenceRunToken:0,inferenceResult:null};
const $ = id => document.getElementById(id);
function setStatus(id,text,kind=''){const node=$(id);node.textContent=text||'';node.className=`status ${kind}`;}
async function api(url,options={}){let response;try{response=await fetch(url,{headers:{'Content-Type':'application/json'},...options});}catch(error){throw new Error(`Network request failed: ${error.message}`);}let data;try{data=await response.json();}catch(error){throw new Error(`Server returned an invalid response (${response.status}).`);}if(!response.ok||data.ok===false)throw new Error(data.error||`Request failed (${response.status})`);return data;}
function esc(value){const node=document.createElement('div');node.textContent=value??'';return node.innerHTML;}
function fmt(value,digits=5){return Number.isFinite(Number(value))?Number(value).toFixed(digits):'n/a';}
function formatCount(value){return Number(value||0).toLocaleString();}
function renderDatasetOverview(){const summary=state.summary;if(!summary)return;const overview=$('datasetOverview');overview.classList.remove('hidden');$('overviewTotal').textContent=formatCount(summary.total);$('overviewTrain').textContent=formatCount(summary.splits?.train?.total);$('overviewTest').textContent=formatCount(summary.splits?.test?.total);$('overviewCategories').textContent=formatCount(summary.category_count);const categories=summary.categories||[];const svg=$('categoryChart');const width=Math.max(760,categories.length*82+74),height=320,left=48,right=22,top=22,bottom=78,plotWidth=width-left-right,plotHeight=height-top-bottom;const maxValue=Math.max(1,...categories.map(item=>Math.max(item.train||0,item.test||0)));const tickStep=Math.max(1,Math.ceil(maxValue/4));const maxTick=tickStep*4;const y=value=>top+plotHeight-(value/maxTick)*plotHeight;const parts=[];for(let tick=0;tick<=4;tick++){const value=tick*tickStep;const yy=y(value);parts.push(`<line class="chart-grid" x1="${left}" y1="${yy}" x2="${width-right}" y2="${yy}"></line><text class="chart-axis" x="${left-9}" y="${yy+4}" text-anchor="end">${formatCount(value)}</text>`);}const groupWidth=plotWidth/Math.max(categories.length,1);const barWidth=Math.min(22,Math.max(10,groupWidth*.24));categories.forEach((item,index)=>{const center=left+groupWidth*(index+.5);const train=item.train||0,test=item.test||0;const trainHeight=(train/maxTick)*plotHeight,testHeight=(test/maxTick)*plotHeight;const label=esc(item.category);parts.push(`<rect x="${center-barWidth-2}" y="${y(train)}" width="${barWidth}" height="${trainHeight}" rx="3" fill="#2563eb"><title>${label}: ${formatCount(train)} train</title></rect><rect x="${center+2}" y="${y(test)}" width="${barWidth}" height="${testHeight}" rx="3" fill="#f59e0b"><title>${label}: ${formatCount(test)} test</title></rect><text class="chart-label" x="${center}" y="${height-bottom+18}" text-anchor="end" transform="rotate(-42 ${center} ${height-bottom+18})">${label}</text>`);});svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.setAttribute('width',width);svg.innerHTML=parts.join('');}
function renderCategories(){const list=$('categoryList');const categories=state.summary?.categories||[];const split=state.summary?.activeSplit||'test';$('categoryTotal').textContent=`${categories.length} categories`;list.innerHTML='';const all=document.createElement('button');all.className=`category-item ${state.selectedCategory===null?'active':''}`;all.innerHTML=`<strong>All categories</strong><span class="count">${state.summary?.splits[split]?.total||0}</span>`;all.onclick=()=>selectCategory(null);list.appendChild(all);categories.forEach(item=>{const count=item[split]||0;if(!count)return;const button=document.createElement('button');button.className=`category-item ${state.selectedCategory===item.category?'active':''}`;button.innerHTML=`<strong title="${esc(item.category)}">${esc(item.category)}</strong><span class="count">${count}</span>`;button.onclick=()=>selectCategory(item.category);list.appendChild(button);});}
function renderSamples(){const split=state.summary?.splits[state.summary?.activeSplit];const samples=(split?.samples||[]).filter(item=>!state.selectedCategory||item.category===state.selectedCategory);const list=$('sampleList');$('sampleTotal').textContent=`${samples.length} samples`;list.innerHTML='';if(!samples.length){list.innerHTML='<div class="empty">No samples match this category.</div>';return;}samples.forEach(item=>{const button=document.createElement('button');button.className=`sample-item ${state.selectedSample===item.asset_name?'active':''}`;button.innerHTML=`<strong>${esc(item.asset_name)}</strong><small>${esc(item.category)}</small>`;button.onclick=()=>loadSample(item);list.appendChild(button);});}
function selectCategory(category){++state.sampleLoadToken;++state.randomizeToken;state.selectedCategory=category;state.selectedSample=null;state.sample=null;state.previewId=null;renderCategories();renderSamples();$('sampleDetail').innerHTML='<div class="panel empty">Select a sample to inspect its data.</div>';}
function setDatasetProgress(percent,message){const value=Math.max(0,Math.min(100,Number(percent)||0));$('datasetProgressFill').style.width=value+'%';$('datasetProgressText').textContent=(message||'Loading dataset')+' · '+Math.round(value)+'%';}
function applyDatasetSummary(summary){state.summary=summary;state.summary.activeSplit=$('datasetSplit').value;state.selectedCategory=null;state.selectedSample=null;renderDatasetOverview();renderCategories();renderSamples();$('sampleDetail').innerHTML='<div class="panel empty">Select a sample to inspect its data.</div>';$('datasetWorkspace').classList.remove('hidden');}
function pollDatasetLoad(jobId,token){clearInterval(state.datasetLoadTimer);const check=async function(){try{const data=await api('/api/dataset/load/'+jobId);if(token!==state.sampleLoadToken)return;const job=data.job;setDatasetProgress(job.percent,job.message||job.state);if(job.state==='done'){clearInterval(state.datasetLoadTimer);state.datasetLoadTimer=null;$('datasetProgress').classList.add('hidden');$('loadDataset').disabled=false;applyDatasetSummary(job.result);setStatus('builderStatus',`${formatCount(state.summary.total)} samples loaded from ${state.summary.file}`,'success');}else if(job.state==='error'){clearInterval(state.datasetLoadTimer);state.datasetLoadTimer=null;$('datasetProgress').classList.add('hidden');$('loadDataset').disabled=false;$('datasetOverview').classList.add('hidden');$('datasetWorkspace').classList.add('hidden');setStatus('builderStatus',job.message||'Dataset load failed.','error');}}catch(error){if(token!==state.sampleLoadToken)return;clearInterval(state.datasetLoadTimer);state.datasetLoadTimer=null;$('datasetProgress').classList.add('hidden');$('loadDataset').disabled=false;$('datasetOverview').classList.add('hidden');$('datasetWorkspace').classList.add('hidden');setStatus('builderStatus',error.message,'error');}};state.datasetLoadTimer=setInterval(check,400);check();}
async function loadDataset(){const token=++state.sampleLoadToken;++state.randomizeToken;clearInterval(state.datasetLoadTimer);state.datasetLoadTimer=null;try{setStatus('builderStatus','Loading dataset...');$('loadDataset').disabled=true;$('datasetProgress').classList.remove('hidden');setDatasetProgress(0,'Starting dataset load');$('datasetOverview').classList.add('hidden');$('datasetWorkspace').classList.add('hidden');const data=await api('/api/dataset/load/start',{method:'POST',body:JSON.stringify({path:$('datasetPath').value})});if(token!==state.sampleLoadToken)return;pollDatasetLoad(data.job_id,token);}catch(error){if(token!==state.sampleLoadToken)return;$('datasetProgress').classList.add('hidden');$('loadDataset').disabled=false;$('datasetOverview').classList.add('hidden');$('datasetWorkspace').classList.add('hidden');setStatus('builderStatus',error.message,'error');}}
async function loadSample(item){const token=++state.sampleLoadToken;++state.randomizeToken;try{state.selectedSample=item.asset_name;state.sample=null;state.previewId=null;state.randomSeed=0;renderSamples();$('sampleDetail').innerHTML='<div class="panel empty">Loading sample...</div>';setStatus('builderStatus',`Loading ${item.asset_name}...`);const data=await api('/api/dataset/sample',{method:'POST',body:JSON.stringify({path:$('datasetPath').value,split:$('datasetSplit').value,asset_name:item.asset_name})});if(token!==state.sampleLoadToken)return;state.sample=data.data;renderSampleDetail(state.sample,$('sampleDetail'),true);setStatus('builderStatus','Sample loaded.','success');}catch(error){if(token!==state.sampleLoadToken)return;state.sample=null;state.previewId=null;$('sampleDetail').innerHTML='<div class="panel empty">Select a sample to inspect its data.</div>';setStatus('builderStatus',error.message,'error');}}
function sampleInfo(sample){
 return '<div class="panel"><div class="split-title"><span class="split-pill">' + esc(sample.split||'stage 01') + '</span><h2>' + esc(sample.asset_name||'Sample') + '</h2></div>' +
   '<div class="info-grid"><div class="info"><label>Category</label><strong>' + esc(sample.category||'Unlabeled') + '</strong></div>' +
   '<div class="info"><label>Instruction type</label><strong>Natural language</strong></div>' +
   '<div class="info"><label>Manual type</label><strong>' + esc(sample.manual_type||'Unavailable') + '</strong></div>' +
   '<div class="info"><label>Moving points</label><strong>' + (sample.point_counts&&sample.point_counts.moving||0) + '</strong></div>' +
   '<div class="info"><label>Fixed points</label><strong>' + (sample.point_counts&&sample.point_counts.fixed||0) + '</strong></div></div>' +
   '<div class="meta" style="margin-top:10px">' + (sample.sample_dir?('Saved folder: '+esc(sample.sample_dir)):'Source HDF5 sample') + '</div></div>';
}
function mediaBlock(sample){
 return '<div class="panel"><div class="panel-head"><h2>Reference images</h2><span class="panel-note">Manual: ' + esc(sample.manual_type||'Unavailable') + '</span></div><div class="image-grid">' +
   '<figure>' + (sample.images&&sample.images.base?'<img src="' + sample.images.base + '" alt="Base state">':'<div class="empty">Unavailable</div>') + '<figcaption>Base state</figcaption></figure>' +
   '<figure>' + (sample.images&&sample.images.assemble?'<img src="' + sample.images.assemble + '" alt="Assembly state">':'<div class="empty">Unavailable</div>') + '<figcaption>Assembly state</figcaption></figure>' +
   '</div></div>';
}
let viewerCount=0;
function pointBlock(title,groups){
 const id='viewer-'+(++viewerCount);
 setTimeout(function(){mountPointCloud(id,groups);},0);
 return '<div class="panel"><div class="panel-head"><h2>' + esc(title) + '</h2><span class="panel-note">Drag to orbit - scroll to zoom</span></div>' +
   '<div id="' + id + '" class="viewer"><span class="viewer-label">Interactive 3D · O / XYZ</span></div></div>';
}
function renderSampleDetail(sample,container,controls){
 const pc=sample.point_clouds||{};
 let html=sampleInfo(sample)+mediaBlock(sample);
 html += '<div class="detail-grid">' +
   pointBlock('Input point clouds',[{points:pc.moving,color:0x94a3b8},{points:pc.fixed,color:0x38bdf8}]) +
   pointBlock('Moving part view',[{points:pc.moving,color:0xf59e0b}]) +
   '</div>';
 html += '<div class="panel"><div class="panel-head"><h2>Sample metadata</h2></div><div class="info-grid">' +
   '<div class="info"><label>Instruction</label><strong>' + esc(sample.instruction||'No instruction stored') + '</strong></div>' +
   '<div class="info"><label>Asset key</label><strong>' + esc(sample.hdf5_key||'n/a') + '</strong></div>' +
   '<div class="info"><label>Source split</label><strong>' + esc(sample.split||'n/a') + '</strong></div>' +
   '</div></div>';
 if(controls){
   const seed=Number.isFinite(Number(state.randomSeed))?Math.trunc(Number(state.randomSeed)):0;
   html += '<div class="panel"><div class="panel-head"><h2>Randomize moving part A</h2><span class="panel-note">Rotation only - mean centered - reference seed</span></div>' +
     '<div class="toolbar" style="margin-top:0;padding:0;border:0"><div class="field"><label for="randomSeed">Seed</label>' +
     '<input id="randomSeed" type="number" min="0" max="4294967294" value="' + seed + '">' +
     '</div><button id="randomizeSample" class="primary">Randomize</button><button id="saveSample" class="secondary" disabled>Save sample</button></div>' +
     '<div id="randomizeStatus" class="status"></div></div>';
 }
 container.innerHTML=html;
 if(controls){
   $('randomizeSample').onclick=randomizeCurrentSample;
   $('saveSample').onclick=saveCurrentPreview;
 }
}
function mountCanvasPointCloud(id,groups){
 const container=$(id);
 if(!container)return;
 if(typeof container.__viewerCleanup==='function')container.__viewerCleanup();
 container.innerHTML='';
 const canvas=document.createElement('canvas');
 canvas.className='point-canvas';
 container.appendChild(canvas);
 const context=canvas.getContext('2d');
 const points=[];
 groups.forEach(function(group,groupIndex){
   (group.points||[]).forEach(function(point,index){
     if(Array.isArray(point)&&point.length>=3){
       points.push({x:Number(point[0]),y:Number(point[1]),z:Number(point[2]),color:group.color||0xffffff,groupIndex:groupIndex,index:index});
     }
   });
 });
 if(!points.length){
   container.innerHTML='<div class="empty">No xyz points available.</div>';
   return;
 }
 const stride=Math.max(1,Math.ceil(points.length/2600));
 const sampled=points.filter(function(_,index){return index%stride===0;});
 // Include the world origin in the framing bounds so O and all three axes
 // remain visible even when the cloud is entirely on one side of the origin.
 const mins=[0,1,2].map(function(axis){
   const values=sampled.map(function(point){return [point.x,point.y,point.z][axis];});
   return Math.min(0,Math.min.apply(null,values));
 });
 const maxs=[0,1,2].map(function(axis){
   const values=sampled.map(function(point){return [point.x,point.y,point.z][axis];});
   return Math.max(0,Math.max.apply(null,values));
 });
 const center=[0,1,2].map(function(axis){return (mins[axis]+maxs[axis])/2;});
 const scale=Math.max.apply(null,maxs.map(function(max,axis){return max-mins[axis];}).concat([1e-5]));
 let yaw=.72,pitch=.38,zoom=1.0,dragging=false,lastX=0,lastY=0;
 const projectPoint=function(x,y,z,width,height){
   const nx=(x-center[0])/scale;
   const ny=(y-center[1])/scale;
   const nz=(z-center[2])/scale;
   const cosY=Math.cos(yaw),sinY=Math.sin(yaw),cosP=Math.cos(pitch),sinP=Math.sin(pitch);
   const rx=nx*cosY-nz*sinY;
   const rz=nx*sinY+nz*cosY;
   const ry=ny*cosP-rz*sinP;
   const depth=ny*sinP+rz*cosP;
   return {x:width/2+rx*width*.82*zoom,y:height/2-ry*height*.82*zoom,depth:depth};
 };
 const resize=function(){
   const rect=container.getBoundingClientRect();
   const dpr=Math.min(window.devicePixelRatio||1,2);
   canvas.width=Math.max(260,Math.floor(rect.width*dpr));
   canvas.height=Math.floor(330*dpr);
   canvas.style.width='100%';
   canvas.style.height='330px';
   context.setTransform(dpr,0,0,dpr,0,0);
   draw();
 };
 const colorHex=function(value){
   const hex=(Number(value)||0xffffff).toString(16).padStart(6,'0').slice(-6);
   return '#'+hex;
 };
 const drawAxes=function(width,height){
   const axisLength=scale*.48;
   const axes=[
     {label:'X',color:'#ef4444',vector:[1,0,0]},
     {label:'Y',color:'#22c55e',vector:[0,1,0]},
     {label:'Z',color:'#3b82f6',vector:[0,0,1]}
   ];
   axes.forEach(function(axis){
     const negative=projectPoint(-axisLength*axis.vector[0],-axisLength*axis.vector[1],-axisLength*axis.vector[2],width,height);
     const positive=projectPoint(axisLength*axis.vector[0],axisLength*axis.vector[1],axisLength*axis.vector[2],width,height);
     context.globalAlpha=.78;
     context.strokeStyle=axis.color;
     context.lineWidth=1.6;
     context.beginPath();
     context.moveTo(negative.x,negative.y);
     context.lineTo(positive.x,positive.y);
     context.stroke();
     context.globalAlpha=1;
     context.fillStyle=axis.color;
     context.font='bold 12px Arial';
     context.fillText(axis.label,positive.x+5,positive.y-5);
   });
 };
 const draw=function(){
   const width=canvas.clientWidth||container.clientWidth||260;
   const height=330;
   context.clearRect(0,0,width,height);
   context.fillStyle='#0c1424';
   context.fillRect(0,0,width,height);
   const projected=sampled.map(function(point){
     const projectedPoint=projectPoint(point.x,point.y,point.z,width,height);
     return {x:projectedPoint.x,y:projectedPoint.y,depth:projectedPoint.depth,color:colorHex(point.color)};
   });
   projected.sort(function(a,b){return a.depth-b.depth;});
   projected.forEach(function(point){
     const radius=Math.max(1.2,2.8+point.depth*1.4);
     context.globalAlpha=.84;
     context.fillStyle=point.color;
     context.beginPath();
     context.arc(point.x,point.y,radius,0,Math.PI*2);
     context.fill();
   });
   context.globalAlpha=1;
   drawAxes(width,height);
   const origin=projectPoint(0,0,0,width,height);
   context.fillStyle='#f8fafc';
   context.beginPath();
   context.arc(origin.x,origin.y,4.5,0,Math.PI*2);
   context.fill();
   context.strokeStyle='#0c1424';
   context.lineWidth=1.5;
   context.stroke();
   context.fillStyle='#f8fafc';
   context.font='bold 11px Arial';
   context.fillText('O',origin.x+7,origin.y-7);
   context.fillStyle='#b9c8df';
   context.font='11px Arial';
   context.fillText('Drag to orbit - scroll to zoom',18,22);
 };
 canvas.addEventListener('pointerdown',function(event){
   dragging=true;
   lastX=event.clientX;
   lastY=event.clientY;
   canvas.setPointerCapture(event.pointerId);
 });
 canvas.addEventListener('pointermove',function(event){
   if(!dragging)return;
   yaw+=(event.clientX-lastX)*.012;
   pitch=Math.max(-1.45,Math.min(1.45,pitch+(event.clientY-lastY)*.012));
   lastX=event.clientX;
   lastY=event.clientY;
   draw();
 });
 canvas.addEventListener('pointerup',function(){dragging=false;});
 canvas.addEventListener('pointercancel',function(){dragging=false;});
 canvas.addEventListener('wheel',function(event){
   event.preventDefault();
   zoom=Math.max(.35,Math.min(3.2,zoom*(event.deltaY<0?1.1:.9)));
   draw();
 },{passive:false});
 window.addEventListener('resize',resize);
 container.__viewerCleanup=function(){window.removeEventListener('resize',resize);};
 resize();
}
function mountPointCloud(id,groups){
 const container=$(id);
 if(!container)return;
 if(typeof container.__viewerCleanup==='function')container.__viewerCleanup();
 container.__viewerCleanup=null;
 if(window.THREE&&THREE.OrbitControls){
   try{
     const width=Math.max(container.clientWidth,260),height=330;
     const scene=new THREE.Scene();
     scene.background=new THREE.Color(0x0c1424);
     const camera=new THREE.PerspectiveCamera(45,width/height,.01,1000);
     const renderer=new THREE.WebGLRenderer({antialias:true});
     renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
     renderer.setSize(width,height);
     container.innerHTML='';
     container.appendChild(renderer.domElement);
     const controls=new THREE.OrbitControls(camera,renderer.domElement);
     controls.enableDamping=true;
     const all=groups.flatMap(function(group){return group.points||[];}).filter(function(point){return Array.isArray(point)&&point.length>=3;});
     if(!all.length){mountCanvasPointCloud(id,groups);return;}
     const stride=Math.max(1,Math.ceil(all.length/2400));
     const sampled=all.filter(function(_,index){return index%stride===0;});
     const mins=[0,1,2].map(function(axis){return Math.min(0,Math.min.apply(null,sampled.map(function(point){return point[axis];})));});
     const maxs=[0,1,2].map(function(axis){return Math.max(0,Math.max.apply(null,sampled.map(function(point){return point[axis];})));});
     const center=mins.map(function(min,axis){return (min+maxs[axis])/2;});
     const span=Math.max.apply(null,maxs.map(function(max,axis){return max-mins[axis];}).concat([1e-4]));
     groups.forEach(function(group){
       const points=(group.points||[]).filter(function(_,index){return index%stride===0;});
       const positions=new Float32Array(points.length*3);
       points.forEach(function(point,index){
         positions[index*3]=point[0]-center[0];
         positions[index*3+1]=point[1]-center[1];
         positions[index*3+2]=point[2]-center[2];
       });
       const geometry=new THREE.BufferGeometry();
       geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));
       scene.add(new THREE.Points(geometry,new THREE.PointsMaterial({color:group.color||0xffffff,size:Math.max(span/130,.002),sizeAttenuation:true})));
     });
     const axesHelper=new THREE.AxesHelper(span*.48);
     axesHelper.position.set(-center[0],-center[1],-center[2]);
     scene.add(axesHelper);
     const originMarker=new THREE.Mesh(
       new THREE.SphereGeometry(Math.max(span*.018,.004),16,8),
       new THREE.MeshBasicMaterial({color:0xf8fafc})
     );
     originMarker.position.set(-center[0],-center[1],-center[2]);
     scene.add(originMarker);
     camera.position.set(span*1.5,span*1.2,span*1.5);
     controls.target.set(0,0,0);
     controls.update();
     const animate=function(){
       if(!document.body.contains(renderer.domElement))return;
       requestAnimationFrame(animate);
       controls.update();
       renderer.render(scene,camera);
     };
     animate();
     return;
   }catch(error){
     console.warn('WebGL viewer unavailable; using canvas viewer.',error);
   }
 }
 mountCanvasPointCloud(id,groups);
}
async function randomizeCurrentSample(){
 if(!state.sample)return;
 const token=++state.randomizeToken;
 const sampleToken=state.sampleLoadToken;
 try{
   const input=$('randomSeed');
   const parsed=Number(input&&input.value);
   const seed=Number.isFinite(parsed)?Math.trunc(parsed):0;
   state.randomSeed=seed;
   setStatus('randomizeStatus','Generating preview...');
   const data=await api('/api/sample/randomize',{
     method:'POST',
     body:JSON.stringify({
       hdf5_path:$('datasetPath').value,
       split:$('datasetSplit').value,
       asset_name:state.sample.asset_name,
       seed:seed,
       centering_mode:'mean'
     })
   });
   if(token!==state.randomizeToken||sampleToken!==state.sampleLoadToken)return;
   state.previewId=data.data.preview_id;
   const returnedSeed=Number(data.data.meta&&data.data.meta.randomization&&data.data.meta.randomization.seed);
   if(Number.isFinite(returnedSeed))state.randomSeed=Math.trunc(returnedSeed);
   renderSampleDetail(data.data,$('sampleDetail'),true);
   $('saveSample').disabled=false;
   setStatus('randomizeStatus','Preview ready. Save it when the visualization is correct.','success');
 }catch(error){
   if(token!==state.randomizeToken||sampleToken!==state.sampleLoadToken)return;
   setStatus('randomizeStatus',error.message,'error');
 }
}
async function saveCurrentPreview(){
 if(!state.previewId)return;
 try{
   setStatus('randomizeStatus','Saving sample...');
   const data=await api('/api/sample/save',{method:'POST',body:JSON.stringify({preview_id:state.previewId})});
   state.stageSample=data.data;
   state.previewId=null;
   $('samplePath').value=state.stageSample.sample_dir;
   renderSampleDetail(state.stageSample,$('sampleDetail'),true);
   $('saveSample').disabled=true;
   setStatus('randomizeStatus','Saved to '+state.stageSample.sample_dir,'success');
 }catch(error){
   setStatus('randomizeStatus',error.message,'error');
 }
}
async function loadModel(){const token=++state.modelLoadToken;clearInterval(state.modelPollTimer);state.modelPollTimer=null;state.modelLoaded=null;state.inferenceResult=null;++state.inferenceRunToken;clearInterval(state.pollTimer);state.pollTimer=null;setInferenceButtonState(false);clearInferenceDetail('Loading a new model...');try{setStatus('modelStatus','Loading model weights...');const data=await api('/api/inference/model/load',{method:'POST',body:JSON.stringify({checkpoint_path:$('checkpointPath').value,config_path:$('configPath').value})});if(token!==state.modelLoadToken)return;pollModel(data.job_id,token);}catch(error){if(token===state.modelLoadToken){setStatus('modelStatus',error.message,'error');clearInferenceDetail('Model could not be loaded.');}}}
function pollModel(jobId,token){clearInterval(state.modelPollTimer);const check=async function(){try{const data=await api(`/api/inference/${jobId}`);if(token!==state.modelLoadToken)return;const job=data.job;setStatus('modelStatus',job.message||job.state,job.state==='error'?'error':job.state==='done'?'success':'');if(job.state==='done'){clearInterval(state.modelPollTimer);state.modelPollTimer=null;state.modelLoaded=job.result;setStatus('inferenceStatus','Model ready. Inspect a Stage 01 sample.','success');}else if(job.state==='error'){clearInterval(state.modelPollTimer);state.modelPollTimer=null;}}catch(error){if(token!==state.modelLoadToken)return;clearInterval(state.modelPollTimer);state.modelPollTimer=null;setStatus('modelStatus',error.message,'error');}};state.modelPollTimer=setInterval(check,1200);check();}
function setInferenceButtonState(running){
 const button=$('runInference');
 if(!button)return;
 button.disabled=running;
 button.textContent=running?'Running...':'Run Inference';
}
function clearInferenceDetail(message){
 const detail=$('inferenceDetail');
 if(detail)detail.innerHTML='<div class="panel empty">'+esc(message)+'</div>';
}
function renderInferenceDetail(sample,result=null){
 const pc=sample.point_clouds||{};
 let html=sampleInfo(sample)+mediaBlock(sample);
 html += '<div class="detail-grid">' +
   pointBlock('Stage 01 input',[{points:pc.moving,color:0xf59e0b},{points:pc.fixed,color:0x38bdf8}]) +
   (result ? pointBlock('Predicted assembly',[{points:result.predicted_point_cloud,color:0xef4444},{points:result.fixed_point_cloud,color:0x38bdf8}]) : '<div class="panel empty">Run inference to view the predicted pose and transformed point cloud.</div>') +
   '</div>';
 if(result){
   const metrics=result.metrics||{};
   const pose={translation:result.pred_translation,rotation_6d:result.pred_rotation_6d,rotation_matrix:result.pred_rotation_matrix};
   html += '<div class="panel"><div class="panel-head"><h2>Inference metrics</h2><span class="panel-note">Model cache: ' + (result.model_cache_hit?'reused':'loaded now') + '</span></div>' +
     '<div class="metrics"><div class="metric"><label>RMSE(T)</label><strong>' + fmt(metrics.RMSE_T) + '</strong></div>' +
     '<div class="metric"><label>CD</label><strong>' + fmt(metrics.CD) + '</strong></div>' +
     '<div class="metric"><label>CD(R)</label><strong>' + fmt(metrics.CD_R) + '</strong></div></div>' +
     '<h3 style="margin:16px 0 8px">Predicted pose</h3><pre class="code-block">' + esc(JSON.stringify(pose,null,2)) + '</pre>' +
     '<div class="meta" style="margin-top:10px">Result folder: ' + esc(result.files&&result.files.predicted_partA_pc||'') + '</div></div>';
 }
 $('inferenceDetail').innerHTML=html;
}
async function inspectStageSample(invalidateInference=true){
 if(invalidateInference){
   ++state.inferenceRunToken;
   clearInterval(state.pollTimer);
   state.pollTimer=null;
   state.inferenceResult=null;
   setInferenceButtonState(false);
 }
 const samplePath=String($('samplePath').value||'').trim();
 try{
   if(!samplePath)throw new Error('Enter a Stage 01 sample folder before inspecting it.');
   setStatus('inferenceStatus','Loading stage sample...');
   state.stageSample=null;
   state.inferenceResult=null;
   clearInferenceDetail('Loading Stage 01 sample...');
   const data=await api('/api/sample/load',{method:'POST',body:JSON.stringify({sample_dir:samplePath})});
   state.stageSample=data.data;
   renderInferenceDetail(state.stageSample);
   setStatus('inferenceStatus','Stage sample loaded.','success');
   return state.stageSample;
 }catch(error){
   state.stageSample=null;
   state.inferenceResult=null;
   clearInferenceDetail('Unable to load Stage 01 sample.');
   setStatus('inferenceStatus',error.message,'error');
   return null;
 }
}
async function runInference(){
 const token=++state.inferenceRunToken;
 clearInterval(state.pollTimer);
 state.pollTimer=null;
 state.inferenceResult=null;
 setInferenceButtonState(true);
 clearInferenceDetail('Preparing a fresh inference run...');
 try{
   if(!state.modelLoaded)throw new Error('Load a model before running inference.');
   // Always reload the folder currently entered by the user. This prevents a
   // second run from rendering the previous sample/result on the right.
   const sample=await inspectStageSample(false);
   if(!sample)throw new Error('Stage 01 sample could not be loaded.');
   if(token!==state.inferenceRunToken)return;
   setStatus('inferenceStatus','Starting inference...');
   const data=await api('/api/inference/start',{method:'POST',body:JSON.stringify({sample_dir:String($('samplePath').value||'').trim()})});
   if(token!==state.inferenceRunToken)return;
   pollInference(data.job_id,token);
 }catch(error){
   if(token===state.inferenceRunToken){
     setStatus('inferenceStatus',error.message,'error');
     clearInferenceDetail('Inference could not be started.');
     setInferenceButtonState(false);
   }
 }
}
function pollInference(jobId,token){
 clearInterval(state.pollTimer);
 const check=async function(){
   try{
     const data=await api('/api/inference/'+jobId);
     if(token!==state.inferenceRunToken)return;
     const job=data.job;
     setStatus('inferenceStatus',job.message||job.state,job.state==='error'?'error':job.state==='done'?'success':'');
     if(job.state==='done'){
       clearInterval(state.pollTimer);
       state.pollTimer=null;
       state.inferenceResult=job.result;
       renderInferenceDetail(state.stageSample,job.result);
       setInferenceButtonState(false);
     }else if(job.state==='error'){
       clearInterval(state.pollTimer);
       state.pollTimer=null;
       clearInferenceDetail('Inference failed. See the status message for details.');
       setInferenceButtonState(false);
     }
   }catch(error){
     if(token!==state.inferenceRunToken)return;
     clearInterval(state.pollTimer);
     state.pollTimer=null;
     setStatus('inferenceStatus',error.message,'error');
     clearInferenceDetail('Inference status could not be retrieved.');
     setInferenceButtonState(false);
   }
 };
 state.pollTimer=setInterval(check,1200);
 check();
}
document.querySelectorAll('.nav button').forEach(button=>button.onclick=()=>{document.querySelectorAll('.nav button').forEach(item=>item.classList.remove('active'));document.querySelectorAll('.view').forEach(item=>item.classList.remove('active'));button.classList.add('active');$(button.dataset.view).classList.add('active');});
$('loadDataset').onclick=loadDataset;$('datasetSplit').onchange=()=>{if(state.summary){++state.sampleLoadToken;state.summary.activeSplit=$('datasetSplit').value;state.selectedCategory=null;state.selectedSample=null;state.sample=null;state.previewId=null;renderCategories();renderSamples();$('sampleDetail').innerHTML='<div class="panel empty">Select a sample to inspect its data.</div>';}};$('loadModel').onclick=loadModel;$('inspectStageSample').onclick=inspectStageSample;$('runInference').onclick=runInference;
if($('datasetPath').value.trim())loadDataset();
if($('checkpointPath').value.trim()&&$('configPath').value.trim())loadModel();
</script>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template_string(
        PAGE,
        default_dataset=DEFAULT_DATASET_PATH,
        default_checkpoint=DEFAULT_CHECKPOINT_PATH,
        default_config=DEFAULT_CONFIG_PATH,
    )


@app.get("/health")
def health():
    return {"ok": True, "service": "assemlm-gui"}


def main() -> None:
    parser = argparse.ArgumentParser(description="AssemLM graphical web interface")
    parser.add_argument("--host", default="0.0.0.0", help="listen address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="listen port (default: 7860)")
    parser.add_argument("--debug", action="store_true", help="enable Flask debug mode")
    parser.add_argument("--dataset", default="", help="optional HDF5 dataset path to prefill and auto-load")
    parser.add_argument("--checkpoint", default="", help="optional model checkpoint path to prefill and auto-load")
    parser.add_argument("--config", default="", help="optional model config YAML path to prefill and auto-load")
    args = parser.parse_args()
    global DEFAULT_DATASET_PATH, DEFAULT_CHECKPOINT_PATH, DEFAULT_CONFIG_PATH
    DEFAULT_DATASET_PATH = str(_project_path(args.dataset)) if args.dataset.strip() else ""
    DEFAULT_CHECKPOINT_PATH = str(_project_path(args.checkpoint)) if args.checkpoint.strip() else ""
    DEFAULT_CONFIG_PATH = str(_project_path(args.config)) if args.config.strip() else ""
    run_dir = _initialize_runtime_logging()
    logger.info("AssemLM GUI running at http://%s:%s", args.host, args.port)
    logger.info("Run directory: %s", run_dir)
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
