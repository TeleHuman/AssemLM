"""Automatic discovery of the processed AssemLM v2 HDF5 datasets."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    """Resolved path for one processed dataset."""

    dataset_name: str
    hdf5_path: str


# ``DATASETS_LEGACY`` is retained as the public registry name used by the
# dataloader. It now contains aliases for files discovered from the data root,
# rather than a hard-coded list of ``*_final_hdf5`` names.
DATASETS_LEGACY: dict[str, DatasetConfig] = {}
CANONICAL_DATASETS: dict[str, DatasetConfig] = {}


def _default_dataset_root() -> Path:
    # dataset_mixture.py lives in AssemLM/assemlm/dataloader/assemble.
    return Path(__file__).resolve().parents[4] / "datasets" / "processed" / "v2"


def _dataset_root() -> Path:
    value = os.environ.get("ASSEMLM_DATA_ROOT")
    return Path(value).expanduser().resolve() if value else _default_dataset_root()


def _short_name(path: Path) -> str:
    stem = path.stem.lower()
    return stem.removeprefix("assemlm_")


def _override_path(short_name: str) -> Path | None:
    """Return an optional per-dataset override for compatibility."""
    candidates = (
        f"ASSEMLM_{short_name.upper()}_HDF5",
        f"ASSEMLM_{short_name.upper()}_FINAL_HDF5",
    )
    for variable in candidates:
        value = os.environ.get(variable)
        if value:
            return Path(value).expanduser().resolve()
    return None


def _discover_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"AssemLM dataset root does not exist: {root}")
    paths = {
        path.resolve()
        for pattern in ("assemlm_*.hdf5", "assemlm_*.h5")
        for path in root.glob(pattern)
        if path.is_file()
    }
    if not paths:
        raise FileNotFoundError(
            f"No assemlm_*.hdf5 or assemlm_*.h5 files found under {root}"
        )
    return sorted(paths, key=lambda path: path.name.lower())


def register_datasets_mixtures() -> None:
    """Discover every ``assemlm_*`` file below ``ASSEMLM_DATA_ROOT``.

    The canonical key is the filename without the ``assemlm_`` prefix and
    extension (for example, ``partnet``). Historical keys such as
    ``partnet_final_hdf5`` remain accepted as aliases so old experiment
    overrides continue to work.
    """
    DATASETS_LEGACY.clear()
    CANONICAL_DATASETS.clear()
    root = _dataset_root()
    paths_by_name: dict[str, Path] = {}
    for path in _discover_paths(root):
        name = _short_name(path)
        if not name:
            continue
        paths_by_name[name] = path

    # Explicit overrides may point outside the root, but only for a dataset
    # whose ``assemlm_<name>`` file is present. This prevents accidentally
    # introducing a second, implicit dataset into an automatic mixture.
    for name, path in list(paths_by_name.items()):
        override = _override_path(name)
        if override is not None:
            paths_by_name[name] = override

    for name in sorted(paths_by_name):
        config = DatasetConfig(name, str(paths_by_name[name]))
        CANONICAL_DATASETS[name] = config
        aliases = {
            name,
            f"{name}_hdf5",
            f"{name}_final_hdf5",
            f"assemlm_{name}",
            f"assemlm_{name}_hdf5",
        }
        for alias in aliases:
            DATASETS_LEGACY[alias.lower()] = config


def available_dataset_names() -> list[str]:
    """Return canonical names in deterministic order."""
    if not CANONICAL_DATASETS:
        register_datasets_mixtures()
    return sorted(CANONICAL_DATASETS)


def resolve_dataset_name(name: str) -> str:
    """Resolve a canonical name or a historical alias to its canonical key."""
    if not DATASETS_LEGACY:
        register_datasets_mixtures()
    key = str(name).strip().lower()
    try:
        return DATASETS_LEGACY[key].dataset_name
    except KeyError as exc:
        available = ", ".join(available_dataset_names())
        raise KeyError(f"Unknown dataset {name!r}. Available datasets: {available}") from exc
