from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List
from urllib import error, request
from assemlm.utils.visualize_utils import save_multi_part_pointcloud_png
import h5py
import numpy as np
from PIL import Image
import torch
import random
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HDF5 = PROJECT_ROOT / "datasets" / "demo.hdf5"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets_tmp"
DEFAULT_API_URL = "http://127.0.0.1:25557/query"


def decode_scalar(value: Any) -> str:
	if isinstance(value, bytes):
		return value.decode("utf-8")
	return str(value)


def ensure_dir(path: Path) -> Path:
	path.mkdir(parents=True, exist_ok=True)
	return path

def bgs(d6s):
	bsz = d6s.shape[0]
	b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
	a2 = d6s[:, :, 1]
	b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
	b3 = torch.cross(b1, b2, dim=1)
	return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)

def random_rotation_matrix():
	rotmat = bgs(torch.rand(1, 6).reshape(-1, 2, 3).permute(0, 2, 1))
	return rotmat[0].cpu().numpy().astype(np.float32)


def center_and_rotate_part_a(pcs: np.ndarray, rng: np.random.Generator) -> Dict[str, Any]:
	pc_center = (pcs.max(axis=0, keepdims=True) + pcs.min(axis=0, keepdims=True)) / 2
	pc_center = pc_center[0]
	new_pcs = pcs - pc_center
	rotmat = random_rotation_matrix()
	new_pcs = (rotmat.reshape(3, 3) @ new_pcs.T).T
	gt_rot = rotmat[:, :2].T.reshape(6).astype(np.float32)
	state = "up" if rotmat[2, 2] >= 0 else "down"
	return {
		"center": pc_center.astype(np.float32),
		"rotmat": rotmat,
		"rot6d": gt_rot,
		"state": state,
		"pcs": new_pcs.astype(np.float32),
	}


def save_image(array: np.ndarray, path: Path) -> str:
	image = Image.fromarray(array.astype(np.uint8))
	image.save(path)
	return str(path)


def save_text(text: str, path: Path) -> str:
	path.write_text(text, encoding="utf-8")
	return str(path)


def save_point_cloud(array: np.ndarray, path: Path) -> str:
	np.save(path, array.astype(np.float32))
	return str(path)


def build_asset_payload(asset_dir: Path, asset_name: str, grp, rng: np.random.Generator) -> Dict[str, Any]:
	category = decode_scalar(grp["category"][()])
	raw_instruction = decode_scalar(grp["instruction"][()]) if "instruction" in grp else ""

	base_part_b = grp["base_partB-pc"][()].astype(np.float32)
	part_a = grp["partA-pc"][()].astype(np.float32)

	transformed = center_and_rotate_part_a(part_a, rng)
	part_a_transformed = transformed["pcs"]

	image_base = save_image(grp["image_base_freestyle"][()], asset_dir / "image_base_freestyle.png")
	image_assemble = save_image(grp["image_assemble_freestyle"][()], asset_dir / "image_assemble_freestyle.png")

	base_pc_path = save_point_cloud(base_part_b, asset_dir / "base_partB_pc.npy")
	part_a_pc_path = save_point_cloud(part_a_transformed, asset_dir / "partA_pc.npy")

	generated_instruction = f"Assemble the {category} object"
	instruction_path = save_text(generated_instruction, asset_dir / "instruction.txt")
	
	save_multi_part_pointcloud_png(
		[part_a_transformed, base_part_b],
		str(asset_dir),
		"input_pc.png",
		colors=["gray", "blue"],
		point_size=50,
	)

	meta = {
		"asset_name": asset_name,
		"category": category,
		"state": transformed["state"],
		"files": {
			"base_partB_pc": base_pc_path,
			"partA_pc": part_a_pc_path,
			"image_base_freestyle": image_base,
			"image_assemble_freestyle": image_assemble,
			"instruction": instruction_path,
		},
	}
	(asset_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
	return meta


def post_json(api_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
	body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
	req = request.Request(
		api_url,
		data=body,
		headers={"Content-Type": "application/json"},
		method="POST",
	)
	opener = request.build_opener(request.ProxyHandler({}))
	with opener.open(req, timeout=60) as resp:
		text = resp.read().decode("utf-8")
		return {
			"status": resp.status,
			"body": text,
		}


def main() -> int:
	parser = argparse.ArgumentParser(description="Prepare AssemLM eval samples and post them to the API.")
	parser.add_argument("--hdf5-path", type=Path, default=DEFAULT_HDF5)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--split", default="test", choices=["test", "train"])
	parser.add_argument("--num-assets-batch", type=int, default=1)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--api-url", default=DEFAULT_API_URL)
	parser.add_argument("--no-post", action="store_true")
	args = parser.parse_args()
    
	split_name = args.split
	rng = np.random.default_rng()
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)

	output_dir = ensure_dir(args.output_dir)
	assets_dir = ensure_dir(output_dir / "assets")

	records: List[Dict[str, Any]] = []
	with h5py.File(args.hdf5_path, "r") as h5f:
		if "split" not in h5f or split_name not in h5f["split"]:
			raise KeyError(f"Split '{split_name}' not found in {args.hdf5_path}")

		split_values = h5f["split"][split_name][()]
		split_names_all = [decode_scalar(x) for x in split_values]
		assets_num = args.num_assets_batch * 4
		assets_num = min(assets_num, len(split_names_all))
		split_names = rng.choice(split_names_all, size=assets_num, replace=False).tolist()
		
		for asset_name in split_names:
			hdf5_key = asset_name.replace("/", "_")
			if hdf5_key not in h5f["objs"]:
				raise KeyError(f"Asset '{asset_name}' not found as '{hdf5_key}' in objs")

			asset_dir = ensure_dir(assets_dir / hdf5_key)
			grp = h5f["objs"][hdf5_key]
			record = build_asset_payload(asset_dir, asset_name, grp, rng)
			record["asset_dir"] = str(asset_dir)
			record["hdf5_key"] = hdf5_key
			records.append(record)
	manifest = {
		"hdf5_path": str(args.hdf5_path),
		"split": split_name,
		"num_assets": len(records),
		"records": records,
	}
	manifest_path = output_dir / "manifest.json"
	manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

	if not args.no_post:
		try:
			response = post_json(args.api_url, manifest)
			print(json.dumps({"posted": True, "response": response}, ensure_ascii=False, indent=2))
		except error.URLError as exc:
			print(json.dumps({"posted": False, "error": str(exc), "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))
	else:
		print(json.dumps({"posted": False, "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))

	return 0
if __name__ == "__main__":
    raise SystemExit(main())