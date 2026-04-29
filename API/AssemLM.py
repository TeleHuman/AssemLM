import os
import uuid
import base64
import numpy as np
from flask import Flask, request, jsonify
import argparse
from omegaconf import OmegaConf
from assemlm.model.framework import build_framework
from pathlib import Path
from assemlm.model.modules.point_encoder.vn_dgcnn.utils import bgs
from assemlm.utils.visualize_utils import save_multi_part_pointcloud_png
import torch
from assemlm.model.framework.AssemLMHF import AssemLMHF
from PIL import Image

app = Flask(__name__)

# Absolute path to this file
this_file = Path(__file__).resolve()

# Directory containing this file
this_dir = this_file.parent # /teamspace/jz/project/AssemLM/API
UPLOAD_FOLDER = os.path.join(this_dir, 'tmp')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def decode_base64_to_file(base64_str, prefix="image"):
    filename = f"{UPLOAD_FOLDER}/{prefix}_{uuid.uuid4().hex}.png"
    with open(filename, "wb") as f:
        f.write(base64.b64decode(base64_str))
    return filename

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", default="")
parser.add_argument("--port", type=int, default=25557)
args, clipargs = parser.parse_known_args()
model_path = args.model_path # /teamspace/jz/project/AssemLM/models/AssemLM-v1
port = args.port

assemlm_model = AssemLMHF.from_pretrained(model_path)


@app.route('/query', methods=['POST'])
def query():
    payload = request.get_json(silent=True)
    print("Received query:", payload)

    results_root = this_dir / "results_tmp"
    results_root.mkdir(parents=True, exist_ok=True)

    records = payload.get("records", []) if isinstance(payload, dict) else []
    asset_count = len(records)
    batch_count = asset_count // 4
    usable_asset_count = batch_count * 4

    for batch_index in range(batch_count):
        batch_records = records[batch_index * 4:(batch_index + 1) * 4]
        batch_point_clouds = []
        batch_fixed_point_clouds = []
        batch_moving_point_clouds = []
        batch_imgs = []
        instructions = []
        for record in batch_records:
            file_info = record.get("files", {})
            base_part_b_path = file_info.get("base_partB_pc")
            part_a_path = file_info.get("partA_pc")
            image_base_path = file_info.get("image_base_freestyle")
            image_assemble_path = file_info.get("image_assemble_freestyle")
            instruction_path = file_info.get("instruction")
            if base_part_b_path is None or part_a_path is None:
                raise KeyError("Each record must contain 'files.base_partB_pc' and 'files.partA_pc'.")
            if image_base_path is None or image_assemble_path is None:
                raise KeyError("Each record must contain 'files.image_base_freestyle' and 'files.image_assemble_freestyle'.")
            if instruction_path is None:
                raise KeyError("Each record must contain 'files.instruction'.")

            base_part_b = np.load(base_part_b_path).astype(np.float32)
            part_a = np.load(part_a_path).astype(np.float32)
            image_base = Image.open(image_base_path).convert("RGB")
            image_assemble = Image.open(image_assemble_path).convert("RGB")
            with open(instruction_path, "r", encoding="utf-8") as f:
                instruction = f.read().strip()
            
            batch_point_clouds.append(np.stack([part_a.T, base_part_b.T], axis=0)) # 4 * (2, 3, 1024)
            batch_fixed_point_clouds.append(base_part_b.T)
            batch_moving_point_clouds.append(part_a.T)
            batch_imgs.append([image_base, image_assemble]) # 4*2*PIL shape: (4, 2, H, W, 3), H, W = 576
            instructions.append(instruction) # 4
        
        batch = assemlm_model.assemlm_interface.build_assemlm_inputs(images=batch_imgs,point_clouds=batch_point_clouds, instructions=instructions)
        

        generated_poses = assemlm_model.generate(batch)
        valid_mask = (generated_poses[:, 0] != -100.0) & (generated_poses != 0).any(dim=-1)
        pred_t = generated_poses[:, :3].view(-1, 3, 1)
        pred_rot_6d = generated_poses[:, 3:9]
        pred_R = bgs(pred_rot_6d.reshape(-1, 2, 3).permute(0, 2, 1))

        for i in range(len(batch_moving_point_clouds)):
            p_R, p_t = pred_R[i], pred_t[i]
            pc_a = torch.from_numpy(batch_moving_point_clouds[i]).to(p_R.device)
            pc_b = torch.from_numpy(batch_fixed_point_clouds[i]).to(p_R.device)
            p_pc_a = p_R.T @ pc_a + p_t
            sample_name = batch_records[i].get("asset_name", f"sample_{batch_index}_{i}")
            sample_dir = results_root / f"batch_{batch_index:04d}" / sample_name.replace("/", "_")
            sample_dir.mkdir(parents=True, exist_ok=True)

            save_multi_part_pointcloud_png(
                [p_pc_a, pc_b], 
                str(sample_dir), 
                f"{sample_name.replace('/', '_')}_pred.png",
                colors=['red', 'blue'], # Prediction: Red (A), Blue (B)
                point_size=50,
            )

            prediction_payload = {
                "batch_index": batch_index,
                "sample_index": i,
                "asset_name": sample_name,
                "valid_mask": bool(valid_mask[i].item()) if hasattr(valid_mask[i], "item") else bool(valid_mask[i]),
                "pred_translation": pred_t[i].squeeze(-1).detach().cpu().tolist(),
                "pred_rotation_6d": pred_rot_6d[i].detach().cpu().tolist(),
                "pred_rotation_matrix": pred_R[i].detach().cpu().tolist(),
            }
            with open(sample_dir / "prediction.json", "w", encoding="utf-8") as f:
                import json
                json.dump(prediction_payload, f, ensure_ascii=False, indent=2)


    return jsonify({
        "ok": True,
        "asset_count": asset_count,
        "usable_asset_count": usable_asset_count,
        "batch_count": batch_count,
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)