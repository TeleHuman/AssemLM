import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional, List, Any
import os

def _ensure_numpy_pc(pc):
    if hasattr(pc, 'cpu'):
        pc = pc.cpu()
    if hasattr(pc, 'numpy'):
        pc = pc.numpy()
    if pc.shape[0] == 3 and pc.shape[1] != 3:
        pc = pc.T
    return pc

def save_multi_part_pointcloud_png(pcs: List[Union[np.ndarray, Any]],
                                 save_dir: str,
                                 filename: str,
                                 colors: Optional[List[Any]] = None,
                                 point_size: int = 10,
                                 dpi: int = 200,
                                 elev: Optional[float] = 30,
                                 azim: Optional[float] = 45) -> str:
    os.makedirs(save_dir, exist_ok=True)
    
    if not filename.lower().endswith('.png'):
        filename += '.png'
    out_path = os.path.join(save_dir, filename)

    fig = plt.figure(figsize=(10, 10), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    ax.view_init(elev=elev, azim=azim)

    if colors is None:
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i % 10) for i in range(len(pcs))]

    for i, pc in enumerate(pcs):
        pc = _ensure_numpy_pc(pc)
        c = colors[i] if i < len(colors) else None
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], 
                   c=[c] if isinstance(c, (list, tuple, np.ndarray)) else c, 
                   marker='.', s=point_size, linewidth=0, alpha=1, label=f'Part {i}')

    all_points = np.concatenate([_ensure_numpy_pc(p) for p in pcs], axis=0)
    max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                         all_points[:, 1].max() - all_points[:, 1].min(),
                         all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0

    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

    return os.path.abspath(out_path)
