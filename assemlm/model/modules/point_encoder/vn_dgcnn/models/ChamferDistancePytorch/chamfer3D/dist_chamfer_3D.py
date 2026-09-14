from torch import nn
from torch.autograd import Function
import torch
import importlib
import importlib.util
import os
chamfer_3D = None

def load_chamfer():
    global chamfer_3D
    if chamfer_3D is not None:
        return chamfer_3D

    import importlib
    import os
    from torch.utils.cpp_extension import load

    # importlib.find_loader was removed in Python 3.12.
    # Use importlib.util.find_spec for cross-version compatibility.
    chamfer_found = importlib.util.find_spec("chamfer_3D") is not None
    if not chamfer_found:
        cur_path = os.path.dirname(os.path.abspath(__file__))
        source_files = [
            os.path.join(cur_path, "chamfer_cuda.cpp"),
            os.path.join(cur_path, "chamfer3D.cu"),
        ]
        missing = [path for path in source_files if not os.path.isfile(path)]
        if missing:
            raise RuntimeError(
                "Chamfer CUDA sources are missing from the installation: "
                + ", ".join(missing)
                + ". Reinstall the AssemLM package with native package data."
            )
        build_path = cur_path.replace('chamfer3D', 'tmp')
        os.makedirs(build_path, exist_ok=True)

        try:
            chamfer_3D = load(
                name="chamfer_3D",
                sources=source_files,
                build_directory=build_path,
            )
        except Exception as exc:
            raise RuntimeError(
                "Unable to build the Chamfer CUDA extension. Ensure nvcc, the "
                "selected PyTorch CUDA toolchain, and ninja are installed."
            ) from exc
    else:
        import chamfer_3D
    return chamfer_3D

# Chamfer's distance module @thibaultgroueix
# GPU tensors only
class chamfer_3DFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        if xyz1.ndim != 3 or xyz2.ndim != 3 or xyz1.size(-1) != 3 or xyz2.size(-1) != 3:
            raise ValueError(
                "Chamfer distance expects [B, N, 3] and [B, M, 3] tensors, "
                f"got {tuple(xyz1.shape)} and {tuple(xyz2.shape)}."
            )
        if not xyz1.is_cuda or not xyz2.is_cuda:
            raise RuntimeError("AssemLM Chamfer distance requires CUDA tensors.")
        if xyz1.device != xyz2.device:
            raise ValueError(
                f"Chamfer inputs must share a device, got {xyz1.device} and {xyz2.device}."
            )
        if xyz1.dtype != xyz2.dtype:
            raise ValueError(
                f"Chamfer inputs must share a dtype, got {xyz1.dtype} and {xyz2.dtype}."
            )
        _chamfer_3D = load_chamfer()
        batchsize, n, dim = xyz1.size()
        _, m, dim = xyz2.size()
        device = xyz1.device

        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.to(device)
        dist2 = dist2.to(device)
        idx1 = idx1.to(device)
        idx2 = idx2.to(device)
        torch.cuda.set_device(device)

        _chamfer_3D.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        _chamfer_3D = load_chamfer()
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        device = graddist1.device

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        gradxyz1 = gradxyz1.to(device)
        gradxyz2 = gradxyz2.to(device)
        _chamfer_3D.backward(
            xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
        )
        return gradxyz1, gradxyz2


class chamfer_3DDist(nn.Module):
    def __init__(self):
        super(chamfer_3DDist, self).__init__()

    def forward(self, input1, input2):
        input1 = input1.contiguous()
        input2 = input2.contiguous()
        return chamfer_3DFunction.apply(input1, input2)
