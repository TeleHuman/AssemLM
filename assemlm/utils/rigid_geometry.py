"""Rotation representation conversion used by the HDF5 dataset."""


def matrix_to_rotation_6d(rotation):
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"rotation must end in [3, 3], got {tuple(rotation.shape)}")
    return rotation[..., :, :2].transpose(-1, -2).reshape(*rotation.shape[:-2], 6)
