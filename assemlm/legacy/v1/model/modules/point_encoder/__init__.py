

def get_point_encoder(type):
    if type == "vn_dgcnn":
        from .vn_dgcnn import PointEncoder
        return PointEncoder
    else:
        raise ValueError(f"Point encoder model {type} not supported")