"""VN-DGCNN patch encoder factory for AssemLM 2.0."""


def get_point_encoder(encoder_type):
    if encoder_type != "vn_dgcnn_patch":
        raise ValueError("The v2 release supports only point_encoder.type=vn_dgcnn_patch.")
    from .vn_dgcnn import VN_DGCNN_Patch

    return VN_DGCNN_Patch
