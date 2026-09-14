"""Point-and-vision language model interface."""

from .pvlm import _PVLM_Interface


def get_pvlm_model(config):
    if str(config.framework.get("pvlm_py", "pvlm")) != "pvlm":
        raise ValueError("The v2 release supports only framework.pvlm_py=pvlm.")
    return _PVLM_Interface(config)


__all__ = ["_PVLM_Interface", "get_pvlm_model"]
