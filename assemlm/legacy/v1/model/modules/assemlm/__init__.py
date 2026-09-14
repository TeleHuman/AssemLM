
def get_assemlm_model(config):
    assemlm_py = config.framework.assemlm_py
    if assemlm_py == "assemlm":
        from .assemlm import _AssemLM_Interface
        return _AssemLM_Interface(config)
    else:
        raise ValueError(f"AssemLM model {assemlm_py} not supported")