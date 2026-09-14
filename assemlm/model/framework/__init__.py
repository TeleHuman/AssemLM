"""Minimal framework factory for AssemLM2."""

from __future__ import annotations

from assemlm.model.tools import FRAMEWORK_REGISTRY


def build_framework(cfg):
    framework_name = str(getattr(cfg.framework, "name", "AssemLM2"))
    if framework_name != "AssemLM2":
        raise ValueError(f"The v2 release supports only framework.name=AssemLM2, got {framework_name!r}.")
    from assemlm.model.framework.AssemLM2 import AssemLM2

    if "AssemLM2" not in FRAMEWORK_REGISTRY._registry:
        FRAMEWORK_REGISTRY.register("AssemLM2")(AssemLM2)
    return AssemLM2(cfg)


__all__ = ["FRAMEWORK_REGISTRY", "build_framework"]
