"""Framework registry for the AssemLM 2.0 release."""


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._registry = {}

    def register(self, key: str):
        def decorator(value):
            self._registry[key] = value
            return value

        return decorator

    def __getitem__(self, key):
        return self._registry[key]


FRAMEWORK_REGISTRY = Registry("frameworks")
