import os

RESOURCE_PATH = os.path.dirname(os.path.realpath(__file__))

MODEL_PATH = os.path.join(RESOURCE_PATH, "chatspace_model/")
CONFIG_PATH = os.path.join(RESOURCE_PATH, "config.json")
VOCAB_PATH = os.path.join(RESOURCE_PATH, "vocab")

__all__ = ["RESOURCE_PATH", "MODEL_PATH", "CONFIG_PATH", "VOCAB_PATH"]
