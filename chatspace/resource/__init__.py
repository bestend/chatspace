"""
Copyright 2019 Pingpong AI Research, ScatterLab

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os

RESOURCE_PATH = os.path.dirname(os.path.realpath(__file__))

VOCAB_PATH = os.path.join(RESOURCE_PATH, "vocab.txt")
MODEL_PATH = os.path.join(RESOURCE_PATH, "model/model.pt")
JIT_MODEL_PATH = os.path.join(RESOURCE_PATH, "model/model.jit.pt")
CONFIG_PATH = os.path.join(RESOURCE_PATH, "config.json")

__all__ = ["RESOURCE_PATH", "CONFIG_PATH", "VOCAB_PATH", "MODEL_PATH", "JIT_MODEL_PATH"]
