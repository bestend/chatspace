import os

from chatspace.resources import CONFIG_PATH, RESOURCE_PATH, VOCAB_PATH


def test_resources_are_all_set():
    assert os.path.exists(RESOURCE_PATH)
    assert os.path.exists(VOCAB_PATH)
    assert os.path.exists(CONFIG_PATH)
