import pytest
import tensorflow as tf

from chatspace.dataset import make_chatspace_training_dataset
from chatspace.resources import VOCAB_PATH

BOS_TOKEN = 1
EOS_TOKEN = 2


@pytest.fixture(scope="session")
def lookup_table():
    with open(VOCAB_PATH, "r") as f:
        content = f.read()
    keys = ["<PAD>", "<s>", "</s>", "<UNK>"] + list(content)
    values = list(range(len(keys)))
    initializer = tf.lookup.KeyValueTensorInitializer(
        keys[:2000], values[:2000], key_dtype=tf.string, value_dtype=tf.int32
    )
    return tf.lookup.StaticHashTable(
        initializer=initializer,
        default_value=3,
    )


def test_make_chatspace_training_dataset(lookup_table: tf.lookup.StaticHashTable):
    test_data = [
        "참 많이 힘들어요",
        "정든 그댈 떠나가기가",
        "단 하루도 참아내지 못한 채",
        "이렇게 난 슬피 울고 있죠",
        "세월은 흘러 사랑도 가고",
        "아팠던 기억도 멀어지는데",
        "사랑은 왜 하늘 아래",
        "사랑은 왜 하늘 아래",
        "내 삶의 끝에서 헤메이는지",
        "기억해 줘 너의 가슴에",
        "아름다운 사랑이 있었다는 걸",
    ]
    space_skip_prob = 0.3

    dataset = tf.data.Dataset.from_tensor_slices(tf.constant(test_data)).map(
        make_chatspace_training_dataset(lookup_table, space_skip_prob)
    )

    for idx, data in enumerate(dataset):
        inputs, labels = data
        tf.debugging.assert_equal(tf.shape(inputs), tf.shape(labels))
        tf.debugging.assert_equal(inputs[0], BOS_TOKEN)
        tf.debugging.assert_equal(inputs[-1], EOS_TOKEN)
