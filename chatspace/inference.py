import json
import re
from typing import List, Union

import tensorflow as tf

from .resources import CONFIG_PATH, MODEL_PATH, VOCAB_PATH


class Chatspace:
    """
    대화체에 잘 맞는 띄어쓰기 모델입니다.

    :param model_path: SavedModel이 저장되어 있는 경로
    :param config_path: Config가 저장되어 있는 경로
    :param vocab_path: Vocab이 저장되어 있는 경로
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        config_path: str = CONFIG_PATH,
        vocab_path: str = VOCAB_PATH,
    ):
        self.loaded = tf.saved_model.load(model_path)
        self.model = self.loaded.signatures["serving_default"]

        with open(config_path, "r") as f:
            self.config = json.load(f)

        with open(vocab_path, "r") as f:
            content = f.read()
            keys = ["<PAD>", "<s>", "</s>", "<UNK>"] + list(content)
            values = list(range(len(keys)))
            initializer = tf.lookup.KeyValueTensorInitializer(
                keys[: self.config["vocab_size"]],
                values[: self.config["vocab_size"]],
                key_dtype=tf.string,
                value_dtype=tf.int32,
            )
            self.lookup_table = tf.lookup.StaticHashTable(initializer=initializer, default_value=3)

    def space(self, texts: Union[List[str], str], batch_size: int = 1) -> Union[List[str], str]:
        """
        주어진 문장 혹은 문장들의 띄어쓰기를 보정합니다.

        :param texts: 띄어쓰기를 하고자 하는 문장 또는 문장들
        :param batch_size: Inference를 수행할 배치의 크기
        :return: 띄어쓰기가 완료된 문장 또는 문장들
        """
        is_single_inference = isinstance(texts, str)
        texts = [texts] if is_single_inference else texts
        dataset = self.make_chatspace_inputs(texts, batch_size=batch_size)

        outputs = []
        for data in dataset:
            pred = self.model(data)["output_1"]
            space_preds = tf.math.argmax(pred, axis=-1)
            outputs.extend(space_preds)

        result = self.generate_text(texts, outputs)

        return result[0] if is_single_inference else result

    def make_chatspace_inputs(self, texts: List[str], batch_size: int = 1):
        """
        주어진 String의 List를 ChatspaceModel에 넣을 수 있는 Dataset으로 변환합니다.

        :param texts: Batch로 묶여 있는 문장들
        :return: Batch로 묶여 있는 tf.Tensor
        """

        @tf.function
        def _mapping_function(x: tf.Tensor):
            x = tf.strings.unicode_split(x, "UTF-8")
            return self.lookup_table.lookup(tf.concat([["<s>"], x, ["</s>"]], axis=0))

        return (
            tf.data.Dataset.from_tensor_slices(tf.constant(texts, dtype=tf.string))
            .map(_mapping_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .padded_batch(batch_size, padded_shapes=[None], padding_values=0)
        )

    def generate_text(self, texts: List[str], space_pred: tf.Tensor) -> str:
        """
        추론된 결과를 바탕으로 실제 문장에 띄어쓰기를 반영합니다.

        :param texts: 띄어쓰기가 옳바르지 않은 원본 문장
        :param space_pred: ChatspaceModel 에서 나온 결과를 Argmax한 Tensor (``[batch, seq_len]``)
        :return: 띄어쓰기가 반영된 문장
        """
        result = []
        for text, pred in zip(texts, space_pred):
            generated_sentence = [
                # BOS 때문에 text[i] 와 pred[i + 1] 이 대응됩니다.
                text[i] + (" " if pred[i + 1] == 1 else "")
                for i in range(len(text))
            ]
            joined_chars = "".join(generated_sentence)
            result.append(re.sub(r"\s+", " ", joined_chars).strip())

        return result
