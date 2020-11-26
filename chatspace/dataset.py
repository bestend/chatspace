import tensorflow as tf


def make_chatspace_training_dataset(
    lookup_table: tf.lookup.StaticHashTable,
    space_skip_prob: float,
):
    """
    Chatspace 모델에 넣을 Input을 만드는 함수를 반환합니다.
    정상 문장을 입력으로 받아 랜덤하게 띄어쓰기를 지웁니다.

    :param lookup_table: Vocab 정보로 초기화된 ``tf.lookup.StaticHashTable``
    :param space_skip_prob: 띄어쓰기를 지울 확률
    """

    @tf.function
    def _mapping_function(
        sentence: tf.Tensor,
    ):
        sentence = tf.strings.unicode_split(sentence, "UTF-8")
        sentence = tf.strings.regex_replace(sentence, " +", " ")
        sentence_length = tf.shape(sentence)[0]

        def cond(index, inputs, labels):
            return index < sentence_length

        def body(index, inputs, labels):
            inputs = tf.concat([inputs, [sentence[index]]], axis=0)

            index, labels = tf.cond(
                index != sentence_length - 1 and sentence[index + 1] == " ",
                lambda: tf.cond(
                    tf.random.uniform([], minval=0, maxval=1) <= space_skip_prob,
                    lambda: (index + 1, tf.concat([labels, [1]], axis=0)),
                    lambda: (index, tf.concat([labels, [0]], axis=0)),
                ),
                lambda: (index, tf.concat([labels, [0]], axis=0)),
            )

            index += 1
            return index, inputs, labels

        _, inputs, labels = tf.while_loop(
            cond,
            body,
            (
                tf.constant(0),
                tf.constant([], dtype=tf.string),
                tf.constant([], dtype=tf.int32),
            ),
            shape_invariants=(
                tf.TensorShape([]),
                tf.TensorShape([None]),
                tf.TensorShape([None]),
            ),
        )

        # 문장의 앞뒤에 BOS, EOS 토큰을 붙입니다.
        inputs = tf.concat([["<s>"], inputs, ["</s>"]], axis=0)
        labels = tf.concat([[0], labels, [0]], axis=0)
        inputs = lookup_table.lookup(inputs)

        return inputs, labels

    return _mapping_function
