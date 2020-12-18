import tensorflow as tf
import tensorflow_addons as tfa


def sparse_categorical_crossentropy_with_ignore(y_true, y_pred, from_logits=False, axis=-1, ignore_id=-1):
    positions = tf.where(y_true != ignore_id)

    y_true = tf.gather_nd(y_true, positions)
    y_pred = tf.gather_nd(y_pred, positions)

    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=from_logits, axis=axis)


def sparse_categorical_accuracy_with_ignore(y_true, y_pred, ignore_id=-1):
    positions = tf.where(y_true != ignore_id)

    y_true = tf.gather_nd(y_true, positions)
    y_pred = tf.gather_nd(y_pred, positions)

    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


class SparseCategoricalCrossentropyWithIgnore(tfa.utils.keras_utils.LossFunctionWrapper):
    """
    ``ignore_id``를 고려하는 SparseCategoricalCrossentropy입니다.

    :param from_logits: SparseCategoricalCrossentropy의 from_logits
    :param reduction: SparseCategoricalCrossentropy의 reduction
    :param ignore_id: 계산에 반영하지 않을 Label 값
    :param name: Graph Node의 이름
    """

    def __init__(
        self,
        from_logits=False,
        reduction=tf.keras.losses.Reduction.AUTO,
        ignore_id=-1,
        name="sparse_categorical_crossentropy_with_ignore",
    ):
        super(SparseCategoricalCrossentropyWithIgnore, self).__init__(
            sparse_categorical_crossentropy_with_ignore,
            name=name,
            reduction=reduction,
            ignore_id=ignore_id,
            from_logits=from_logits,
        )
