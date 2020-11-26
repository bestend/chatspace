import tensorflow as tf

from chatspace.measure import sparse_categorical_accuracy_with_ignore, sparse_categorical_crossentropy_with_ignore


def test_sparse_categorical_crossentropy_with_ignore():
    y_true = tf.constant([[0, 0, 1, 0, 0, -1, -1, -1, -1]], dtype=tf.int32)
    y_pred = tf.constant(
        [[[0.6, 0.4], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.2, 0.8], [0.2, 0.8], [0.2, 0.8]]],
        dtype=tf.float32,
    )

    result = sparse_categorical_crossentropy_with_ignore(y_true, y_pred, ignore_id=-1)
    assert tf.shape(result) == (5,)


def test_sparse_categorical_accuracy_with_ignore():
    y_true = tf.constant([[0, 0, 1, 0, 0, -1, -1, -1, -1]], dtype=tf.int32)
    y_pred = tf.constant(
        [[[0.6, 0.4], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.2, 0.8], [0.2, 0.8], [0.2, 0.8]]],
        dtype=tf.float32,
    )

    result = sparse_categorical_accuracy_with_ignore(y_true, y_pred, ignore_id=-1)
    assert tf.shape(result) == (5,)
