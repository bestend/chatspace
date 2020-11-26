import tensorflow as tf

from chatspace.model import ChatspaceModel


def test_chatspace_model_shape():
    model = ChatspaceModel(2000, 128, 128, 3, "relu", 0.3)
    batch_size = 10
    sequence_length = 128

    result = model(tf.random.uniform(shape=(batch_size, sequence_length), maxval=2000, dtype=tf.int32))

    tf.debugging.assert_equal(tf.shape(result), (batch_size, sequence_length, 2))
