import tensorflow as tf


class ChatspaceModel(tf.keras.Model):
    """
    Chatspace의 모델입니다.
    두 번의 Conv1D를 통하여 각 글자 별로 띄어쓰기 여부를 계산합니다.

    :param vocab_size: 등장 단어 수
    :param embedding_size: 임베딩 차원의 크기
    :param filter_size: 필터의 크기
    :param kernel_size: 커널의 크기
    :param conv_activation: Convolution Layer의 Activation
    :param dropout_prob: Dropout 확률
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        filter_size: int,
        kernel_size: int,
        conv_activation: str,
        dropout_prob: float,
    ):
        super(ChatspaceModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.conv_activation = conv_activation
        self.dropout_prob = dropout_prob

        self.embedding_layer = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)
        self.conv_layer1 = tf.keras.layers.Conv1D(
            filter_size,
            kernel_size,
            padding="same",
            activation=self.conv_activation,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
        )
        self.dropout_layer = tf.keras.layers.Dropout(rate=self.dropout_prob)
        self.conv_layer2 = tf.keras.layers.Conv1D(
            2,
            3,
            padding="same",
            activation="softmax",
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
        )

    def call(self, input_tensor, training=True):
        # outputs shape: [batch_size, sequence_length, embedding_size]
        outputs = self.embedding_layer(input_tensor)

        # outputs shape: [batch_size, sequence_length, filter_size]
        outputs = self.conv_layer1(outputs)
        outputs = self.dropout_layer(outputs)

        # outputs shape: [batch_size, sequence_length, 2]
        outputs = self.conv_layer2(outputs)

        return outputs
