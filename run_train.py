import argparse
import glob

import tensorflow as tf

from chatspace.config import TrainingConfig
from chatspace.dataset import make_chatspace_training_dataset
from chatspace.measure import SparseCategoricalCrossentropyWithIgnore, sparse_categorical_accuracy_with_ignore
from chatspace.model import ChatspaceModel

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--vocab-path", type=str, required=True, help="vocab 이 위치한 경로")
parser.add_argument("--training-data-path", type=str, required=True, help="training data 가 위치한 경로")
parser.add_argument("--model-checkpoint-path", type=str, default="./model/checkpoint-{epoch}.ckpt", help="model의 checkpoint 를 저장할 경로")
parser.add_argument("--model-logs-path", type=str, default="./logs", help="model의 logs 를 저장할 경로")
parser.add_argument("--epochs", type=int, default=50, help="training epoch 횟수")
parser.add_argument("--learning-rate", type=float, default=0.002, help="learning rate")
parser.add_argument("--dropout-prob", type=float, default=0.3, help="dropout 확률")
parser.add_argument("--train-batch-size", type=int, default=8192, help="train batch 의 크기")
parser.add_argument("--dev-data-size", type=int, default=100000, help="dev data 개수")
parser.add_argument("--dev-batch-size", type=int, default=8192, help="dev batch 의 크기")
parser.add_argument("--vocab-size", type=int, default=2000, help="vocab 의 크기")
parser.add_argument("--embedding-size", type=int, default=128, help="embedding 차원의 크기")
parser.add_argument("--filter-size", type=int, default=128, help="convolution layer 에서 filter 의 크기")
parser.add_argument("--kernel-size", type=int, default=3, help="convolution layer 에서 kernel 의 크기")
parser.add_argument("--conv-activation", type=str, default="relu", help="convolution layer 의 activation 함수")
parser.add_argument("--shuffle-buffer-size", type=int, default=10000000, help="dataset shuffle buffer 크기")
parser.add_argument("--space-skip-prob", type=float, default=0.7, help="공백을 생략할 확률")
# fmt: on


def main(args: argparse.Namespace):
    config = TrainingConfig(
        vocab_path=args.vocab_path,
        training_data_path=args.training_data_path,
        model_checkpoint_path=args.model_checkpoint_path,
        model_logs_path=args.model_logs_path,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        dropout_prob=args.dropout_prob,
        train_batch_size=args.train_batch_size,
        dev_batch_size=args.dev_batch_size,
        dev_data_size=args.dev_data_size,
        vocab_size=args.vocab_size,
        embedding_size=args.embedding_size,
        filter_size=args.filter_size,
        kernel_size=args.kernel_size,
        conv_activation=args.conv_activation,
        shuffle_buffer_size=args.shuffle_buffer_size,
        space_skip_prob=args.space_skip_prob,
    )

    with open(config.vocab_path, "r") as f:
        content = f.read()
        keys = ["<PAD>", "<s>", "</s>", "<UNK>"] + list(content)
        values = list(range(len(keys)))
        initializer = tf.lookup.KeyValueTensorInitializer(
            keys[: config.vocab_size], values[: config.vocab_size], key_dtype=tf.string, value_dtype=tf.int32
        )
        lookup_table = tf.lookup.StaticHashTable(
            initializer=initializer,
            default_value=3,
        )

    files = glob.glob(config.training_data_path)
    dataset = tf.data.TextLineDataset(files)

    dataset = (
        dataset.shuffle(config.shuffle_buffer_size)
        .map(
            make_chatspace_training_dataset(
                lookup_table=lookup_table,
                space_skip_prob=config.space_skip_prob,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .cache()
        .prefetch(
            tf.data.experimental.AUTOTUNE,
        )
    )

    # Input은 0, Label은 -1로 Padding함
    dev_dataset = dataset.take(config.dev_data_size).padded_batch(config.dev_batch_size, padding_values=(0, -1))
    train_dataset = dataset.skip(config.dev_data_size).padded_batch(config.train_batch_size, padding_values=(0, -1))

    model = ChatspaceModel(
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        embedding_size=config.embedding_size,
        filter_size=config.filter_size,
        kernel_size=config.kernel_size,
        conv_activation=config.conv_activation,
        dense_activation=config.dense_activation,
        dropout_prob=config.dropout_prob,
    )

    optimizer = tf.optimizers.Adam(learning_rate=config.learning_rate)
    loss = SparseCategoricalCrossentropyWithIgnore(ignore_id=-1)
    metrics = sparse_categorical_accuracy_with_ignore

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[metrics],
    )

    model.fit(
        train_dataset,
        epochs=config.epochs,
        validation_data=dev_dataset,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=args.model_checkpoint_path,
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
            ),
            tf.keras.callbacks.TensorBoard(log_dir=args.model_logs_path),
            tf.keras.callbacks.ReduceLROnPlateau(patience=2, verbose=1),
        ],
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
