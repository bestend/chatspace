"""주어진 Checkpoint를 로딩하여 SavedModel로 바꾸어 저장하는 코드입니다."""
import argparse

import tensorflow as tf

from chatspace.model import ChatspaceModel

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint-path", type=str, default="./model", help="model의 checkpoint들이 저장되어 있는 디렉토리의 경로")
parser.add_argument("--output-path", type=str, default="./saved_model", help="SavedModel을 저장할 경로")
parser.add_argument("--hidden-size", type=int, default=64, help="hidden layer 의 크기")
parser.add_argument("--vocab-size", type=int, default=2000, help="vocab 의 크기")
parser.add_argument("--embedding-size", type=int, default=128, help="embedding 차원의 크기")
parser.add_argument("--filter-size", type=int, default=128, help="convolution layer 에서 filter 의 크기")
parser.add_argument("--kernel-size", type=int, default=3, help="convolution layer 에서 kernel 의 크기")
parser.add_argument("--conv-activation", type=str, default="relu", help="convolution layer 의 activation 함수")
parser.add_argument("--dense-activation", type=str, default="relu", help="dense layer 의 activation 함수")
# fmt: on


def main(args):
    print("[+] Initializing...")
    model = ChatspaceModel(
        vocab_size=args.vocab_size,
        embedding_size=args.embedding_size,
        filter_size=args.filter_size,
        kernel_size=args.kernel_size,
        conv_activation=args.conv_activation,
        dropout_prob=0.0,
    )

    print("[+] Tracing...")
    model(tf.keras.Input(shape=[None, None], dtype=tf.int32))

    print("[+] Loading previous checkpoint...")
    latest = tf.train.latest_checkpoint(args.checkpoint_path)
    model.load_weights(latest)

    print("[+] Saving...")
    tf.saved_model.save(model, args.output_path)

    print("[+] Done.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
