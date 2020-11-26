"""Chatspace 모델의 속도를 측정하는 코드입니다."""

import argparse
import time

import tensorflow as tf

from chatspace.model import ChatspaceModel

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--hidden-size", type=int, default=64, help="hidden layer 의 크기")
parser.add_argument("--vocab-size", type=int, default=2000, help="vocab 의 크기")
parser.add_argument("--embedding-size", type=int, default=128, help="embedding 차원의 크기")
parser.add_argument("--filter-size", type=int, default=128, help="convolution layer 에서 filter 의 크기")
parser.add_argument("--kernel-size", type=int, default=3, help="convolution layer 에서 kernel 의 크기")
parser.add_argument("--conv-activation", type=str, default="relu", help="convolution layer 의 activation 함수")
parser.add_argument("--dense-activation", type=str, default="relu", help="dense layer 의 activation 함수")
parser.add_argument("--dropout-prob", type=float, default=0.3, help="dropout 확률")
parser.add_argument("--learning-rate", type=float, default=0.002, help="learning rate")
# fmt: on


def main(args):
    model = ChatspaceModel(
        vocab_size=args.vocab_size,
        embedding_size=args.embedding_size,
        filter_size=args.filter_size,
        kernel_size=args.kernel_size,
        conv_activation=args.conv_activation,
        dropout_prob=args.dropout_prob,
    )
    test_data = [
        tf.random.uniform(shape=(1, 128), minval=0, maxval=args.vocab_size, dtype=tf.int32) for _ in range(9000)
    ]

    print("[+] Tracing...")
    model(test_data[0])

    print("[+] Measuring...")
    start_time = time.time()

    # spacing model inference
    for data in test_data:
        model(data)

    end_time = time.time()

    print(f"[+] Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
