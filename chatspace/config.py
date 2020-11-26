from typing import NamedTuple


class TrainingConfig(NamedTuple):
    #: vocab file 경로
    vocab_path: str
    #: training data 경로
    training_data_path: str
    #: model 의 checkpoint 를 저장할 경로
    model_checkpoint_path: str
    #: model 의 logs 를 저장할 경로
    model_logs_path: str

    #: epoch 횟수
    epochs: int
    #: learning rate
    learning_rate: float
    #: dropout 확룰
    dropout_prob: float
    #: train batch 크기
    train_batch_size: int
    #: dev batch_size
    dev_batch_size: int
    #: dev data 의 개수
    dev_data_size: int

    #: vocab 크기
    vocab_size: int
    #: 임베딩 차원의 크기
    embedding_size: int
    #: filter 의 크기
    filter_size: int
    #: kernel 의 크기
    kernel_size: int
    #: conv layer 의 activation 함수
    conv_activation: str
    #: dataset shuffle buffer 크기
    shuffle_buffer_size: int
    #: 공백을 생략할 확률
    space_skip_prob: float
