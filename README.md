# Chatspace

[![CircleCI](https://circleci.com/gh/pingpong-ai/chatspace.svg?style=svg)](https://circleci.com/gh/pingpong-ai/chatspace)

Chatspace는 [스캐터랩 핑퐁팀](https://scatterlab.co.kr/pingpong)에서 만든 대화체에 잘 맞는 띄어쓰기 모델입니다.

## 사용 방법

### 한 문장 추론

```python
import chatspace

spacer = chatspace.Chatspace()
spacer.space("따뜻한봄날이되면그때는편안히만날수있으면좋겠어요.")
# 따뜻한 봄날이 되면 그때는 편안히 만날 수 있으면 좋겠어요.
```

### 여러 문장 추론 (Batch)

```python
import chatspace

spacer = chatspace.Chatspace()
spacer.space(["여러문장이", "들어있는리스트입니다"], batch_size=BATCH_SIZE)
# ["여러 문장이", "들어있는 리스트입니다"]
```

## Benchmark

아래는 iMac (Retina 5K, 27-inch, 2017), Intel Core i7 (4.2 GHz), 32GB RAM에서
Batch Size를 1로 맞추었을 때 9,000개의 샘플을 실행한 결과입니다.

```shell
$ python -m tools.benchmark
[+] Tracing...
[+] Measuring...
[+] Elapsed time: 00:00:11
```

## Installation

Chatspace는 PyPI와 Github에서 각각 설치할 수 있습니다.

### From PyPI

```shell
$ pip install chatspace
```

### From Github

```shell
$ pip install git+https://github.com/pingpong-ai/chatspace#egg=chatspace
```
