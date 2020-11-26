import pytest

from chatspace.inference import Chatspace


def test_single_space():
    spacer = Chatspace()

    text = "따뜻한봄날이되면그때는편안히만날수있으면좋겠어요."
    result = "따뜻한 봄날이 되면 그때는 편안히 만날 수 있으면 좋겠어요."

    assert spacer.space(text) == result


@pytest.mark.parametrize(
    "texts, results",
    [pytest.param(["여러문장이", "들어있는리스트입니다"], ["여러 문장이", "들어있는 리스트입니다"]), pytest.param([], [])],
    ids=["Normal Case", "Empty List"],
)
def test_batch_space(texts, results):
    spacer = Chatspace()

    assert spacer.space(texts, batch_size=max(len(texts), 1)) == results
