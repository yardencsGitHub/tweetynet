import torch
import pytest

import tweetynet


@pytest.mark.parametrize(
    'padding',
    [
        'SAME',
        'same'
    ]
)
def test_Conv2dTF_same_padding(padding):
    input_size = (1, 513, 88)
    batch_size = tuple((10,) + input_size[:])
    batch_of_spects = torch.rand(batch_size)
    n_filters = 64
    kernel_size = (5, 5)

    conv2dtf = tweetynet.network.Conv2dTF(in_channels=input_size[0],
                                          out_channels=n_filters,
                                          kernel_size=kernel_size,
                                          padding=padding)

    out = conv2dtf(batch_of_spects)
    assert tuple(out.shape)[2:] == batch_size[2:]


@pytest.mark.parametrize(
    'hidden_size',
    [
        None,
        2048
    ]
)
def test_hidden_size(hidden_size):
    NUM_CLASSES = 11

    net = tweetynet.network.TweetyNet(num_classes=NUM_CLASSES,
                                      hidden_size=hidden_size)

    assert net.rnn.input_size == net.rnn_input_size
    if hidden_size is not None:
        assert net.rnn.hidden_size == net.hidden_size == hidden_size

