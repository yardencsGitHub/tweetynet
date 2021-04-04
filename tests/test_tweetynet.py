import unittest

import torch

import tweetynet


class TestConv2dTF(unittest.TestCase):
    def test_Conv2dTF_same_padding(self):
        input_size = (1, 513, 88)
        batch_size = tuple((10,) + input_size[:])
        batch_of_spects = torch.rand(batch_size)
        n_filters = 64
        kernel_size = (5, 5)

        conv2dtf = tweetynet.network.Conv2dTF(in_channels=input_size[0],
                                              out_channels=n_filters,
                                              kernel_size=kernel_size,
                                              padding="SAME")

        out = conv2dtf(batch_of_spects)
        assert tuple(out.shape)[2:] == batch_size[2:]


class TestHiddenSize(unittest.TestCase):
    def test_default(self):
        NUM_CLASSES = 11

        net = tweetynet.network.TweetyNet(num_classes=NUM_CLASSES)

        self.assertTrue(
            net.rnn.input_size == net.rnn_input_size
        )

        # test that we default to hidden size equal to input size
        self.assertTrue(
            net.rnn.hidden_size == net.rnn_input_size
        )

    def test_not_default(self):
        NUM_CLASSES = 11
        HIDDEN_SIZE = 2048

        net = tweetynet.network.TweetyNet(num_classes=NUM_CLASSES,
                                          hidden_size=HIDDEN_SIZE)

        self.assertTrue(
            net.rnn.input_size == net.rnn_input_size
        )

        # test that we default to hidden size equal to input size
        self.assertTrue(
            net.rnn.hidden_size == net.hidden_size == HIDDEN_SIZE
        )
