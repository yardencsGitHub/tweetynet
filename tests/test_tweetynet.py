import torch

import tweetynet


class TestConv2dTF:
    def test_Conv2dTF_same_padding(self):
        input_size = (1, 513, 88)
        batch_size = tuple((10,) + input_size[:])
        batch_of_spects = torch.rand(batch_size)
        n_filters = 64
        kernel_size = (5, 5)

        conv2dtf = tweetynet.network.Conv2dTF(in_channels=input_size[0],
                                              out_channels=n_filters,
                                              kernel_size=kernel_size,
                                              padding_mode="same")

        out = conv2dtf(batch_of_spects)
        assert tuple(out.shape)[2:] == batch_size[2:]
