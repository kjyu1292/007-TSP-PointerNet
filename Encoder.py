import torch
import torch.nn as nn
from torch.nn import Parameter

class Encoder(nn.Module):
    """
    Encoder class for Pointer-Net
    """

    def __init__(
        self, embedding_dim,
        hidden_dim,
        n_layers,
        dropout,
        bidir
    ):
        """
        Initiate Encoder

        :param Tensor embedding_dim: Number of embbeding channels
        :param int hidden_dim: Number of hidden units for the LSTM
        :param int n_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(Encoder, self).__init__()
        self.n_layers = n_layers*2 if bidir else n_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size = embedding_dim,
            hidden_size = hidden_dim,
            num_layers = n_layers,
            bias = False,
            batch_first = False,
            dropout = dropout,
            bidirectional = bidir
        )

        # Used for propagating .cuda() command
        self.h0 = Parameter(torch.zeros(1), requires_grad = False)
        self.c0 = Parameter(torch.zeros(1), requires_grad = False)

    def forward(self, embedded_inputs, hidden):
        """
        Encoder - Forward-pass

        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor hidden: Initiated hidden units for the LSTMs (h, c)
        :return: LSTMs outputs and hidden units (h, c)
        """
        embedded_inputs = embedded_inputs.permute(1, 0, 2)

        outputs, hidden = self.lstm(embedded_inputs, hidden)

        return outputs.permute(1, 0, 2), hidden

    def init_hidden(self, embedded_inputs):
        """
        Initiate hidden units

        :param Tensor embedded_inputs: The embedded input of Pointer-NEt
        :return: Initiated hidden units for the LSTMs (h, c)
        """

        batch_size = embedded_inputs.shape[0]

        # Reshaping (Expanding)
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers, batch_size, self.hidden_dim)
        c0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers, batch_size, self.hidden_dim)

        return h0, c0