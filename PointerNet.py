import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from Encoder import Encoder
from Decoder import Attention, Decoder

class PointerNet(nn.Module):
    """
    Pointer-Net
    """

    def __init__(
        self, embedding_dim,
        hidden_dim,
        lstm_layers,
        dropout,
        bidir = False
    ):
        """
        Initiate Pointer-Net

        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(PointerNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.bidir = bidir
        self.embedding = nn.Linear(2, embedding_dim)
        self.encoder = Encoder(
            embedding_dim, hidden_dim,
            lstm_layers, dropout,
            bidir
        )
        self.decoder = Decoder(embedding_dim, hidden_dim)
        self.decoder_input0 = Parameter(torch.FloatTensor(embedding_dim), requires_grad = False)

        # Initialize decoder_input0
        nn.init.uniform_(self.decoder_input0, -1, 1)

    def forward(self, inputs):
        """
        PointerNet - Forward-pass

        :param Tensor inputs: Input sequence
        :return: Pointers probabilities and indices
        """

        batch_size = inputs.size(0)
        input_length = inputs.size(1)

        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)

        inputs = inputs.view(batch_size * input_length, -1)
        embedded_inputs = self.embedding(inputs).view(batch_size, input_length, -1)

        encoder_hidden0 = self.encoder.init_hidden(embedded_inputs)
        encoder_outputs, encoder_hidden = self.encoder(embedded_inputs, encoder_hidden0)

        if self.bidir:
            decoder_hidden0 = (encoder_hidden[0][-1].squeeze(0), encoder_hidden[1][-1].squeeze(0))
        else:
            decoder_hidden0 = (encoder_hidden[0][-1], encoder_hidden[1][-1])
        
        (outputs, pointers), decoder_hidden = self.decoder(
            embedded_inputs,
            decoder_input0,
            decoder_hidden0,
            encoder_outputs
        )

        return  outputs, pointers