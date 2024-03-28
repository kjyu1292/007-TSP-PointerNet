import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Attention model for Pointer-Net
    """

    def __init__(self, q_dim, k_dim, hidden_dim):
        """
        Initiate Attention

        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(Attention, self).__init__()

        self.q_dim = q_dim
        self.k_dim = k_dim
        self.hidden_dim = hidden_dim

        self.project_queries = nn.Linear(q_dim, hidden_dim, bias = False)
        self.project_keys = nn.Linear(k_dim, hidden_dim, bias = False)
        self.V = nn.Linear(hidden_dim, 1, bias = False)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad = False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = 0)

    def forward(self, queries, keys, mask):
        """
        Attention - Forward-pass

        :param Tensor queries: hidden state of decoder from step function in Decoder object (batch, hidden_dim)
        :param Tensor keys: encoder outputs, or context (seq_len, batch, 2*hidden_dim)
        :param ByteTensor mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """
        self.inf = self._inf.unsqueeze(1).expand(mask.size())

        # (batch, hidden_dim)
        projected_q = self.project_queries(queries)
        # (seq_len, batch, hidden_dim)
        projected_k = self.project_keys(keys)
        
        # (batch, seq_len)
        attention_pointer = self.V(
            self.tanh(projected_q + projected_k)
        ).squeeze(-1).permute(-1, 0)
        # if len(attention_pointer[mask]) > 0:
        #     attention_pointer[mask] = self.inf[mask]
        alpha = self.softmax(attention_pointer)

        # (256, 512) = bmm (256, 512, 5) (256, 5, 1)
        hidden_state = torch.bmm(
            projected_k.permute(1, 2, 0), alpha.unsqueeze(2)
        ).squeeze(2)

        return hidden_state, alpha

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


class Decoder(nn.Module):
    """
    Decoder model for Pointer-Net
    """

    def __init__(
        self, embedding_dim,
        hidden_dim
    ):
        """
        Initiate Decoder

        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        """

        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.input_to_hidden = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att = Attention(q_dim = hidden_dim, k_dim = 2*hidden_dim, hidden_dim = hidden_dim)

        # Used for propagating .cuda() command
        self.mask = Parameter(torch.ones(1), requires_grad = False)
        self.runner = Parameter(torch.zeros(1), requires_grad = False)
        self.index0 = Parameter(torch.zeros(1), requires_grad = False)

    def forward(
        self, embedded_inputs,
        decoder_input,
        hidden,
        context
    ):
        """
        Decoder - Forward-pass

        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor decoder_input: First decoder's input
        :param Tensor hidden: First decoder's hidden states
        :param Tensor context: Encoder's outputs
        :return: (Output probabilities, Pointers indices), last hidden state
        """

        batch_size = embedded_inputs.size(0)
        input_length = embedded_inputs.size(1)

        # (batch, seq_len)
        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)

        # Generating arang(input_length), broadcasted across batch_size
        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        # Initilize the first iteration of index, which is 0 since we want to start at that index
        index0 = self.index0.repeat(batch_size)

        outputs = []
        pointers = []

        def step(x, hidden):
            """
            Recurrence step function

            :param Tensor x: Input at time t
            :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1
            :return: Hidden states at time t (h, c), Attention probabilities (Alpha)
            """

            # Regular LSTM
            h, c = hidden

            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)
            input, forget, cell, out = gates.chunk(4, -1)

            input = F.sigmoid(input)
            forget = F.sigmoid(forget)
            cell = F.tanh(cell)
            out = F.sigmoid(out)

            c_t = (forget * c) + (input * cell)
            h_t = out * F.tanh(c_t)

            # Attention section
            hidden_t, output = self.att(h_t, context, torch.eq(mask, 0))
            hidden_t = F.tanh(self.hidden_out(torch.cat((hidden_t, h_t), 1)))

            return hidden_t, c_t, output

        # Recurrence loop
        for _ in range(input_length + 2):
            h_t, c_t, outs = step(decoder_input, hidden)
            hidden = (h_t, c_t)

            # Masking selected inputs
            masked_outs = outs * mask

            # Get maximum probabilities and indices
            max_probs, indices = masked_outs.max(1)
            
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()

            # Update mask to ignore seen indices
            mask  = mask * (1 - one_hot_pointers)

            # Get embedded inputs by max indices
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.embedding_dim).byte()
            decoder_input = embedded_inputs[embedding_mask.data > 0].view(batch_size, self.embedding_dim)

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 2, 0)
        pointers = torch.cat(pointers, 1)

        return (outputs, pointers), hidden