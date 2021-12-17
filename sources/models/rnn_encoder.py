
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNEncoder(nn.Module):

    def __init__(self, hidden_size, rnn_type='lstm'):
        super(RNNEncoder, self).__init__()
        assert rnn_type in ['lstm', 'gru']
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type

        self.rnn = nn.LSTM(hidden_size, hidden_size, bidirectional=True) if rnn_type == 'lstm' \
            else nn.GRU(hidden_size, hidden_size, bidirectional=True)

    def forward(self, embedded, seq_lens):
        """
        Layer definition.
        Args:
            embedded (torch.Tensor): embedded input, [B, T, H]
            seq_lens (list): lengths of each sequences in the batch

        Returns:
            (torch.Tensor, torch.Tensor):
                - hidden state of each time step, [T, B, H]
                - hidden state of the last time step, [2, B, H]
        """
        assert embedded.size()[-1] == self.hidden_size, \
            f'Except dimension of embedded input {self.hidden_size}, got {embedded.size()[-1]}'
        embedded = embedded.transpose(0, 1)     # [T, B, H]
        packed = pack_padded_sequence(embedded, seq_lens, enforce_sorted=False)
        # hidden: [2, B, H]
        outputs, hidden = self.rnn(packed)
        if self.rnn_type == 'lstm':
            hidden, _ = hidden
        outputs, _ = pad_packed_sequence(outputs)  # [T, B, 2*H]
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]   # [T, B, H]

        hidden = hidden[0] + hidden[1]      # [B, H]

        return outputs, hidden
