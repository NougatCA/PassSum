import torch
import torch.nn as nn

from .rnn_encoder import RNNEncoder


class PathEncoding(nn.Module):

    def __init__(self, d_model, code_embedding: nn.Embedding, node_vocab_size: int):
        super(PathEncoding, self).__init__()
        self.d_model = d_model
        self.code_embedding = code_embedding
        self.node_embedding = nn.Embedding(node_vocab_size, d_model)

        self.id_encoder = RNNEncoder(hidden_size=self.d_model,
                                     rnn_type='gru')
        self.node_encoder = RNNEncoder(hidden_size=self.d_model,
                                       rnn_type='gru')
        self.combine_linear = nn.Linear(2 * self.d_model, d_model)

    def forward(self, id_inputs, id_seq_lens, node_inputs, node_seq_lens, path_seq_lens):

        id_embedded = self.code_embedding(id_inputs)        # [B, T]
        node_embedded = self.node_embedding(node_inputs)    # [B, T]

        # [path_B, H]
        _, id_hidden = self.id_encoder(embedded=id_embedded, seq_lens=id_seq_lens)
        _, node_hidden = self.node_encoder(embedded=node_embedded, seq_lens=node_seq_lens)

        combined = torch.cat([node_hidden, id_hidden], dim=-1)  # [path_B, 2*H]
        combined = self.combine_linear(combined)                # [path_B, H]

        outputs = PathEncoding.unpack_batch(combined, path_seq_lens)    # [B, T, H]

        return outputs

    @staticmethod
    def unpack_batch(x, seq_lens):

        batch_size = len(seq_lens)
        max_len = max(seq_lens)
        hidden_size = x.size(-1)

        outputs = torch.zeros([batch_size, max_len, hidden_size], dtype=torch.float, device=x.device)     # [B, T, H]

        start_index = 0
        for batch_index, batch_len in enumerate(seq_lens):
            outputs[batch_index, :batch_len, :] = x[start_index: start_index + batch_len]
            start_index += batch_len

        return outputs


