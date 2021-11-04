import torch
import torch.nn as nn
from .def_attn_components import (
    Debedder,
    get_clones,
    EncoderLayer,
    DebedderNeuronGroup,
    DebedderNeuronGroup_index,
)
from .def_decoder import Decoder

class DecoderTransformer(nn.Module):
    def __init__(self, config):
        super(DecoderTransformer, self).__init__()

        # def __init__(self, input_dim, embed_dim, N, heads, max_seq_len, dropout, d_ff):
        self.N = config["model::N_attention_blocks"]
        self.input_dim = config["model::i_dim"]
        self.embed_dim = config["model::dim_attention_embedding"]
        self.normalize = config["model::normalize"]
        self.heads = config["model::N_attention_heads"]
        self.dropout = config["model::dropout"]
        self.d_ff = config["model::attention_hidden_dim"]
        self.latent_dim = config["model::latent_dim"]
        self.device = config["device"]

        # get token encodings
        if config.get("model::encoding", "weight") == "weight":
            # encode each weight separately
            self.max_seq_len = self.input_dim
            self.token_debeddings = Debedder(self.input_dim, self.embed_dim)
        elif config.get("model::encoding", "weight") == "neuron":
            if config.get("model::index_dict", None) is None:
                # encode weights of one neuron together
                self.max_seq_len = 9
                self.token_debeddings = DebedderNeuronGroup(self.embed_dim)
            else:
                index_dict = config.get("model::index_dict", None)
                self.token_debeddings = DebedderNeuronGroup_index(
                    index_dict, self.embed_dim
                )
                self.max_seq_len = self.token_debeddings.__len__()

        # get learned position embedding
        self.position_embeddings = nn.Embedding(self.max_seq_len, self.embed_dim)

        self.layers = get_clones(
            EncoderLayer(
                d_model=self.embed_dim,
                heads=self.heads,
                normalize=self.normalize,
                dropout=self.dropout,
                d_ff=self.d_ff,
            ),
            self.N,
        )

        bottleneck = config.get("model::bottleneck", "linear")
        if bottleneck == "linear":
            # fc to map latent space to embedding
            self.nec2vec = nn.Linear(self.latent_dim, self.embed_dim * self.max_seq_len)
        elif bottleneck == "mlp":
            h_layers_mlp = config.get("model::bottleneck::h_lays", 3)
            config_mlp = {
                "model::res_blocks": 0,
                "model::res_block_lays": 0,
                "model::h_layers": h_layers_mlp,
                "model::i_dim": self.embed_dim * self.max_seq_len,
                "model::latent_dim": self.latent_dim,
                "model::transition": "lin",
                "model::nlin": "leakyrelu",
                "model::dropout": self.dropout,
                "model::init_type": "kaiming_normal",
            }
            self.nec2vec = Decoder(config_mlp)

    def forward(self, z, mask=None):
        attn_scores = []  # not yet implemented, to prep interface

        # decompress
        y = self.nec2vec(z)
        y = y.view(z.shape[0], self.max_seq_len, self.embed_dim)
        # y = self.position_embeddings(y)

        # embedd positions
        positions = torch.arange(self.max_seq_len, device=y.device).unsqueeze(0)
        y = y + self.position_embeddings(positions).expand_as(y)
        # apply attention

        # y = self.encoder(y, mask)
        for ndx in range(self.N):
            y, scores = self.layers[ndx](y, mask)
            attn_scores.append(scores)

        # map back to original space.
        y = self.token_debeddings(y)
        #
        return y, attn_scores
