import torch
import torch.nn as nn
from .def_attn_components import (
    Embedder,
    get_clones,
    EncoderLayer,
    EmbedderNeuronGroup,
    EmbedderNeuronGroup_index,
)
from .def_encoder import Encoder

from .def_attn_embedder import AttnEmbedder

from einops import repeat


class EncoderTransformer(nn.Module):
    def __init__(self, config):
        super(EncoderTransformer, self).__init__()

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
        print(f"init attn encoder")
        # token embeddings
        if config.get("model::encoding", "weight") == "weight":
            # encode each weight separately
            self.max_seq_len = self.input_dim
            self.token_embeddings = Embedder(self.input_dim, self.embed_dim)
        elif config.get("model::encoding", "weight") == "neuron":
            # encode weights of one neuron together
            if config.get("model::index_dict", None) is None:
                self.max_seq_len = 9
                self.token_embeddings = EmbedderNeuronGroup(self.embed_dim)
            # attn embedder
            elif config.get("model::encoder") == "attn":
                print("## attention encoder -- use index_dict")
                index_dict = config.get("model::index_dict", None)
                d_embed = config.get("model::attn_embedder_dim")
                n_heads = config.get("model::attn_embedder_nheads")
                self.token_embeddings = AttnEmbedder(
                    index_dict,
                    d_model=int(self.embed_dim),
                    d_embed=d_embed,
                    n_heads=n_heads,
                )
                self.max_seq_len = self.token_embeddings.__len__()

            else:
                print("## encoder -- use index_dict")
                index_dict = config.get("model::index_dict", None)
                self.token_embeddings = EmbedderNeuronGroup_index(
                    index_dict, self.embed_dim
                )
                self.max_seq_len = self.token_embeddings.__len__()

        # compression token embedding
        self.compression_token = config.get("model::compression_token", False)
        if self.compression_token:
            self.comp_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
            # add sequence length of 1
            self.max_seq_len += 1

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

        # fc to map to latent space
        bottleneck = config.get("model::bottleneck", "linear")
        # compute input dimension to bottlneck
        if self.compression_token:
            bottleneck_input = self.embed_dim
        else:
            bottleneck_input = self.embed_dim * self.max_seq_len

        if bottleneck == "linear":
            self.vec2neck = nn.Sequential(
                nn.Linear(bottleneck_input, self.latent_dim), nn.Tanh()
            )
        elif bottleneck == "mlp":
            h_layers_mlp = config.get("model::bottleneck::h_lays", 3)
            config_mlp = {
                "model::res_blocks": 0,
                "model::res_block_lays": 0,
                "model::h_layers": h_layers_mlp,
                "model::i_dim": bottleneck_input,
                "model::latent_dim": self.latent_dim,
                "model::transition": "lin",
                "model::nlin": "leakyrelu",
                "model::dropout": self.dropout,
                "model::init_type": "kaiming_normal",
                "model::normalize_latent": True,
            }
            self.vec2neck = Encoder(config_mlp)

    def forward(self, x, mask=None):
        attn_scores = []  # not yet implemented, to prep interface
        # embedd weights
        x = self.token_embeddings(x)
        # add a compression token to the beginning of each sequence (dim = 1)
        if self.compression_token:
            b, n, _ = x.shape
            copm_tokens = repeat(self.comp_token, "() n d -> b n d", b=b)
            x = torch.cat((copm_tokens, x), dim=1)
        # embedd positions
        positions = torch.arange(self.max_seq_len, device=x.device).unsqueeze(0)
        x = x + self.position_embeddings(positions).expand_as(x)

        # pass through encoder
        # x = self.encoder(x, mask)
        for ndx in range(self.N):
            x, scores = self.layers[ndx](x, mask)
            attn_scores.append(scores)

        # compress to bottleneck
        if self.compression_token:
            # take only first part of the sequence / token
            x = x[:, 0, :]
        else:
            x = x.view(x.shape[0], x.shape[1] * x.shape[2])

        x = self.vec2neck(x)
        #
        return x, attn_scores
