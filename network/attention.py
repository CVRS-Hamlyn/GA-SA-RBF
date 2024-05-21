from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.checkpoint import checkpoint

layer_idx = 0

class MultiheadSeqAttention(nn.MultiheadAttention):
    """
    Multihead attention
    """

    def __init__(self, embed_dim, num_heads):
        super(MultiheadSeqAttention, self).__init__(embed_dim, num_heads, dropout=0.0, bias=True,
                                                         add_bias_kv=False, add_zero_attn=False,
                                                         kdim=None, vdim=None)

    def forward(self, fvec_c, fvec_seq, dis_enc_c=None, dis_enc_seq=None):
        """
        Multihead attention
        :feat_f: [B, L, C]
        :param attn_mask: mask to invalidate attention, -inf is used for invalid attention, [1,L]
        :param freq_enc: [L,C]
        :return: output value vector, attention with softmax (for debugging) and raw attention (used for last layer)
        """
        b, l, embed_dim = fvec_seq.size()
        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # project to get qkv
        # cross-attention
        _b = self.in_proj_bias
        _start = 0
        _end = embed_dim
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        q = F.linear(fvec_c, _w, _b)

        _b = self.in_proj_bias
        _start = embed_dim
        _end = None
        _w = self.in_proj_weight[_start:, :]
        if _b is not None:
            _b = _b[_start:]
        k, v = F.linear(fvec_seq, _w, _b).chunk(2, dim=-1)
      


        # project to find q_r, k_r
        if (dis_enc_c is not None) and (dis_enc_seq is not None):
            # compute k_r, q_r
            dis_enc_c = dis_enc_c.squeeze(0)
            dis_enc_seq = dis_enc_seq.squeeze(0)
            # input(freq_enc.shape)
            _start = 0
            _end = embed_dim
            _w_c = self.in_proj_weight[_start:_end, :]
            _b_c = self.in_proj_bias[_start:_end]
            q_r = F.linear(dis_enc_c, _w_c, _b_c)

            _w_seq = self.in_proj_weight[_end:2*_end, :]
            _b_seq = self.in_proj_bias[_end:2*_end]
            k_r = F.linear(dis_enc_seq, _w_seq, _b_seq)

        else:
            q_r = None
            k_r = None

        # scale query
        scaling = float(head_dim) ** -0.5
        q = q * scaling
        if q_r is not None:
            q_r = q_r * scaling

        # reshape
        q = q.contiguous().view(b, 1, self.num_heads, head_dim)  # BxHWxExC'
        k = k.contiguous().view(b, l, self.num_heads, head_dim)
        v = v.contiguous().view(b, l, self.num_heads, head_dim)

        if q_r is not None:
            q_r = q_r.contiguous().view(b, 1, self.num_heads, head_dim)# BxHWxExC'
        if k_r is not None:
            k_r = k_r.contiguous().view(b, l, self.num_heads, head_dim)

        # compute attn weight
        attn_feat = torch.einsum('bnec,bmec->benm', q, k)  # BxExHWxHW

        # add positional terms
        if (dis_enc_c is not None) and (dis_enc_seq is not None):
            # 0.3 s
            attn_feat_freq = torch.einsum('bnec,bvec->benv', q, k_r)  # BxExHWxHW'
            attn_freq_feat = torch.einsum('bmec,bvec->bemv', q_r, k)  # BxExHWxHW'
            # 0.1 s
            attn = attn_feat + attn_feat_freq + attn_freq_feat
        else:
            attn = attn_feat

        assert list(attn.size()) == [b, self.num_heads, 1, l]

        # raw attn
        raw_attn = attn

        # softmax
        attn = F.softmax(attn, dim=-1)

        # compute v, equivalent to einsum('',attn,v),
        # need to do this because apex does not support einsum when precision is mixed
        v_o = torch.bmm(attn.view(b * self.num_heads, 1, l),
                        v.permute(0, 2, 1, 3).reshape(b * self.num_heads, l, head_dim))  # BEx1xL, BExLxC -> BEx1xC
        assert list(v_o.size()) == [b * self.num_heads, 1, head_dim]
        v_o = v_o.reshape(b, self.num_heads, 1, head_dim).permute(0, 2, 1, 3).reshape(b, 1, embed_dim)
        v_o = F.linear(v_o, self.out_proj.weight, self.out_proj.bias)

        # # average attention weights over heads
        # attn = attn.sum(dim=1) / self.num_heads

        # # raw attn
        # raw_attn = raw_attn.sum(dim=1)

        return v_o


class Seq_Atten(nn.Module):

    def __init__(self, embed_channels, out_channels, n_heads):
        super().__init__()
        self.seq_attn = MultiheadSeqAttention(embed_channels, n_heads)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_channels),
            nn.Dropout(p=0.2),
            nn.Linear(embed_channels, 256),
            nn.LayerNorm(256),
            nn.Linear(256, out_channels)
        )
    def forward(self, fvec_c, fvec_seq, dis_enc_c=None, dis_enc_seq=None):
        f_o = self.seq_attn(fvec_c, fvec_seq, dis_enc_c, dis_enc_seq)
        out = self.mlp_head(f_o)

        return out


# def get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# class FrequencySelfAttnLayer(nn.Module):
#     """
#     Self attention layer
#     """

#     def __init__(self, hidden_dim, nhead):
#         super().__init__()
#         self.self_attn = MultiheadAttentionFrequency(hidden_dim, nhead)

#         self.norm1 = nn.LayerNorm(hidden_dim)

#     def forward(self, feat, freq=None):
#         """
#         :param feat: image feature [B,HW,C]
#         :param freq: frequency encoding [HW,C]
#         :return: updated image feature
#         """
#         feat2 = self.norm1(feat)

#         # torch.save(feat2, 'feat_self_attn_input_' + str(layer_idx) + '.dat')

#         feat2, attn_weight, _ = self.self_attn(feat2, freq_enc=freq)

#         # torch.save(attn_weight, 'self_attn_' + str(layer_idx) + '.dat')

#         feat = feat + feat2

#         return feat

    


# class Freq_Atten(nn.Module):
#     """
#     Freq_Atten computes self (intra image) and cross (inter image) attention
#     """

#     def __init__(self, hidden_dim, nhead, num_attn_layers):
#         super().__init__()

#         self_attn_layer = FrequencySelfAttnLayer(hidden_dim, nhead)
#         self.self_attn_layers = get_clones(self_attn_layer, num_attn_layers)

#         self.norm = nn.LayerNorm(hidden_dim)

#         self.hidden_dim = hidden_dim
#         self.nhead = nhead
#         self.num_attn_layers = num_attn_layers

#     def _alternating_attn(self, feat, freq_enc):
#         """
#         Alternate self and cross attention with gradient checkpointing to save memory
#         :param feat: image feature concatenated from left and right, [W,2HN,C]
#         :param freq_enc: positional encoding, [W,HN,C]
#         :param pos_indexes: indexes to slice positional encoding, [W,HN,C]
#         :param hn: size of HN
#         :return: attention weight [N,H,W,W]
#         """

#         global layer_idx
#         # alternating
#         for idx, self_attn in enumerate(self.self_attn_layers):
#             layer_idx = idx

#             # checkpoint self attn
#             def create_custom_self_attn(module):
#                 def custom_self_attn(*inputs):
#                     return module(*inputs)

#                 return custom_self_attn
#             feat = checkpoint(create_custom_self_attn(self_attn), feat, freq_enc)

#         layer_idx = 0
#         return feat

#     def forward(self, feat, freq_enc = None):
#         """
#         :param feat: feature descriptor at frequency domain, [B,C,H,W]
#         :param freq_enc: RBF frequency encoding, [HW, C]
#         :return: feat_attn, [B,C,H,W]
#         """

#         # flatten NxCxHxW to WxHNxC
#         bs, c, h, w = feat.shape

#         feat = feat.flatten(-2).permute(0, 2, 1)  # BxCxHxW -> BxCxHW -> BxHWxC

#         # compute attention
#         feat_attn = self._alternating_attn(feat, freq_enc)
#         feat_attn = feat_attn.permute(0, 2, 1).reshape(bs, c, h, w)  # NxHxWxW, dim=2 left image, dim=3 right image

#         return feat_attn