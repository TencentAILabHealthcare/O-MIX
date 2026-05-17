import gc
import math
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
# from flash_attn.modules.mha import MHA
from torch import Tensor, nn
from torch.distributions import Bernoulli
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tqdm import trange

flash_attn_available = True

# from omix.model.performer_pytorch_adapter import Performer
from omix.model.performer_pytorch import Performer

from .dsbn import DomainSpecificBatchNorm1d
from .grad_reverse import grad_reverse


class PerformerModel(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        nlayers_cls: int = 3,
        n_cls: int = 1,
        vocab: Any = None,
        dropout: float = 0.5,
        pad_token: str = "<pad>",
        pad_value: int = -2,
        do_mvc: bool = False,
        do_dab: bool = False,
        use_batch_labels: bool = False,
        use_modality_tokens: bool = False,
        num_batch_labels: Optional[int] = None,
        domain_spec_batchnorm: Union[bool, str] = False,
        input_emb_style: str = "category",
        mvc_decoder_style: str = "inner product",
        n_input_bins: Optional[int] = None,
        cell_emb_style: str = "cls",
        explicit_zero_prob: bool = False,
        use_fast_transformer: bool = False,
        fast_transformer_backend: str = "flash",
        pre_norm: bool = False,
        feature_redraw_interval=1000,
        auto_check_redraw=True,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.do_dab = do_dab
        self.use_batch_labels = use_batch_labels
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.input_emb_style = input_emb_style
        self.cell_emb_style = cell_emb_style
        self.explicit_zero_prob = explicit_zero_prob
        self.norm_scheme = "pre" if pre_norm else "post"
        if self.input_emb_style not in ["category", "continuous", "scaling"]:
            raise ValueError(
                f"input_emb_style should be one of category, continuous, scaling, "
                f"got {input_emb_style}"
            )
        if cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")
        if use_fast_transformer:
            if not flash_attn_available:
                warnings.warn(
                    "flash-attn is not installed, using pytorch transformer instead. "
                    "Set use_fast_transformer=False to avoid this warning. "
                    "Installing flash-attn is highly recommended."
                )
                use_fast_transformer = False
        self.use_fast_transformer = use_fast_transformer

        # TODO: add dropout in the GeneEncoder
        self.encoder = GeneEncoder(ntoken, d_model, padding_idx=vocab[pad_token])

        # Value Encoder, NOTE: the scaling style is also handled in _encode method
        if input_emb_style == "continuous":
            self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        elif input_emb_style == "category":
            assert n_input_bins > 0
            self.value_encoder = CategoryValueEncoder(
                n_input_bins, d_model, padding_idx=pad_value
            )
        else:
            self.value_encoder = nn.Identity()  # nn.Softmax(dim=1)
            # TODO: consider row-wise normalization or softmax
            # TODO: Correct handle the mask_value when using scaling

        # Batch Encoder
        if use_batch_labels:
            self.batch_encoder = BatchLabelEncoder(num_batch_labels, d_model)

        if domain_spec_batchnorm is True or domain_spec_batchnorm == "dsbn":
            use_affine = True if domain_spec_batchnorm == "do_affine" else False
            print(f"Use domain specific batchnorm with affine={use_affine}")
            self.dsbn = DomainSpecificBatchNorm1d(
                d_model, num_batch_labels, eps=6.1e-5, affine=use_affine
            )
        elif domain_spec_batchnorm == "batchnorm":
            print("Using simple batchnorm instead of domain specific batchnorm")
            self.bn = nn.BatchNorm1d(d_model, eps=6.1e-5)

        if use_fast_transformer:
            if fast_transformer_backend == "linear":
                self.transformer_encoder = FastTransformerEncoderWrapper(
                    d_model, nhead, d_hid, nlayers, dropout
                )
            elif fast_transformer_backend == "flash":
                encoder_layers = FlashTransformerEncoderLayer(
                    d_model,
                    nhead,
                    d_hid,
                    dropout,
                    batch_first=True,
                    norm_scheme=self.norm_scheme,
                )
                self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        else:
            # encoder_layers = TransformerEncoderLayer(
            #     d_model, nhead, d_hid, dropout, batch_first=True
            # )
            # self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
            
            ff_mult = d_hid // d_model

            self.transformer_encoder = Performer(
                dim=d_model,
                depth=nlayers,
                heads=nhead,
                dim_head=d_model // nhead,
                ff_mult=ff_mult,
                ff_dropout=dropout,
                attn_dropout=dropout,
                causal=False,  # 设置为 False，因为我们用作编码器
                feature_redraw_interval=feature_redraw_interval,  
                auto_check_redraw=auto_check_redraw,
            )

        self.cls_decoder = ClsDecoder(d_model, n_cls, nlayers=nlayers_cls)

        if do_dab:
            self.grad_reverse_discriminator = AdversarialDiscriminator(
                d_model,
                n_cls=num_batch_labels,
                reverse_grad=True,
            )

        self.decoder = ExprDecoder(
            d_model,
            explicit_zero_prob=explicit_zero_prob,
            use_modality_tokens=use_modality_tokens,
        )

        self.mvc_decoder = MVCDecoder(
            d_model,
            arch_style=mvc_decoder_style,
            explicit_zero_prob=explicit_zero_prob,
            use_modality_tokens=use_modality_tokens,
        )


        self.sim = Similarity(temp=0.5)  # TODO: auto set temp

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # TODO: check if this initialization is helpful and shall we apply to all?
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)

    def _encode(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        batch_labels: Optional[Tensor] = None,  # (batch,)
        gene_token_embs: Optional[Tensor] = None,
        output_attention: bool = False
    ):
        self._check_batch_labels(batch_labels)

        if gene_token_embs is None:
            src = self.encoder(src)  # (batch, seq_len, embsize)
        else:
            src = gene_token_embs
        # self.cur_gene_token_embs = src

        values = self.value_encoder(values)  # (batch, seq_len, embsize)
        if self.input_emb_style == "scaling":
            values = values.unsqueeze(2)
            total_embs = src * values
        else:
            total_embs = src + values

        if getattr(self, "dsbn", None) is not None:
            batch_label = int(batch_labels[0].item())
            total_embs = self.dsbn(total_embs.permute(0, 2, 1), batch_label).permute(
                0, 2, 1
            )  # the batch norm always works on dim 1
        elif getattr(self, "bn", None) is not None:
            total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        mask = ~src_key_padding_mask

        if output_attention:
            output, attn_weights = self.transformer_encoder(
                total_embs, mask=mask, output_attentions=output_attention
            )
            return output, attn_weights  # (batch, seq_len, embsize)
        else:
            output = self.transformer_encoder(
                total_embs, mask=mask
            )
            return output

    def _get_cell_emb_from_layer(
        self, layer_output: Tensor, weights: Tensor = None
    ) -> Tensor:
        """
        Args:
            layer_output(:obj:`Tensor`): shape (batch, seq_len, embsize)
            weights(:obj:`Tensor`): shape (batch, seq_len), optional and only used
                when :attr:`self.cell_emb_style` is "w-pool".

        Returns:
            :obj:`Tensor`: shape (batch, embsize)
        """
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]  # (batch, embsize)
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)

        return cell_emb

    def _check_batch_labels(self, batch_labels: Tensor) -> None:
        if self.use_batch_labels or self.domain_spec_batchnorm:
            assert batch_labels is not None
        elif batch_labels is not None:
            raise ValueError(
                "batch_labels should only be provided when `self.use_batch_labels`"
                " or `self.domain_spec_batchnorm` is True"
            )
    def get_attn_weights(self, attn_factors_list, mask):
        # 2. 提取最后一层的因子 (Last Layer Extraction)
        # attn_factors_list[-1] 是最后一层的 (q_prime, k_prime)
        # q_prime, k_prime 维度: [Batch, Heads, Seq_Len, Features]
        last_q_prime, last_k_prime = attn_factors_list[-1]
        
        # 3. 提取 CLS Token 的 Query (Index 0)
        # 维度变成: [Batch, Heads, 1, Features]
        cls_q = last_q_prime[:, :, 0:1, :]
        
        # 4. 计算注意力分数 (Dot Product)
        # [B, H, 1, F] @ [B, H, F, L] -> [B, H, 1, L]
        # 这一步计算量很小，不会 OOM
        raw_scores = torch.matmul(cls_q, last_k_prime.transpose(-1, -2))
        
        # 5. 聚合多头 (Average Heads)
        # [B, H, 1, L] -> [B, 1, L]
        raw_scores = raw_scores.mean(dim=1)
        
        # 去掉维度 1 -> [B, L]
        attn_weights = raw_scores.squeeze(1)
        
        # 6. 处理 Padding Mask
        # mask 为 True 的地方通常是 Padding (取决于你的定义)
        # 如果 mask 是 boolean (True=Padding)，则:
        if mask is not None:
            # Performer 的 Kernel 都是正数，所以我们可以把 padding 设为 0
            attn_weights = attn_weights.masked_fill(mask, 0.0)
            
        # 7. 归一化 (Normalization)
        # 使得权重和为 1，方便解释 "Importance"
        # 加上 1e-10 防止全 0 除法
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-10)
        return attn_weights

    def forward(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        return_embedding: bool=False,
        return_seq_embedding: bool=False,
        return_transformer_output: bool = False,  # <--- 新增参数
        output_attention: bool = False
    ):
        """
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]
            batch_labels (:obj:`Tensor`): batch labels, shape [batch_size]
            CLS (:obj:`bool`): if True, return the celltype classification objective
                (CLS) output
            CCE (:obj:`bool`): if True, return the contrastive cell embedding objective
                (CCE) output
            MVC (:obj:`bool`): if True, return the masked value prediction for cell
                embedding MVC output
            ECS (:obj:`bool`): if True, return the elastic cell similarity objective
                (ECS) output.

        Returns:
            dict of output Tensors.
        """
        # src gene_ids tokens: (bs, seq_len)
        # values gene values: (bs, seq_len)
        # transformer_output: (bs, seq_len, 512)
        # gene_token_embs = self.encoder(src)
        # if output_attention:
        #     transformer_output, attn_weights = self._encode(src, values, src_key_padding_mask, gene_token_embs=gene_token_embs)
        # else:
        #     transformer_output = self._encode(src, values, src_key_padding_mask, gene_token_embs=gene_token_embs)
        # if output_attention and return_transformer_output:
        #     return return_transformer_output, attn_weights
        
        # if return_transformer_output:
        #     return transformer_output

        # if return_seq_embedding:
        #     return transformer_output, gene_token_embs

        # # cell_emb: (bs, 512)
        # cell_emb = self._get_cell_emb_from_layer(transformer_output, values)

        # if return_embedding:
        #     return cell_emb


        # # risk: (bs, n_cls)
        # risk = self.cls_decoder(cell_emb)

        # return risk
        # 1. 基础 Embedding 提取
        gene_token_embs = self.encoder(src)
        
        # 2. Transformer 编码阶段
        # 只有在明确需要 attention 时才请求它
        encode_out = self._encode(
            src, values, src_key_padding_mask, 
            gene_token_embs=gene_token_embs, 
            output_attention=output_attention
        )
        
        # 解析编码器输出
        if output_attention:
            transformer_output, attn_factors_list = encode_out
            attn_weights = self.get_attn_weights(attn_factors_list, src_key_padding_mask)
            
        else:
            transformer_output = encode_out

        # --- 开始逻辑判断返回 (优先级排序) ---

        # 情况 A: return_seq_embedding (独占)
        if return_seq_embedding:
            return transformer_output, gene_token_embs

        # 情况 B: return_transformer_output 与 output_attention (可组合)
        if return_transformer_output:
            if output_attention:
                return transformer_output, attn_weights
            return transformer_output

        # 情况 C: return_embedding (独占)
        # 只有当前面两个都不满足时，才计算 cell_emb，节省算力
        cell_emb = self._get_cell_emb_from_layer(transformer_output, values)
        if return_embedding:
            return cell_emb

        # 情况 D: 默认返回 risk
        risk = self.cls_decoder(cell_emb)
        return risk

    def decode_only(self, transformer_output, gene_token_embs):
        output = {}
        # 包括decode出来的expression + zero_probs (optional)
        mlm_output = self.decoder(
            transformer_output
        )
        output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)

        # # cell_emb: (bs, 512)
        cell_emb = self._get_cell_emb_from_layer(transformer_output)

        mvc_output = self.mvc_decoder(
            cell_emb, gene_token_embs
        )
        output["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)

        return output
    
    def decode_only_for_mlm(self, transformer_output):
        output = {}
        # 包括decode出来的expression + zero_probs (optional)
        mlm_output = self.decoder(
            transformer_output
        )
        output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)
        return output

    def encode_batch(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_size: int,
        batch_labels: Optional[Tensor] = None,
        output_to_cpu: bool = True,
        time_step: Optional[int] = None,
        return_np: bool = False,
    ) -> Tensor:
        """
        Args:
            src (Tensor): shape [N, seq_len]
            values (Tensor): shape [N, seq_len]
            src_key_padding_mask (Tensor): shape [N, seq_len]
            batch_size (int): batch size for encoding
            batch_labels (Tensor): shape [N, n_batch_labels]
            output_to_cpu (bool): whether to move the output to cpu
            time_step (int): the time step index in the transformer output to return.
                The time step is along the second dimenstion. If None, return all.
            return_np (bool): whether to return numpy array

        Returns:
            output Tensor of shape [N, seq_len, embsize]
        """
        N = src.size(0)
        device = next(self.parameters()).device

        # initialize the output tensor
        array_func = np.zeros if return_np else torch.zeros
        float32_ = np.float32 if return_np else torch.float32
        shape = (
            (N, self.d_model)
            if time_step is not None
            else (N, src.size(1), self.d_model)
        )
        outputs = array_func(shape, dtype=float32_)

        for i in trange(0, N, batch_size):
            raw_output = self._encode(
                src[i : i + batch_size].to(device),
                values[i : i + batch_size].to(device),
                src_key_padding_mask[i : i + batch_size].to(device),
                batch_labels[i : i + batch_size].to(device)
                if batch_labels is not None
                else None,
            )
            output = raw_output.detach()
            if output_to_cpu:
                output = output.cpu()
            if return_np:
                output = output.numpy()
            if time_step is not None:
                output = output[:, time_step, :]
            outputs[i : i + batch_size] = output

        return outputs


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


class FastTransformerEncoderWrapper(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.fast_transformer_encoder = self.build_fast_transformer_encoder(
            d_model, nhead, d_hid, nlayers, dropout
        )

    @staticmethod
    def build_fast_transformer_encoder(
        d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float
    ) -> nn.Module:
        from fast_transformers.builders import TransformerEncoderBuilder

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model must be divisible by nhead, "
                f"got d_model={d_model} and nhead={nhead}"
            )
        builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=nlayers,
            n_heads=nhead,
            query_dimensions=d_model // nhead,
            value_dimensions=d_model // nhead,
            feed_forward_dimensions=d_hid,
            attention_type="linear",
            attention_dropout=dropout,
            dropout=dropout,
            activation="gelu",
        )
        assert builder.attention_type == "linear"
        return builder.get()

    @staticmethod
    def build_length_mask(
        src: Tensor,
        src_key_padding_mask: torch.BoolTensor,
    ) -> "LengthMask":
        from fast_transformers.masking import LengthMask

        seq_len = src.shape[1]
        num_paddings = src_key_padding_mask.sum(dim=1)
        actual_seq_len = seq_len - num_paddings  # (N,)
        length_mask = LengthMask(actual_seq_len, max_len=seq_len, device=src.device)

        if src_key_padding_mask[length_mask.bool_matrix].sum() != 0:
            raise ValueError(
                "Found padding tokens in the middle of the sequence. "
                "src_key_padding_mask and length_mask are not compatible."
            )
        return length_mask

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: torch.BoolTensor,
    ) -> Tensor:
        """
        Args:
            src: Tensor, shape [N, seq_len, embsize]
            src_key_padding_mask: Tensor, shape [N, seq_len]

        Returns:
            output Tensor of shape [N, seq_len, embsize]
        """
        if src_key_padding_mask.shape != src.shape[:2]:
            raise ValueError(
                f"src_key_padding_mask shape {src_key_padding_mask.shape} "
                f"does not match first two dims of src shape {src.shape[:2]}"
            )

        if src_key_padding_mask.dtype != torch.bool:
            raise ValueError(
                f"src_key_padding_mask needs to be of type torch.bool, "
                f"got {src_key_padding_mask.dtype}"
            )

        length_mask = self.build_length_mask(src, src_key_padding_mask)
        output = self.fast_transformer_encoder(src, length_mask=length_mask)
        return output


# class FlashTransformerEncoderLayer(nn.Module):
#     r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
#     The class is modified from torch.nn.TransformerEncoderLayer to support the
#     FlashAttention.
#
#     Args:
#         d_model: the number of expected features in the input (required).
#         nhead: the number of heads in the multiheadattention models (required).
#         dim_feedforward: the dimension of the feedforward network model (default=2048).
#         dropout: the dropout value (default=0.1).
#         activation: the activation function of intermediate layer, relu or gelu (default=relu).
#         layer_norm_eps: the eps value in layer normalization components (default=1e-5).
#         batch_first: If ``True``, then the input and output tensors are provided
#             as (batch, seq, feature). Default: ``False``.
#
#     Examples::
#         >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
#         >>> src = torch.rand(10, 32, 512)
#         >>> out = encoder_layer(src)
#
#     Alternatively, when ``batch_first`` is ``True``:
#         >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
#         >>> src = torch.rand(32, 10, 512)
#         >>> out = encoder_layer(src)
#     """
#     __constants__ = ["batch_first"]
#
#     def __init__(
#         self,
#         d_model,
#         nhead,
#         dim_feedforward=2048,
#         dropout=0.1,
#         activation="relu",
#         layer_norm_eps=1e-5,
#         batch_first=True,
#         device=None,
#         dtype=None,
#         norm_scheme="post",  # "pre" or "post"
#     ) -> None:
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         self.self_attn = MHA(
#             embed_dim=d_model,
#             num_heads=nhead,
#             dropout=dropout,
#             use_flash_attn=True,
#             # device=factory_kwargs.get('device'),
#             # dtype=factory_kwargs.get('dtype'),
#             **factory_kwargs,
#         )
#         # Version compatibility workaround
#         if not hasattr(self.self_attn, "batch_first"):
#             self.self_attn.batch_first = batch_first
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
#
#         self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#
#         self.activation = self._get_activation_fn(activation)
#         self.norm_scheme = norm_scheme
#         if self.norm_scheme not in ["pre", "post"]:
#             raise ValueError(f"norm_scheme should be pre or post, not {norm_scheme}")
#
#     @staticmethod
#     def _get_activation_fn(activation):
#         if activation == "relu":
#             return F.relu
#         elif activation == "gelu":
#             return F.gelu
#
#         raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
#
#     def __setstate__(self, state):
#         if "activation" not in state:
#             state["activation"] = F.relu
#         super().__setstate__(state)
#
#     def forward(
#         self,
#         src: Tensor,
#         src_mask: Optional[Tensor] = None,
#         src_key_padding_mask: Optional[Tensor] = None,
#         **kwargs,
#     ) -> Tensor:
#         r"""Pass the input through the encoder layer.
#
#         Args:
#             src: the sequence to the encoder layer (required).
#             src_mask: the mask for the src sequence (optional).
#             src_key_padding_mask: the mask for the src keys per batch (optional).
#
#         Shape:
#             see the docs in Transformer class.
#         """
#         if src_mask is not None:
#             raise ValueError("FlashTransformerEncoderLayer does not support src_mask")
#
#         if src_key_padding_mask is None:
#             src_key_padding_mask_ = None
#         elif not src_key_padding_mask.any().item():
#             src_key_padding_mask_ = None
#         else:
#             if src_key_padding_mask.dtype != torch.bool:
#                 src_key_padding_mask = src_key_padding_mask.bool()
#             # NOTE: the FlashMHA uses mask 0 for padding tokens, which is the opposite
#             src_key_padding_mask_ = ~src_key_padding_mask
#
#         if self.norm_scheme == "pre":
#             src = self.norm1(src)
#             src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask_)[0]
#             src = src + self.dropout1(src2)
#             src = self.norm2(src)
#             src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#             src = src + self.dropout2(src2)
#         else:
#             src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask_)[0]
#             src = src + self.dropout1(src2)
#             src = self.norm1(src)
#             src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#             src = src + self.dropout2(src2)
#             src = self.norm2(src)
#
#         return src
class FlashTransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    The class is modified from torch.nn.TransformerEncoderLayer to support the
    FlashAttention.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=True,
        device=None,
        dtype=None,
        norm_scheme="post",  # "pre" or "post"
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MHA(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            use_flash_attn=True,
            # device=factory_kwargs.get('device'),
            # dtype=factory_kwargs.get('dtype'),
            **factory_kwargs,
        )
        # Version compatibility workaround
        if not hasattr(self.self_attn, "batch_first"):
            self.self_attn.batch_first = batch_first
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if self.norm_scheme not in ["pre", "post"]:
            raise ValueError(f"norm_scheme should be pre or post, not {norm_scheme}")

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False,
            **kwargs,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
                 Shape: (batch, seq_len, embed_dim).
            src_mask: the mask for the src sequence (optional).
                      FlashAttention does not support this, will raise error.
            src_key_padding_mask: the mask for the src keys per batch (optional).
                                  Shape: (batch, seq_len).

        Shape:
            see the docs in Transformer class.
        """
        if src_mask is not None:
            raise ValueError("FlashTransformerEncoderLayer does not support `src_mask`.")

        # 预先处理 Pre-Norm
        if self.norm_scheme == "pre":
            src_normalized = self.norm1(src)
        else:
            src_normalized = src

        has_padding = src_key_padding_mask is not None and src_key_padding_mask.any().item()

        # --- Attention 计算 ---
        if has_padding:
            # --- 情况1: 输入序列包含 Padding ---
            valid_token_mask = ~src_key_padding_mask.bool()
            src_unpadded = src_normalized[valid_token_mask]

            actual_lengths = valid_token_mask.sum(dim=1, dtype=torch.int32)
            cu_seqlens = torch.cat(
                [torch.zeros(1, dtype=torch.int32, device=src.device),
                 torch.cumsum(actual_lengths, dim=0, dtype=torch.int32)]
            )
            max_seqlen = src.size(1)

            context_unpadded = self.self_attn(
                src_unpadded,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                **kwargs
            )[0]
            context_unpadded = context_unpadded.to(src_normalized.dtype)
            context = torch.zeros_like(src_normalized)
            context[valid_token_mask] = context_unpadded

        else:
            # --- 情况2: 输入序列没有 Padding ---
            batch_size, seq_len, _ = src_normalized.shape
            src_unpadded = src_normalized.reshape(-1, src_normalized.size(-1))
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * seq_len, step=seq_len, dtype=torch.int32, device=src.device
            )
            max_seqlen = seq_len

            context_unpadded = self.self_attn(
                src_unpadded,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                **kwargs
            )[0]

            context_unpadded = context_unpadded.to(src_normalized.dtype)
            context = context_unpadded.reshape(batch_size, seq_len, -1)

        if self.norm_scheme == "pre":
            src = src + self.dropout1(context)
            src_ffn = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
            src = src + self.dropout2(src_ffn)
        else:  # Post-Norm
            src = self.norm1(src + self.dropout1(context))
            src_ffn = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = self.norm2(src + self.dropout2(src_ffn))

        return src


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # TODO: test using actual embedding layer if input is categorical
        # expand last dimension
        x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class CategoryValueEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        cc = np.array(x.cpu())
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class BatchLabelEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.enc_norm(x)
        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        use_modality_tokens: bool = False,
    ):
        super().__init__()
        d_in = d_model * 2 if use_modality_tokens else d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.fc(x).squeeze(-1)  # (batch, seq_len)

        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_logits = self.zero_logit(x).squeeze(-1)  # (batch, seq_len)
        zero_probs = torch.sigmoid(zero_logits)
        return dict(pred=pred_value, zero_probs=zero_probs)
        # TODO: note that the return currently is only for training. Since decoder
        # is not used in the test setting for the integration task, the eval/inference
        # logic is not implemented yet. However, remember to implement it when
        # the decoder is used in any test setting. The inference logic will need
        # to sample from the bernoulli distribution with the zero_probs.


class ClsDecoder(nn.Module):
    """
    Decoder for classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class MVCDecoder(nn.Module):
    """
    Decoder for the masked value prediction for cell embeddings.
    """

    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
        explicit_zero_prob: bool = False,
        use_modality_tokens: bool = False,
    ) -> None:
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "concat query" or 3. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
        """
        super().__init__()
        d_in = d_model * 2 if use_modality_tokens else d_model
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if explicit_zero_prob:  # by default, gene-wise prob rate
                self.W_zero_logit = nn.Linear(d_model, d_in)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob

    def forward(
        self, cell_emb: Tensor, gene_embs: Tensor
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)
            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model)
        """
        gene_embs = gene_embs.detach() if self.do_detach else gene_embs
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(2)  # (batch, embsize, 1)
            # the pred gene expr values, # (batch, seq_len)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
            if not self.explicit_zero_prob:
                return dict(pred=pred_value)
            # zero logits need to based on the cell_emb, because of input exprs
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred=pred_value, zero_probs=zero_probs)
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            # expand cell_emb to (batch, seq_len, embsize)
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)

            h = self.hidden_activation(
                self.fc1(torch.cat([cell_emb, query_vecs], dim=2))
            )
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)

            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)


class AdversarialDiscriminator(nn.Module):
    """
    Discriminator for the adversarial training for batch correction.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.LeakyReLU,
        reverse_grad: bool = False,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)
        self.reverse_grad = reverse_grad

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0)
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)
