import warnings

warnings.filterwarnings("ignore")
import numpy as np
from glob import glob
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch import Tensor, nn
import math
from sklearn.manifold import TSNE
from typing import Tuple, Type
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Type
from torchvision.models import (
    convnext_base,
    convnext_small,
    convnext_tiny,
    swin_b,
    swin_v2_b,
    swin_v2_s,
    swin_v2_t,
    mobilenet_v3_large,
    efficientnet_v2_m,
)
import numpy as np
import os
import albumentations as albu
import argparse
import torchvision.models as models
from matplotlib import pyplot as plt
from tqdm import tqdm

dict_list = []

def compute_contrastive_loss(x, margin=1.0, eps=1e-6):
    """
    计算类内和类间距离的对比损失。

    Args:
        x: 输入张量，形状为(batch_size, 7, 256)
        margin: 类间距离的最小间隔
        eps: 防止除以零的小数

    Returns:
        loss: 损失值
    """
    batch_size, num_classes, feat_dim = x.shape

    # 计算类中心
    class_means = torch.mean(x, dim=0)  # (7, 256)

    # 类内距离：每个类内样本到中心的平均距离
    diff_intra = x - class_means.unsqueeze(0)  # 广播到(batch_size, 7, 256)
    squared_diff_intra = torch.sum(diff_intra**2, dim=2)  # (batch_size, 7)
    intra_loss = torch.mean(squared_diff_intra)  # 平均类内距离

    # 类间距离：所有类中心之间的距离
    # 扩展为(7,1,256)和(1,7,256)，计算差值矩阵
    diff_inter = class_means.unsqueeze(1) - class_means.unsqueeze(0)  # (7,7,256)
    squared_diff_inter = torch.sum(diff_inter**2, dim=2)  # (7,7)

    # 取上三角部分（不包括对角线）
    triu_indices = torch.triu_indices(num_classes, num_classes, offset=1)
    inter_distances = squared_diff_inter[triu_indices[0], triu_indices[1]]
    inter_loss = torch.mean(inter_distances)

    # 组合损失：类内距离 + max(margin - 类间距离, 0)
    # loss = intra_loss + torch.relu(margin - inter_loss)

    # 或使用比值形式
    loss = intra_loss / (inter_loss + eps)

    return loss


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class DynamicQueryModule(nn.Module):
    def __init__(self, transformer_dim, token_length, query_ratio):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.token_length = token_length
        self.query_ratio = query_ratio
        self.num_basic_queries = token_length * query_ratio

        # 基础查询嵌入
        self.basic_queries = nn.Embedding(token_length, transformer_dim)

        self.token_mlp = nn.Linear(transformer_dim, query_ratio * transformer_dim)

    def forward(self, query_weights):
        """
        image_features: 输入图像的特征，形状为 (batch_size, transformer_dim)
        """
        batch_size = query_weights.shape[0]

        # 基础查询嵌入
        basic_queries = self.basic_queries.weight.unsqueeze(0)
        query_embed = self.token_mlp(
            basic_queries
        )  # token_length * (query_ratio * transformer_dim)
        query_embed = query_embed.view(
            self.token_length * self.query_ratio, self.transformer_dim
        )

        # 动态查询生成
        dynamic_queries = []
        for b in range(batch_size):
            # 对每个样本进行加权组合
            weighted_queries = F.conv1d(
                query_embed.unsqueeze(0),
                query_weights[b].unsqueeze(-1),
                groups=self.token_length,
            )
            dynamic_queries.append(weighted_queries)

        dynamic_queries = torch.stack(
            dynamic_queries, dim=0
        )  # (batch_size, token_length, transformer_dim)

        return dynamic_queries.squeeze(1), basic_queries.expand(batch_size, -1, -1)


class Modulator(nn.Module):
    def __init__(
        self,
        transformer_dim,
        token_length,
        query_ratio,
        hidden_sizes=[128, 256, 512, 1024],
    ):
        super().__init__()

        self.token_length = token_length
        self.query_ratio = query_ratio

        self.mlp1 = FeatureMLP(
            input_dim=hidden_sizes[-1], output_dim=transformer_dim
        )  # 取f4作为调制

        # global max pooling branch
        self.gmp_branch = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(transformer_dim // 2, transformer_dim // 2, kernel_size=1),
        )
        # global avg pooling branch
        self.gap_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(transformer_dim // 2, transformer_dim // 2, kernel_size=1),
        )

        # 字典注意力生成器(其实相当于注意力融合)
        self.mlp2 = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim * 2),
            nn.ReLU(),
            nn.Linear(transformer_dim * 2, token_length * query_ratio),
        )

    def forward(self, x: torch.Tensor):
        bs, c, h, w = x.shape

        x = self.mlp1(x)  # (bs, h*w, c)

        # reshape -> (bs, c, h, w)
        x = x.permute(0, 2, 1).reshape(bs, -1, h, w)

        x1, x2 = x.chunk(2, dim=1)

        max_channel_attention = self.gmp_branch(x1)
        avg_channel_attention = self.gap_branch(x2)

        concated_channel_attention = torch.cat(
            [max_channel_attention, avg_channel_attention], dim=1
        )  # torch.Size([BS, 256, 1, 1])

        flatten_channel_attention = concated_channel_attention.flatten(
            1
        )  # torch.Size([BS, 256])

        fused_channel_attention = self.mlp2(
            flatten_channel_attention
        )  # (batch_size, token_length * query_ratio)

        fused_channel_attention = fused_channel_attention.view(
            x.shape[0], self.token_length, self.query_ratio
        )
        fused_channel_attention = F.softmax(
            fused_channel_attention, dim=-1
        )  # 归一化，确保权重和为1XX

        return fused_channel_attention


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        token_length,
        transformer_dim: 256,
        interactor: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
        query_ratio=4,
        use_aux=True,
        num_classes=None,
        has_conv=False,
        hidden_sizes=None,
        last_line=False,
        all_one=False,
        all_zero=False,
        only_static=False,
        has_interactor=True,
        has_contrastive_loss=False,
    ) -> None:

        super().__init__()
        self.has_contrastive_loss = has_contrastive_loss
        self.has_interactor = has_interactor
        self.use_aux = use_aux
        self.transformer_dim = transformer_dim
        self.interactor = interactor
        self.all_one = all_one
        self.all_zero = all_zero
        self.token_length = token_length
        self.only_static = only_static
        self.token = DynamicQueryModule(
            transformer_dim, token_length, query_ratio=query_ratio
        )
        self.feature_modulator = Modulator(
            transformer_dim, token_length, query_ratio, hidden_sizes=hidden_sizes
        )
        # self.token = nn.Embedding(token_length, transformer_dim)

        self.upsampler = nn.Sequential(
            nn.PixelShuffle(2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.PixelShuffle(2),
            activation(),
        )

        if not last_line:
            self.output_hypernetwork_mlps = MLP(
                transformer_dim, transformer_dim, transformer_dim // 16, 3
            )
            self.num_classes = num_classes
            if has_conv:
                self.output_head = nn.Linear(token_length, num_classes)
            self.has_conv = has_conv
        else:
            self.output_head = nn.Conv2d(
                transformer_dim // 16, num_classes, kernel_size=1
            )
        self.last_line = last_line

    def inference(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        image_embeddings - torch.Size([1, 256, 128, 128])
        image_pe - torch.Size([1, 256, 128, 128])
        """
        src = image_embeddings  ##### torch.Size([4, 256, 128, 128])
        pos_src = image_pe.expand(image_embeddings.size(0), -1, -1, -1)
        b, c, h, w = src.shape

        if self.has_interactor:
            # Run the transformer
            hs, src = self.interactor(
                src, pos_src, tokens
            )  ####### hs - torch.Size([BS, 11, 256]), src - torch.Size([BS, 16348, 256])
        else:
            src = src.flatten(2).permute(0, 2, 1)
            hs = tokens
        mask_token_out = hs[:, :, :]

        src = src.transpose(1, 2).view(b, c, h, w)  ##### torch.Size([4, 256, 128, 128])
        upscaled_embedding = self.upsampler(
            src
        )  ##### torch.Size([4, 32, 512, 512])
        hyper_in = self.output_hypernetwork_mlps(
            mask_token_out
        )  ##### torch.Size([1, 11, 32])
        if self.has_conv:
            hyper_in = hyper_in.permute(0, 2, 1)
            hyper_in = self.output_head(hyper_in)
            hyper_in = hyper_in.permute(0, 2, 1)

        b, c, h, w = upscaled_embedding.shape
        seg_output = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(
            b, -1, h, w
        )  ##### torch.Size([1, 11, 512, 512])
        # if seg_output.shape[1] > num_classes:
        #     seg_output = seg_output[:, :num_classes, :, :]
        return seg_output

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        image_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        image_embeddings - torch.Size([1, 256, 128, 128])
        image_pe - torch.Size([1, 256, 128, 128])
        image_feature -torch.Size([1,256,H,W])
        """
        if self.last_line:
            out = self.upsampler(image_embeddings)
            out = self.output_head(out)
            return out

        if self.all_one:
            token = torch.ones(
                size=(
                    image_embeddings.shape[0],
                    self.token_length,
                    self.transformer_dim,
                ),
                requires_grad=False,
            ).to(image_embeddings.device)
            output = self.inference(image_embeddings, image_pe, token)
            return output

        if self.all_zero:
            token = torch.zeros(
                size=(
                    image_embeddings.shape[0],
                    self.token_length,
                    self.transformer_dim,
                ),
                requires_grad=False,
            ).to(image_embeddings.device)
            output = self.inference(image_embeddings, image_pe, token)
            return output

        if self.only_static:
            token = self.token.basic_queries.weight.expand(
                image_embeddings.shape[0], -1, -1
            )
            output = self.inference(image_embeddings, image_pe, token)
            return output

        image_feature = self.feature_modulator.forward(image_feature)
        dynamic_token, basci_token = self.token.forward(image_feature)
        global dict_list
        dict_list.append(basci_token)
        dict_list.append(dynamic_token)
        contrastive_loss = None
        if self.has_contrastive_loss:
            contrastive_loss = compute_contrastive_loss(dynamic_token)
        dynamic_output = self.inference(image_embeddings, image_pe, dynamic_token)
        if self.training and self.use_aux:
            basci_output = self.inference(image_embeddings, image_pe, basci_token)
            if self.has_contrastive_loss:
                return dynamic_output, basci_output, contrastive_loss
            else:
                return dynamic_output, basci_output
        if self.training and self.has_contrastive_loss:
            return dynamic_output, contrastive_loss
        else:
            return dynamic_output


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class Interactor(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        do_attn1: bool = True,
        do_attn2: bool = True,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                InteractorBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    do_attention1=do_attn1,
                    do_attention2=do_attn2,
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding
        global dict_list
        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )
            dict_list.append(queries)

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class InteractorBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        do_attention1: bool = True,
        do_attention2: bool = True,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.do_attention1 = do_attention1
        self.do_attention2 = do_attention2
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        # if self.skip_first_layer_pe:
        #     queries = self.self_attn(q=queries, k=queries, v=queries)
        # else:
        #     q = queries + query_pe
        #     attn_out = self.self_attn(q=q, k=q, v=queries)
        #     queries = queries + attn_out
        # queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        # q = queries + query_pe
        # k = keys + key_pe
        # attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        # queries = queries + attn_out
        # queries = self.norm2(queries)

        queries = self.do_attn1(queries, keys, query_pe, key_pe)
        keys = self.do_attn2(queries, keys, query_pe, key_pe)

        # MLP block
        # mlp_out = self.mlp(queries)
        # queries = queries + mlp_out
        # queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        # q = queries + query_pe
        # k = keys + key_pe
        # attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        # keys = keys + attn_out
        # keys = self.norm4(keys)

        return queries, keys

    def do_attn1(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tensor:
        if not self.do_attention1:
            return queries
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)
        return queries

    def do_attn2(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tensor:
        if not self.do_attention2:
            return keys
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        return keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class FeatureMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, input_dim, output_dim=256):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class DynamicDictionaryLearning(nn.Module):
    def __init__(
        self,
        input_resolution,
        model,
        token_length,
        weights="IMAGENET1K_V1",
        pretrained=True,
        use_aux=True,
        do_attn1: bool = True,
        do_attn2: bool = True,
        l=2,
        embedding_dim=256,
        num_classes=None,
        has_conv=False,
        last_line=False,
        all_one=False,
        all_zero=False,
        only_static=False,
        has_aggregator=True,
        has_interactor=True,
        has_contrastive_loss=False,
        froze_backbone=False,
    ):
        """
        last line: 没有字典学习,走框架图最上面那条线.
        all one: 将字典替换为全1的向量.
        all zero: 将字典替换为全0的向量.
        only static: 只有静态字典.
        has aggregator: 是否将多模态特征,否则取最后一个
        has interactor: 是否有交互器.
        contrastive_loss: 对比损失.
        """
        super(DynamicDictionaryLearning, self).__init__()
        self.input_resolution = input_resolution
        self.model = model
        self.has_aggregator = has_aggregator

        if self.model == "swin_base":
            swin_v2 = swin_b(weights=weights)
            self.backbone = torch.nn.Sequential(*(list(swin_v2.children())[:-1]))
            self.target_layer_names = ["0.1", "0.3", "0.5", "0.7"]
            self.multi_scale_features = []

        if self.model == "swinv2_base":
            swin_v2 = swin_v2_b(weights=weights)
            self.backbone = torch.nn.Sequential(*(list(swin_v2.children())[:-1]))
            self.target_layer_names = ["0.1", "0.3", "0.5", "0.7"]
            self.multi_scale_features = []

        if self.model == "swinv2_small":
            swin_v2 = swin_v2_s(weights=weights)
            self.backbone = torch.nn.Sequential(*(list(swin_v2.children())[:-1]))
            self.target_layer_names = ["0.1", "0.3", "0.5", "0.7"]
            self.multi_scale_features = []

        if self.model == "swinv2_tiny":
            swin_v2 = swin_v2_t(weights=weights)
            self.backbone = torch.nn.Sequential(*(list(swin_v2.children())[:-1]))
            self.target_layer_names = ["0.1", "0.3", "0.5", "0.7"]
            self.multi_scale_features = []

        if self.model == "convnext_base":
            convnext = convnext_base(pretrained=pretrained)
            self.backbone = torch.nn.Sequential(*(list(convnext.children())[:-1]))
            self.target_layer_names = ["0.1", "0.3", "0.5", "0.7"]
            self.multi_scale_features = []

        if self.model == "convnext_small":
            convnext = convnext_small(pretrained=pretrained)
            self.backbone = torch.nn.Sequential(*(list(convnext.children())[:-1]))
            self.target_layer_names = ["0.1", "0.3", "0.5", "0.7"]
            self.multi_scale_features = []

        if self.model == "convnext_tiny":
            convnext = convnext_tiny(pretrained=pretrained)
            self.backbone = torch.nn.Sequential(*(list(convnext.children())[:-1]))
            self.target_layer_names = ["0.1", "0.3", "0.5", "0.7"]
            self.multi_scale_features = []

        if self.model == "resnet":
            resnet101 = models.resnet101(pretrained=pretrained)
            self.backbone = torch.nn.Sequential(*(list(resnet101.children())[:-1]))
            self.target_layer_names = ["4", "5", "6", "7"]
            self.multi_scale_features = []

        if self.model == "mobilenet":
            mobilenet = mobilenet_v3_large(pretrained=pretrained).features
            self.backbone = mobilenet
            self.target_layer_names = ["3", "6", "12", "16"]
            self.multi_scale_features = []

        if self.model == "efficientnet":
            efficientnet = efficientnet_v2_m(pretrained=pretrained).features
            self.backbone = efficientnet
            self.target_layer_names = ["2", "3", "5", "8"]
            self.multi_scale_features = []

        embed_dim = 1024
        out_chans = embedding_dim

        self.pe_layer = PositionEmbeddingRandom(out_chans // 2)

        for name, module in self.backbone.named_modules():
            if name in self.target_layer_names:
                module.register_forward_hook(self.save_features_hook(name))

        num_encoder_blocks = 4
        if self.model in ["swin_base", "swinv2_base", "convnext_base"]:
            hidden_sizes = [128, 256, 512, 1024]  ### Swin Base and ConvNext Base
        if self.model in ["resnet"]:
            hidden_sizes = [256, 512, 1024, 2048]  ### ResNet
        if self.model in [
            "swinv2_small",
            "swinv2_tiny",
            "convnext_small",
            "convnext_tiny",
        ]:
            hidden_sizes = [
                96,
                192,
                384,
                768,
            ]  ### Swin Small/Tiny and ConvNext Small/Tiny
        if self.model in ["mobilenet"]:
            hidden_sizes = [24, 40, 112, 960]  ### MobileNet
        if self.model in ["efficientnet"]:
            hidden_sizes = [48, 80, 176, 1280]  ### EfficientNet
        decoder_hidden_size = embedding_dim
        self.hidden_sizes = hidden_sizes
        mlps = []
        for i in range(num_encoder_blocks):
            mlp = FeatureMLP(input_dim=hidden_sizes[i], output_dim=embedding_dim)
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        self.decoder = Decoder(
            token_length=token_length,
            transformer_dim=embedding_dim,
            interactor=Interactor(
                depth=l,
                embedding_dim=embedding_dim,
                mlp_dim=2048,
                num_heads=8,
                do_attn1=do_attn1,
                do_attn2=do_attn2,
            ),
            use_aux=use_aux,
            num_classes=num_classes,
            has_conv=has_conv,
            hidden_sizes=hidden_sizes,
            last_line=last_line,
            all_one=all_one,
            all_zero=all_zero,
            only_static=only_static,
            has_interactor=has_interactor,
            has_contrastive_loss=has_contrastive_loss,
        )

        # The following 3 layers implement the ConvModule of the original implementation
        self.aggregator = nn.Conv2d(
            in_channels=decoder_hidden_size * num_encoder_blocks,
            out_channels=decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )

        if froze_backbone:
            self.backbone.requires_grad_(False)
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def save_features_hook(self, name):
        def hook(module, input, output):
            if self.model in [
                "swin_base",
                "swinv2_base",
                "swinv2_small",
                "swinv2_tiny",
            ]:
                self.multi_scale_features.append(
                    output.permute(0, 3, 1, 2).contiguous()
                )  ### Swin, Swinv2
            if self.model in [
                "convnext_base",
                "convnext_small",
                "convnext_tiny",
                "mobilenet",
                "efficientnet",
                "resnet",
            ]:
                self.multi_scale_features.append(
                    output
                )  ### ConvNext, ResNet, EfficientNet, MobileNet

        return hook

    def forward(self, x):
        global dict_list
        dict_list = []
        self.multi_scale_features.clear()

        _, _, h, w = x.shape
        features = self.backbone(x).squeeze()

        batch_size = self.multi_scale_features[-1].shape[0]
        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(self.multi_scale_features, self.linear_c):
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(
                batch_size, -1, height, width
            )

            # upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state,
                size=self.multi_scale_features[0].size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            all_hidden_states += (encoder_hidden_state,)
        if self.has_aggregator:
            fused_states = self.aggregator(
                torch.cat(all_hidden_states[::-1], dim=1)
            )  #### torch.Size([BS, 256, 128, 128])
        else:
            fused_states = encoder_hidden_state
        image_pe = self.pe_layer(
            (fused_states.shape[2], fused_states.shape[3])
        ).unsqueeze(0)
        seg_output = self.decoder(
            image_embeddings=fused_states,
            image_pe=image_pe,
            image_feature=self.multi_scale_features[-1],
        )

        dict_list = torch.stack(dict_list)
        return dict_list





class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, filenames):
        self.filenames = filenames
        self.transformers = albu.Normalize()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx])
        img = np.array(img)
        aug = self.transformers(image=img)
        img = aug["image"]
        img = np.transpose(img, (2, 0, 1))
        return img


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default="checkpoints/grass/ours_l3/ours_l3.ckpt")
    parser.add_argument("--img_dir", default="data/grass/img_dir/*")
    parser.add_argument("--batch_size", default=16)
    parser.add_argument("--save_dir", default="data/grass/tsne")
    parser.add_argument("--token_length", default=5)
    parser.add_argument("--l", default=3)
    parser.add_argument("--suffix", default=".png")
    args = parser.parse_args()
    return args


import pandas as pd

def plot_tsne(
    data,
    labels=None,  # 修改：保留用户传入的 labels
    perplexity=30,
    learning_rate=200,
    n_iter=2000,
    random_state=42,
    title="t-SNE Visualization",
    args=None,
    filename="tsne.png",
    use_cache=True,  # 新增：是否使用缓存
    cmap="tab10"     # 新增：指定颜色映射
):
    # 生成 CSV 文件名（与图片同名，扩展名不同）
    csv_filename = os.path.splitext(filename)[0] + ".csv"
    csv_path = os.path.join(args.save_dir, csv_filename)
    
    # 尝试读取缓存数据
    if use_cache and os.path.exists(csv_path):
        print(f"Loading cached t-SNE data from {csv_path}")
        df = pd.read_csv(csv_path)
        reduced_data = df[["x", "y"]].values
        labels = df["label"].values
    else:
        # 预处理数据
        dim = data.shape[-1]
        data = data.reshape(-1, dim)
        bs = data.shape[0]
        
        # 修复标签生成逻辑（仅在未提供时生成）
        if labels is None:
            labels = np.array([i for i in range(args.token_length)])
            labels = np.tile(labels, (bs + len(labels) - 1) // len(labels))[:bs]
        else:
            labels = np.asarray(labels).flatten()
            assert len(labels) == bs, "Labels length does not match data size"

        # 计算 t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            random_state=random_state,
        )
        reduced_data = tsne.fit_transform(data)
        
        # 保存数据到 CSV
        df = pd.DataFrame({
            "x": reduced_data[:, 0],
            "y": reduced_data[:, 1],
            "label": labels
        })
        df.to_csv(csv_path, index=False)
        print(f"Saved t-SNE data to {csv_path}")

    # 绘图部分保持不变（使用缓存数据或新数据）
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    # 修改颜色映射
    scatter = plt.scatter(
        reduced_data[:, 0], 
        reduced_data[:, 1], 
        c=labels, 
        cmap=cmap,  # 使用参数指定的颜色映射
        alpha=0.7, 
        s=2
    )
    
    # 保存和清理
    plt.axis("off")
    plt.savefig(os.path.join(args.save_dir, filename), bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # 距离计算保持不变...


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_model(args, device):
    model = DynamicDictionaryLearning(
        model="convnext_base",
        token_length=args.token_length,
        l=args.l,
    )
    weights = torch.load(args.ckpt_path)["state_dict"]
    state_dict = {}
    for k, v in weights.items():
        state_dict[k[4:]] = v
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    return model


def get_loader(args):
    filenames = glob(os.path.join(args.img_dir, "*"))
    dataset = SimpleDataset(filenames)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )
    return loader


if __name__ == "__main__":
    args = get_args()
    device = get_device()
    model = get_model(args, device)

    loader = get_loader(args)
    os.makedirs(args.save_dir, exist_ok=True)

    dict_list_true = [None for _ in range(args.l + 2)]
    with torch.no_grad():
        for img in tqdm(loader):
            img = img.to(device)
            output = model(img).cpu().numpy()  # [depth+2, batch_size, token_length,256]
            for i in range(args.l + 2):
                if dict_list_true[i] is None:
                    dict_list_true[i] = output[i]
                else:
                    dict_list_true[i] = np.concatenate(
                        [dict_list_true[i], output[i]], axis=0
                    )

    titles = ["Static Dictionary", "Dynamic Dictionary"] + [
        f"D{i}" for i in range(1,args.l+1)
    ]
    filenames = [str(i) for i in range(-1, args.l + 1)]
    for i in range(args.l + 2):
        print(titles[i], end=" ")
        plot_tsne(
            dict_list_true[i],
            title=titles[i],
            args=args,
            filename=filenames[i] + args.suffix,
        )
